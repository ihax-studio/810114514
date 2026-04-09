"""
iHax Memory System
==================
LLMに永続記憶を与える3層メモリアーキテクチャ。

┌─────────────────────────────────────────────┐
│          Short-term (コンテキスト内)           │
│  直近N回の会話をそのままプロンプトに含める      │
│  容量: context_length に依存                  │
│  速度: 即時 (追加コストなし)                   │
├─────────────────────────────────────────────┤
│          Mid-term (セマンティック検索)          │
│  SQLite + Embedding で関連する過去の会話を検索  │
│  容量: ディスク容量次第 (事実上無限)           │
│  速度: ~50ms (ローカルembedding)              │
├─────────────────────────────────────────────┤
│          Long-term (圧縮記憶)                  │
│  古い会話をLLMで要約 → ユーザープロファイル     │
│  容量: 小 (要約済み)                          │
│  速度: 即時 (事前計算済み)                     │
└─────────────────────────────────────────────┘

依存: pip install numpy sentence-transformers
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).parent / "data" / "memory.db"


def _get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")  # 並行読み書き対応
    conn.execute("PRAGMA synchronous=NORMAL")
    _init_tables(conn)
    return conn


def _init_tables(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,          -- 'user' or 'assistant'
            content TEXT NOT NULL,
            embedding BLOB,              -- numpy float32 配列
            created_at REAL NOT NULL,
            metadata TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            summary TEXT NOT NULL,
            turn_range TEXT NOT NULL,     -- '1-50' のような範囲
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS user_profile (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            confidence REAL DEFAULT 0.5,  -- 確信度 0.0-1.0
            updated_at REAL NOT NULL,
            source TEXT DEFAULT 'inferred'
        );

        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,       -- 'english', 'math', 'preference', etc.
            fact TEXT NOT NULL,
            embedding BLOB,
            confidence REAL DEFAULT 0.5,
            created_at REAL NOT NULL,
            last_accessed REAL
        );

        CREATE INDEX IF NOT EXISTS idx_conv_session
            ON conversations(session_id);
        CREATE INDEX IF NOT EXISTS idx_conv_created
            ON conversations(created_at);
        CREATE INDEX IF NOT EXISTS idx_facts_category
            ON facts(category);
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Embedding (軽量ローカルモデル)
# ---------------------------------------------------------------------------
_embedder = None


def _get_embedder():
    """
    ローカル埋め込みモデル (all-MiniLM-L6-v2, ~90MB)。
    LLMとは別物。記憶検索専用の小さいモデル。

    Apple Silicon では CoreML バックエンドで高速に動く。
    """
    global _embedder
    if _embedder is not None:
        return _embedder

    try:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="mps",  # Apple Silicon GPU
        )
        return _embedder
    except ImportError:
        return None
    except Exception:
        # MPS 未対応の場合 CPU fallback
        try:
            from sentence_transformers import SentenceTransformer

            _embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            return _embedder
        except Exception:
            return None


def embed_text(text: str) -> Optional[np.ndarray]:
    embedder = _get_embedder()
    if embedder is None:
        return None
    vec = embedder.encode(text, normalize_embeddings=True)
    return vec.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Short-term Memory (コンテキストウィンドウ内)
# ---------------------------------------------------------------------------
class ShortTermMemory:
    """
    直近の会話をそのまま保持。
    プロンプトに含めてLLMに渡す。

    70Bモデル時: max_turns=4  (コンテキスト節約)
    33Bモデル時: max_turns=8
    8Bモデル時:  max_turns=16
    """

    def __init__(self, max_turns: int = 8):
        self.max_turns = max_turns
        self.turns: list[dict] = []

    def add(self, role: str, content: str):
        self.turns.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
        })
        # 古いターンを削除
        if len(self.turns) > self.max_turns * 2:
            self.turns = self.turns[-self.max_turns * 2:]

    def get_context(self) -> str:
        """プロンプト用のコンテキスト文字列を生成"""
        if not self.turns:
            return ""
        lines = []
        for t in self.turns[-self.max_turns * 2:]:
            prefix = "User" if t["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {t['content']}")
        return "\n".join(lines)

    def clear(self):
        self.turns.clear()


# ---------------------------------------------------------------------------
# Mid-term Memory (セマンティック検索)
# ---------------------------------------------------------------------------
class MidTermMemory:
    """
    SQLite + Embedding でセマンティック検索。
    「前にこの単語について聞いたよね？」的な検索ができる。
    """

    def __init__(self):
        self.db = _get_db()

    def store(self, session_id: str, role: str, content: str,
              metadata: Optional[dict] = None):
        vec = embed_text(content)
        blob = vec.tobytes() if vec is not None else None
        meta = json.dumps(metadata or {}, ensure_ascii=False)

        self.db.execute(
            """INSERT INTO conversations
               (session_id, role, content, embedding, created_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, role, content, blob, time.time(), meta),
        )
        self.db.commit()

    def search(self, query: str, top_k: int = 5,
               session_id: Optional[str] = None) -> list[dict]:
        """
        クエリに意味的に近い過去の会話を検索。

        embeddingモデルがない場合はSQLite FTS (全文検索) にフォールバック。
        """
        query_vec = embed_text(query)

        if query_vec is not None:
            return self._search_by_embedding(query_vec, top_k, session_id)
        else:
            return self._search_by_keyword(query, top_k, session_id)

    def _search_by_embedding(self, query_vec: np.ndarray, top_k: int,
                             session_id: Optional[str]) -> list[dict]:
        if session_id:
            rows = self.db.execute(
                """SELECT id, role, content, embedding, created_at, metadata
                   FROM conversations WHERE session_id = ? AND embedding IS NOT NULL
                   ORDER BY created_at DESC LIMIT 500""",
                (session_id,),
            ).fetchall()
        else:
            rows = self.db.execute(
                """SELECT id, role, content, embedding, created_at, metadata
                   FROM conversations WHERE embedding IS NOT NULL
                   ORDER BY created_at DESC LIMIT 500""",
            ).fetchall()

        scored = []
        for row in rows:
            stored_vec = np.frombuffer(row[3], dtype=np.float32)
            score = cosine_similarity(query_vec, stored_vec)
            scored.append({
                "id": row[0],
                "role": row[1],
                "content": row[2],
                "score": score,
                "created_at": row[4],
                "metadata": json.loads(row[5]),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _search_by_keyword(self, query: str, top_k: int,
                           session_id: Optional[str]) -> list[dict]:
        words = query.split()
        like_clauses = " AND ".join(["content LIKE ?"] * len(words))
        params = [f"%{w}%" for w in words]

        if session_id:
            like_clauses += " AND session_id = ?"
            params.append(session_id)

        rows = self.db.execute(
            f"""SELECT id, role, content, created_at, metadata
                FROM conversations WHERE {like_clauses}
                ORDER BY created_at DESC LIMIT ?""",
            params + [top_k],
        ).fetchall()

        return [
            {
                "id": r[0],
                "role": r[1],
                "content": r[2],
                "score": 0.5,  # キーワード検索はスコア固定
                "created_at": r[3],
                "metadata": json.loads(r[4]),
            }
            for r in rows
        ]

    def get_recent(self, session_id: str, limit: int = 20) -> list[dict]:
        rows = self.db.execute(
            """SELECT role, content, created_at FROM conversations
               WHERE session_id = ? ORDER BY created_at DESC LIMIT ?""",
            (session_id, limit),
        ).fetchall()
        return [
            {"role": r[0], "content": r[1], "created_at": r[2]}
            for r in reversed(rows)
        ]


# ---------------------------------------------------------------------------
# Long-term Memory (圧縮記憶 + ユーザープロファイル)
# ---------------------------------------------------------------------------
class LongTermMemory:
    """
    古い会話を要約して保存。
    ユーザーの英語レベル、苦手分野、学習パターンを蓄積。
    """

    def __init__(self):
        self.db = _get_db()

    # --- User Profile ---

    def set_profile(self, key: str, value: str,
                    confidence: float = 0.5, source: str = "inferred"):
        self.db.execute(
            """INSERT INTO user_profile (key, value, confidence, updated_at, source)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET
                   value = excluded.value,
                   confidence = MAX(excluded.confidence, user_profile.confidence),
                   updated_at = excluded.updated_at,
                   source = excluded.source""",
            (key, value, confidence, time.time(), source),
        )
        self.db.commit()

    def get_profile(self, key: Optional[str] = None) -> dict:
        if key:
            row = self.db.execute(
                "SELECT value, confidence FROM user_profile WHERE key = ?",
                (key,),
            ).fetchone()
            return {"value": row[0], "confidence": row[1]} if row else {}

        rows = self.db.execute(
            "SELECT key, value, confidence FROM user_profile ORDER BY key"
        ).fetchall()
        return {r[0]: {"value": r[1], "confidence": r[2]} for r in rows}

    def get_profile_prompt(self) -> str:
        """プロンプトに注入するユーザープロファイル文字列"""
        profile = self.get_profile()
        if not profile:
            return ""

        lines = ["[User Profile]"]
        for k, v in profile.items():
            if v["confidence"] >= 0.3:
                lines.append(f"- {k}: {v['value']}")
        return "\n".join(lines)

    # --- Facts (学習した事実) ---

    def store_fact(self, category: str, fact: str, confidence: float = 0.5):
        vec = embed_text(fact)
        blob = vec.tobytes() if vec is not None else None
        self.db.execute(
            """INSERT INTO facts (category, fact, embedding, confidence, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (category, fact, blob, confidence, time.time()),
        )
        self.db.commit()

    def search_facts(self, query: str, category: Optional[str] = None,
                     top_k: int = 5) -> list[dict]:
        query_vec = embed_text(query)

        conditions = ["embedding IS NOT NULL"]
        params = []
        if category:
            conditions.append("category = ?")
            params.append(category)

        where = " AND ".join(conditions)
        rows = self.db.execute(
            f"""SELECT id, category, fact, embedding, confidence
                FROM facts WHERE {where}
                ORDER BY created_at DESC LIMIT 200""",
            params,
        ).fetchall()

        if query_vec is None:
            return [
                {"category": r[1], "fact": r[2], "confidence": r[4], "score": 0.5}
                for r in rows[:top_k]
            ]

        scored = []
        for r in rows:
            stored_vec = np.frombuffer(r[3], dtype=np.float32)
            score = cosine_similarity(query_vec, stored_vec)
            scored.append({
                "category": r[1],
                "fact": r[2],
                "confidence": r[4],
                "score": score,
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    # --- Summaries (会話要約) ---

    def store_summary(self, session_id: str, summary: str, turn_range: str):
        self.db.execute(
            """INSERT INTO summaries (session_id, summary, turn_range, created_at)
               VALUES (?, ?, ?, ?)""",
            (session_id, summary, turn_range, time.time()),
        )
        self.db.commit()

    def get_summaries(self, session_id: Optional[str] = None,
                      limit: int = 10) -> list[dict]:
        if session_id:
            rows = self.db.execute(
                """SELECT summary, turn_range, created_at FROM summaries
                   WHERE session_id = ? ORDER BY created_at DESC LIMIT ?""",
                (session_id, limit),
            ).fetchall()
        else:
            rows = self.db.execute(
                """SELECT summary, turn_range, created_at FROM summaries
                   ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [
            {"summary": r[0], "turn_range": r[1], "created_at": r[2]}
            for r in rows
        ]


# ---------------------------------------------------------------------------
# Memory Manager (3層統合)
# ---------------------------------------------------------------------------
class MemoryManager:
    """
    3層メモリを統合管理。

    使い方:
        mm = MemoryManager(model_tier="70b")
        mm.add_turn("user", "What does 'abundant' mean?")
        context = mm.build_context("What does 'abundant' mean?")
        # → context にはプロファイル + 関連記憶 + 直近会話が含まれる
        mm.add_turn("assistant", "'abundant' means ...")
    """

    # モデルサイズ別の設定
    # 70B: メモリカツカツ → コンテキスト最小限
    # 33B: 余裕あり → コンテキスト中程度
    # 8B:  メモリ余裕大 → コンテキスト最大
    TIER_CONFIG = {
        "70b": {"short_turns": 4, "mid_results": 2, "max_context_chars": 2000},
        "33b": {"short_turns": 8, "mid_results": 4, "max_context_chars": 4000},
        "8b": {"short_turns": 16, "mid_results": 6, "max_context_chars": 8000},
    }

    def __init__(self, model_tier: str = "8b", session_id: Optional[str] = None):
        cfg = self.TIER_CONFIG.get(model_tier, self.TIER_CONFIG["8b"])

        self.session_id = session_id or f"session_{int(time.time())}"
        self.short = ShortTermMemory(max_turns=cfg["short_turns"])
        self.mid = MidTermMemory()
        self.long = LongTermMemory()
        self.max_context_chars = cfg["max_context_chars"]
        self.mid_results = cfg["mid_results"]
        self.model_tier = model_tier
        self._turn_count = 0

    def add_turn(self, role: str, content: str, metadata: Optional[dict] = None):
        """会話ターンを全層に記録"""
        self.short.add(role, content)
        self.mid.store(self.session_id, role, content, metadata)
        self._turn_count += 1

        # 50ターンごとに要約を生成するトリガー
        # (実際の要約はLLM呼び出しが必要なのでserver.pyで実行)
        if self._turn_count % 50 == 0:
            self._needs_summary = True

    def build_context(self, current_query: str) -> str:
        """
        現在のクエリに対する最適なコンテキストを構築。

        構造:
        [System: User Profile]     ← Long-term
        [Related memories]         ← Mid-term (セマンティック検索)
        [Recent conversation]      ← Short-term
        """
        parts = []

        # 1. Long-term: ユーザープロファイル
        profile = self.long.get_profile_prompt()
        if profile:
            parts.append(profile)

        # 2. Mid-term: 関連する過去の会話
        related = self.mid.search(current_query, top_k=self.mid_results)
        if related:
            memory_lines = ["[Related memories]"]
            for r in related:
                if r["score"] > 0.3:  # 関連度閾値
                    ts = datetime.fromtimestamp(r["created_at"]).strftime("%m/%d %H:%M")
                    memory_lines.append(f"- [{ts}] {r['role']}: {r['content'][:200]}")
            if len(memory_lines) > 1:
                parts.append("\n".join(memory_lines))

        # 3. Long-term: 関連する学習済み事実
        facts = self.long.search_facts(current_query, top_k=3)
        if facts:
            fact_lines = ["[Known facts]"]
            for f in facts:
                if f["score"] > 0.4:
                    fact_lines.append(f"- {f['fact']}")
            if len(fact_lines) > 1:
                parts.append("\n".join(fact_lines))

        # 4. Short-term: 直近の会話
        recent = self.short.get_context()
        if recent:
            parts.append(f"[Recent conversation]\n{recent}")

        # コンテキスト長制限 (70Bはカツカツなので厳しく制限)
        context = "\n\n".join(parts)
        if len(context) > self.max_context_chars:
            context = context[-self.max_context_chars:]

        return context

    def learn_from_interaction(self, user_msg: str, assistant_msg: str):
        """
        会話からユーザー情報を抽出して Long-term に保存。

        例: ユーザーが英検2級の問題を間違えた
        → profile["eiken2_weak_areas"] に記録
        → 次回の問題生成で苦手分野を重点出題
        """
        # 英検関連のキーワード検出
        if any(kw in user_msg.lower() for kw in ["eiken", "英検", "toeic"]):
            self.long.set_profile(
                "study_focus", "english_certification",
                confidence=0.7, source="conversation"
            )

        # 数学関連
        if any(kw in user_msg.lower() for kw in ["三角関数", "微分", "積分", "math"]):
            self.long.set_profile(
                "study_focus_math", "active",
                confidence=0.6, source="conversation"
            )

        # 間違えた問題の記録
        if "wrong" in assistant_msg.lower() or "不正解" in assistant_msg:
            self.long.store_fact(
                "weakness",
                f"間違えた内容: {user_msg[:200]}",
                confidence=0.8,
            )

    def get_stats(self) -> dict:
        """メモリの使用統計"""
        conv_count = self.mid.db.execute(
            "SELECT COUNT(*) FROM conversations"
        ).fetchone()[0]
        fact_count = self.mid.db.execute(
            "SELECT COUNT(*) FROM facts"
        ).fetchone()[0]
        summary_count = self.mid.db.execute(
            "SELECT COUNT(*) FROM summaries"
        ).fetchone()[0]
        profile = self.long.get_profile()

        return {
            "model_tier": self.model_tier,
            "session_id": self.session_id,
            "short_term_turns": len(self.short.turns),
            "mid_term_conversations": conv_count,
            "long_term_facts": fact_count,
            "long_term_summaries": summary_count,
            "profile_keys": len(profile),
            "db_path": str(DB_PATH),
        }
