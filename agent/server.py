"""
iHax Agent Server
=================
Apple Silicon (M5 Pro 64GB) + MLX でローカルLLMを動かし、
PWAアプリ (iN glish, 英検2級, 数学) にAI機能を提供する。

2層モデル構成:
- balanced: 33B(推論) + 8B(記憶管理) 同時運用
- max_quality: 70B(3bit) オンデマンド + 8B常駐
- extreme: 70B(4bit) 単体フルパワー

3層記憶システム:
- Short-term: コンテキスト内の直近会話
- Mid-term: SQLite + Embedding セマンティック検索
- Long-term: 圧縮要約 + ユーザープロファイル

翻訳: 無料CDN (ローカルLLMのメモリは記憶・推論に全振り)

起動: python agent/server.py
API:  http://localhost:8000/docs
"""

from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from mlx_engine import MLXEngine, get_hardware_config, PRESETS, MODEL_CATALOG
from memory import MemoryManager

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="iHax Agent",
    description="ローカルLLM API for iHax (M5 Pro 64GB, 3層記憶, 2層モデル)",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Engine + Memory (遅延初期化)
# ---------------------------------------------------------------------------
engine: Optional[MLXEngine] = None
memory: Optional[MemoryManager] = None

DEFAULT_PRESET = "balanced"  # 33B + 8B 同時。M5 Pro 64GBに最適


def get_engine() -> MLXEngine:
    global engine
    if engine is None:
        config = get_hardware_config(DEFAULT_PRESET)
        engine = MLXEngine(config)
    return engine


def get_memory() -> MemoryManager:
    global memory
    if memory is None:
        config = get_hardware_config(DEFAULT_PRESET)
        memory = MemoryManager(
            model_tier=config["memory_tier"],
            session_id=f"ihax_{int(time.time())}",
        )
    return memory


# ---------------------------------------------------------------------------
# Cache (簡易インメモリ)
# ---------------------------------------------------------------------------
_cache: dict[str, tuple[str, float]] = {}
CACHE_TTL = 3600


def cache_key(prefix: str, text: str, **kwargs) -> str:
    raw = f"{prefix}:{text}:{json.dumps(kwargs, sort_keys=True)}"
    return hashlib.md5(raw.encode()).hexdigest()


def get_cached(key: str) -> Optional[str]:
    if key in _cache:
        result, ts = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return result
        del _cache[key]
    return None


def set_cache(key: str, value: str):
    _cache[key] = (value, time.time())


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------
class TranslateRequest(BaseModel):
    text: str
    direction: str = Field(
        default="ja_to_en",
        description="ja_to_en or en_to_ja",
    )
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)


class TranslateResponse(BaseModel):
    translated: str
    direction: str
    cached: bool = False


class QuizRequest(BaseModel):
    topic: str = Field(description="e.g. 'vocabulary', 'grammar', 'eiken2'")
    count: int = Field(default=5, ge=1, le=20)
    difficulty: str = Field(default="medium", description="easy/medium/hard")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class QuizResponse(BaseModel):
    questions: list[dict]


class ExplainRequest(BaseModel):
    question: str
    subject: str = Field(default="english", description="english/math")
    language: str = Field(default="ja", description="ja or en")
    temperature: float = Field(default=0.5, ge=0.0, le=2.0)


class ExplainResponse(BaseModel):
    explanation: str


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_k: int = Field(default=40, ge=1, le=200)


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    tokens_per_second: float


class HealthResponse(BaseModel):
    status: str
    preset: str
    preset_description: str
    primary_model: str
    secondary_model: Optional[str]
    memory_gb: float
    estimated_usage_gb: float
    remaining_gb: float
    chip: str
    memory_stats: dict


class PresetRequest(BaseModel):
    preset: str = Field(description="balanced / max_quality / extreme / lightweight")


class MemorySearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class MemorySearchResponse(BaseModel):
    results: list[dict]


class ProfileUpdateRequest(BaseModel):
    key: str
    value: str
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class FactStoreRequest(BaseModel):
    category: str
    fact: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    """ヘルスチェック + ハードウェア + 記憶統計"""
    config = get_hardware_config(DEFAULT_PRESET)
    eng = get_engine()
    mem = get_memory()
    return HealthResponse(
        status="ok",
        preset=config["preset"],
        preset_description=config["preset_description"],
        primary_model=config["primary_model"],
        secondary_model=config.get("secondary_model"),
        memory_gb=config["memory_gb"],
        estimated_usage_gb=config["estimated_usage_gb"],
        remaining_gb=config["remaining_gb"],
        chip=config["chip"],
        memory_stats=mem.get_stats(),
    )


@app.get("/presets")
async def list_presets():
    """利用可能なプリセット一覧"""
    result = {}
    for name, p in PRESETS.items():
        cat = MODEL_CATALOG[p["primary"]]
        result[name] = {
            "description": p["description"],
            "primary": p["primary"],
            "secondary": p["secondary"],
            "total_gb": cat["total_gb"],
            "quality": cat["quality"],
        }
    return result


@app.post("/preset")
async def switch_preset(req: PresetRequest):
    """モデルプリセット切り替え (70B↔33B↔8B)"""
    global engine, memory
    if req.preset not in PRESETS:
        raise HTTPException(400, f"Unknown preset: {req.preset}")

    eng = get_engine()
    eng.switch_preset(req.preset)

    config = get_hardware_config(req.preset)
    memory = MemoryManager(
        model_tier=config["memory_tier"],
        session_id=get_memory().session_id,
    )

    return {"status": "ok", "preset": req.preset}


@app.get("/model/status")
async def model_status():
    """現在のモデルロード状態"""
    eng = get_engine()
    return eng.get_status()


@app.post("/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest):
    """日英・英日翻訳 (ローカルLLM)"""
    key = cache_key("translate", req.text, direction=req.direction)
    cached = get_cached(key)
    if cached:
        return TranslateResponse(
            translated=cached, direction=req.direction, cached=True
        )

    eng = get_engine()

    if req.direction == "ja_to_en":
        prompt = f"Translate the following Japanese text to natural English. Output only the translation.\n\nJapanese: {req.text}\nEnglish:"
    else:
        prompt = f"Translate the following English text to natural Japanese. Output only the translation.\n\nEnglish: {req.text}\nJapanese:"

    result = eng.generate(prompt, max_tokens=256, temperature=req.temperature)
    set_cache(key, result)

    return TranslateResponse(
        translated=result, direction=req.direction, cached=False
    )


@app.post("/quiz", response_model=QuizResponse)
async def generate_quiz(req: QuizRequest):
    """英検・英語問題の自動生成"""
    eng = get_engine()

    prompt = f"""Generate {req.count} multiple-choice English quiz questions.
Topic: {req.topic}
Difficulty: {req.difficulty}

Output as JSON array. Each question has:
- "question": the question text
- "choices": array of 4 choices
- "answer": index of correct answer (0-3)
- "explanation": brief explanation in Japanese

Output only valid JSON, no other text."""

    result = eng.generate(prompt, max_tokens=2048, temperature=req.temperature)

    try:
        questions = json.loads(result)
        if not isinstance(questions, list):
            questions = [questions]
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="モデルの出力をパースできませんでした")

    return QuizResponse(questions=questions)


@app.post("/explain", response_model=ExplainResponse)
async def explain(req: ExplainRequest):
    """英語・数学の解説生成"""
    key = cache_key("explain", req.question, subject=req.subject, lang=req.language)
    cached = get_cached(key)
    if cached:
        return ExplainResponse(explanation=cached)

    eng = get_engine()

    lang_instruction = "日本語で" if req.language == "ja" else "in English"

    if req.subject == "math":
        prompt = f"""あなたは数学の教師です。以下の質問に{lang_instruction}わかりやすく答えてください。
数式はそのまま書いてください。

質問: {req.question}

解説:"""
    else:
        prompt = f"""あなたは英語の教師です。以下の質問に{lang_instruction}わかりやすく答えてください。
例文も含めてください。

質問: {req.question}

解説:"""

    result = eng.generate(prompt, max_tokens=1024, temperature=req.temperature)
    set_cache(key, result)

    return ExplainResponse(explanation=result)


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """
    汎用テキスト生成 (Temperature + TopK)。
    記憶コンテキストを自動注入。
    """
    eng = get_engine()
    mem = get_memory()

    # 記憶からコンテキストを構築
    memory_context = mem.build_context(req.prompt)

    # メモリ付きプロンプト構築
    if memory_context:
        full_prompt = f"{memory_context}\n\nUser: {req.prompt}\nAssistant:"
    else:
        full_prompt = req.prompt

    t0 = time.time()
    result = eng.generate(
        full_prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
    )
    elapsed = time.time() - t0
    n_tokens = len(result.split())

    # 会話を記憶に保存
    mem.add_turn("user", req.prompt)
    mem.add_turn("assistant", result)
    mem.learn_from_interaction(req.prompt, result)

    return GenerateResponse(
        text=result,
        tokens_generated=n_tokens,
        tokens_per_second=n_tokens / max(elapsed, 0.001),
    )


# ---------------------------------------------------------------------------
# Memory Endpoints
# ---------------------------------------------------------------------------
@app.post("/memory/search", response_model=MemorySearchResponse)
async def memory_search(req: MemorySearchRequest):
    """過去の会話をセマンティック検索"""
    mem = get_memory()
    results = mem.mid.search(req.query, top_k=req.top_k)
    return MemorySearchResponse(results=results)


@app.get("/memory/profile")
async def memory_profile():
    """ユーザープロファイル取得"""
    mem = get_memory()
    return mem.long.get_profile()


@app.post("/memory/profile")
async def memory_profile_update(req: ProfileUpdateRequest):
    """ユーザープロファイル更新"""
    mem = get_memory()
    mem.long.set_profile(req.key, req.value, req.confidence, source="explicit")
    return {"status": "ok"}


@app.post("/memory/fact")
async def memory_store_fact(req: FactStoreRequest):
    """学習した事実を保存"""
    mem = get_memory()
    mem.long.store_fact(req.category, req.fact, req.confidence)
    return {"status": "ok"}


@app.get("/memory/stats")
async def memory_stats():
    """記憶システムの統計"""
    mem = get_memory()
    return mem.get_stats()


@app.post("/memory/summarize")
async def memory_summarize():
    """直近の会話を要約して Long-term に保存 (8Bモデル使用)"""
    eng = get_engine()
    mem = get_memory()

    recent = mem.mid.get_recent(mem.session_id, limit=20)
    if not recent:
        return {"status": "no_conversations"}

    conv_text = "\n".join(
        f"{r['role']}: {r['content']}" for r in recent
    )

    summary = eng.summarize(conv_text)
    mem.long.store_summary(
        mem.session_id,
        summary,
        turn_range=f"last_{len(recent)}",
    )

    return {"status": "ok", "summary": summary}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    config = get_hardware_config(DEFAULT_PRESET)

    print("=" * 60)
    print("  iHax Agent Server v0.2")
    print("  http://localhost:8000")
    print("  API docs: http://localhost:8000/docs")
    print("=" * 60)
    print(f"  Chip:      {config['chip']}")
    print(f"  Memory:    {config['memory_gb']:.0f} GB")
    print(f"  Preset:    {config['preset']} - {config['preset_description']}")
    print(f"  Primary:   {config['primary_model']}")
    print(f"  Secondary: {config.get('secondary_model', 'None')}")
    print(f"  Usage:     ~{config['estimated_usage_gb']} GB / {config['memory_gb']:.0f} GB")
    print(f"  Remaining: ~{config['remaining_gb']:.0f} GB (記憶+embedding用)")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000)
