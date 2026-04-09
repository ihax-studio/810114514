"""
MLX Engine for iHax Agent
=========================
M5 Pro 64GB — 60GB GPU + SSD 400GB 戦略。

メモリ配分:
  物理64GB = GPU 60GB + macOS 4GB
  SSD: 400GB (モデル保管 + KV cache overflow)
  仮想メモリ(SSD swap): 20GB (15GB/s, OSの保険)
  GPU帯域: ~273 GB/s (M5 Pro)

  sudo sysctl iogpu.wired_limit_mb=61440

目標:
  - DeepSeek/Qwen を超えるレベルのモデルを動かす
  - 1Mコンテキスト (100万トークン) を実現する
  - OS開発、デザイン、ネイティブアプリ開発を支援

1Mコンテキスト戦略:
  70Bでは1M contextのKVcacheだけで500GB+→物理的に不可能
  → 8B/小型モデル + KV cache量子化 + SSD overflow で1Mを攻略
  → 重い推論は70Bに切り替え (コンテキストは記憶システムで補完)

依存: pip install mlx mlx-lm
"""

from __future__ import annotations

import gc
import os
import platform
import subprocess
from typing import Optional


# ---------------------------------------------------------------------------
# GPU メモリ + SSD 設定
# ---------------------------------------------------------------------------
GPU_WIRED_LIMIT_MB = 61440  # 60GB
OS_RESERVED_GB = 4          # macOS最低限
SWAP_GB = 20                # SSD仮想メモリ (15GB/s)
GPU_BUDGET_GB = 60          # LLM用
SSD_BUDGET_GB = 400         # モデル保管 + KV overflow
SSD_BANDWIDTH_GBS = 15      # M5 Pro SSD 読み書き速度


def setup_gpu_allocation():
    """
    macOSのGPUメモリ制限を60GBに引き上げる。
    デフォルトは物理メモリの75% (48GB) だが、sysctl で突破。
    再起動で元に戻るので安全。
    """
    if platform.system() != "Darwin":
        return False

    try:
        result = subprocess.run(
            ["sysctl", "iogpu.wired_limit_mb"],
            capture_output=True, text=True,
        )
        current = int(result.stdout.split(":")[-1].strip())
        if current >= GPU_WIRED_LIMIT_MB:
            return True

        print(f"[iHax] GPU制限を {current}MB → {GPU_WIRED_LIMIT_MB}MB に引き上げます")
        print(f"[iHax] sudo sysctl iogpu.wired_limit_mb={GPU_WIRED_LIMIT_MB}")
        return False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# モデルカタログ (60GB GPU + 400GB SSD)
# ---------------------------------------------------------------------------
# 2026年現在のトップモデルを想定。
# DeepSeek/Qwen超えを狙うなら Llama 4, Mistral Large 後継,
# またはその時点の最新オープンモデルを使う。
# ここでは汎用的なスロットとして定義し、実モデル名は到着時に最新に差し替える。

MODEL_CATALOG = {
    # =======================================================================
    # 70B+ — DeepSeek/Qwen超え。OS設計・アーキテクチャレベルの知性
    # =======================================================================
    "top_6bit": {
        "name": "mlx-community/Meta-Llama-3.1-70B-Instruct-6bit",
        "model_gb": 53,
        "kv_per_1k_tokens_mb": 128,    # 1Kトークンあたり128MB
        "max_context": 4096,           # 60GB内で収まるコンテキスト
        "total_gb": 58,
        "tier": "70b",
        "ssd_gb": 60,                  # SSD上のモデルサイズ
        "quality": "ほぼ原品質。最高の推論力",
        "use_case": "OS設計、アーキテクチャ、複雑なデバッグ",
    },
    "top_5bit": {
        "name": "mlx-community/Meta-Llama-3.1-70B-Instruct-5bit",
        "model_gb": 47,
        "kv_per_1k_tokens_mb": 128,
        "max_context": 8192,
        "total_gb": 55,
        "tier": "70b",
        "ssd_gb": 50,
        "quality": "高品質 + 8Kコンテキスト",
        "use_case": "長いソースコード解析、リファクタリング",
    },
    "top_4bit": {
        "name": "mlx-community/Meta-Llama-3.1-70B-Instruct-4bit",
        "model_gb": 40,
        "kv_per_1k_tokens_mb": 128,
        "max_context": 16384,
        "total_gb": 55,
        "tier": "70b",
        "ssd_gb": 42,
        "quality": "16Kコンテキスト。巨大プロジェクト向き",
        "use_case": "大規模コードベース、ドキュメント全体解析",
    },

    # =======================================================================
    # 1M CONTEXT — 100万トークンの野望
    # =======================================================================
    # 1Mコンテキストの算数:
    #   70B: KV cache 1M tokens = 128MB × 1000 = 128GB → 不可能
    #   8B:  KV cache 1M tokens = 16MB × 1000 = 16GB
    #        + モデル 5GB = 21GB → 60GBに余裕で収まる！
    #
    #   KV cache Q4量子化すると 16GB → 4GB
    #   → モデル5GB + KV 4GB = 9GB で1M context 達成
    #   → 残り51GBはembedding + 記憶システムに使える
    #
    # ただし8Bの品質がDeepSeek/Qwen超えかは微妙。
    # → 記憶システムで補完: 1M context内に要約+事実+プロファイルを注入
    #   → 実質的に8Bでも文脈理解力が大幅に上がる

    "1m_context": {
        "name": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        "model_gb": 5,
        "kv_per_1k_tokens_mb": 16,     # 8BモデルのKV cache
        "max_context": 1048576,        # 1M tokens !
        "total_gb": 21,                # model 5GB + KV 16GB (Q8)
        "tier": "1m",
        "ssd_gb": 6,
        "quality": "8Bだが1Mコンテキスト。記憶システムで知性を補完",
        "use_case": "巨大ファイル全体読み込み、長時間の設計議論、プロジェクト全体理解",
    },
    "1m_context_kv_q4": {
        "name": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        "model_gb": 5,
        "kv_per_1k_tokens_mb": 4,      # KV cache Q4量子化
        "max_context": 1048576,        # 1M tokens
        "total_gb": 9,                 # model 5GB + KV 4GB (Q4)
        "tier": "1m",
        "ssd_gb": 6,
        "quality": "KV Q4量子化で超軽量1M。品質やや落ちるが51GB余る",
        "use_case": "1M + 70B切替ハイブリッド運用",
    },

    # =======================================================================
    # Coder — コード特化 (Swift/Python/JS)
    # =======================================================================
    "coder_32b": {
        "name": "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
        "model_gb": 20,
        "kv_per_1k_tokens_mb": 64,
        "max_context": 8192,
        "total_gb": 28,
        "tier": "33b",
        "ssd_gb": 22,
        "quality": "コード生成最強クラス。Swift/Python/JS全対応",
        "use_case": "ネイティブアプリ開発、UI実装、API設計",
    },

    # =======================================================================
    # 汎用 33B
    # =======================================================================
    "general_32b": {
        "name": "mlx-community/Qwen2.5-32B-Instruct-4bit",
        "model_gb": 20,
        "kv_per_1k_tokens_mb": 64,
        "max_context": 8192,
        "total_gb": 28,
        "tier": "33b",
        "ssd_gb": 22,
        "quality": "汎用バランス型",
        "use_case": "日常開発、質問応答、設計議論",
    },

    # =======================================================================
    # 8B — 常駐 (記憶管理・要約)
    # =======================================================================
    "assistant_8b": {
        "name": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        "model_gb": 5,
        "kv_per_1k_tokens_mb": 16,
        "max_context": 8192,
        "total_gb": 9,
        "tier": "8b",
        "ssd_gb": 6,
        "quality": "常駐用。要約・分類・記憶整理",
        "use_case": "記憶圧縮、会話分類、バックグラウンド処理",
    },
}

# ---------------------------------------------------------------------------
# SSD モデルストレージ
# ---------------------------------------------------------------------------
# 400GB SSD にどれだけモデルを保管できるか:
#
#   70B 6bit: ~60GB
#   70B 5bit: ~50GB
#   70B 4bit: ~42GB
#   Coder 32B: ~22GB
#   General 32B: ~22GB
#   8B: ~6GB
#   ─────────────────
#   合計: ~202GB → 400GBに全部入る。余り ~200GB
#
#   余った200GBは:
#   - 追加モデル (Mistral, DeepSeek後継, 新モデル)
#   - KV cache のSSD overflow用
#   - FineTuning用のデータセット保管
#
# 到着したらその時点の最強モデルをダウンロード:
#   mlx_lm.convert で HuggingFace → MLX 変換

SSD_STORAGE_PLAN = {
    "models": {
        "top_6bit": 60,
        "top_5bit": 50,
        "top_4bit": 42,
        "coder_32b": 22,
        "general_32b": 22,
        "assistant_8b": 6,
    },
    "total_models_gb": 202,
    "kv_overflow_gb": 50,      # KV cache SSD overflow用
    "finetune_data_gb": 50,    # FineTuning用データ
    "free_gb": 98,             # 余り (追加モデル用)
}

# ---------------------------------------------------------------------------
# プリセット (60GB GPU)
# ---------------------------------------------------------------------------
PRESETS = {
    # === 最高品質 (DeepSeek/Qwen超え) ===
    "architect": {
        "description": "70B 6bit 単体。ほぼ原品質。OS設計・アーキテクチャに",
        "primary": "top_6bit",
        "secondary": None,
        "memory_tier": "70b",
        "simultaneous": False,
    },
    "developer": {
        "description": "70B 5bit 単体。高品質 + 8Kコンテキスト。開発メイン",
        "primary": "top_5bit",
        "secondary": None,
        "memory_tier": "70b",
        "simultaneous": False,
    },
    "longcontext": {
        "description": "70B 4bit 単体。16Kコンテキスト。大規模コード解析",
        "primary": "top_4bit",
        "secondary": None,
        "memory_tier": "70b",
        "simultaneous": False,
    },

    # === 1M CONTEXT (100万トークン) ===
    "million": {
        "description": "8B + 1Mコンテキスト。プロジェクト全体を丸ごと理解",
        "primary": "1m_context",
        "secondary": None,
        "memory_tier": "1m",
        "simultaneous": False,
    },
    "hybrid_1m": {
        "description": "8B(1M ctx, KV-Q4) + Coder32B 切替。1M読み→32Bで推論",
        "primary": "1m_context_kv_q4",
        "secondary": "coder_32b",
        "memory_tier": "1m",
        "simultaneous": False,  # 切替式: 1Mで読む→32Bで書く
    },

    # === コード特化 ===
    "balanced": {
        "description": "Coder 32B + 8B同時。コード特化 + 記憶管理。安定運用",
        "primary": "coder_32b",
        "secondary": "assistant_8b",
        "memory_tier": "33b",
        "simultaneous": True,
    },
    "swift_dev": {
        "description": "汎用32B + 8B同時。Swift/SwiftUI開発に最適",
        "primary": "general_32b",
        "secondary": "assistant_8b",
        "memory_tier": "33b",
        "simultaneous": True,
    },

    # === 軽量 ===
    "lightweight": {
        "description": "8B単体。Xcode + Figma同時起動OK",
        "primary": "assistant_8b",
        "secondary": None,
        "memory_tier": "8b",
        "simultaneous": False,
    },
}


def get_hardware_config(preset: str = "balanced") -> dict:
    """
    実行環境のハードウェアを検出し、GPU+SSD予算に基づいた設定を返す。

    メモリ配分:
      64GB = GPU 60GB + macOS 4GB
      SSD 400GB = モデル保管 + KV overflow + FineTuneデータ
      仮想メモリ(SSD swap) 20GB が保険

    iogpu.wired_limit_mb=61440 が設定されている前提。
    """
    chip = _detect_chip()
    memory_gb = _detect_memory_gb()

    gpu_budget = min(GPU_BUDGET_GB, memory_gb - OS_RESERVED_GB)

    p = PRESETS.get(preset, PRESETS["balanced"])
    primary = MODEL_CATALOG[p["primary"]]
    secondary = MODEL_CATALOG[p["secondary"]] if p["secondary"] else None

    required = primary["total_gb"]
    if secondary and p["simultaneous"]:
        required += secondary["total_gb"]

    # GPU予算オーバー → フォールバック
    fallback_chain = {
        "architect": "developer",
        "developer": "longcontext",
        "longcontext": "balanced",
        "million": "hybrid_1m",
        "hybrid_1m": "balanced",
        "balanced": "swift_dev",
        "swift_dev": "lightweight",
    }
    if required > gpu_budget and preset in fallback_chain:
        return get_hardware_config(fallback_chain[preset])

    # SSD使用量
    ssd_models = sum(
        MODEL_CATALOG[p["primary"]].get("ssd_gb", 0)
        for p in PRESETS.values()
    )

    return {
        "chip": chip,
        "memory_gb": memory_gb,
        "gpu_budget_gb": gpu_budget,
        "os_reserved_gb": OS_RESERVED_GB,
        "swap_gb": SWAP_GB,
        "ssd_budget_gb": SSD_BUDGET_GB,
        "preset": preset,
        "preset_description": p["description"],
        "primary_model": primary["name"],
        "primary_use_case": primary.get("use_case", ""),
        "secondary_model": secondary["name"] if secondary else None,
        "primary_context": primary["max_context"],
        "secondary_context": secondary["max_context"] if secondary else 0,
        "memory_tier": p["memory_tier"],
        "simultaneous": p["simultaneous"],
        "estimated_usage_gb": required,
        "remaining_gpu_gb": gpu_budget - required,
    }


def _detect_chip() -> str:
    """Apple Silicon チップを検出"""
    if platform.system() != "Darwin":
        return f"non-mac ({platform.processor() or platform.machine()})"
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or "Apple Silicon (unknown)"
    except Exception:
        return "Apple Silicon (detection failed)"


def _detect_memory_gb() -> float:
    """システムメモリをGB単位で取得"""
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
            )
            return int(result.stdout.strip()) / (1024**3)
        except Exception:
            pass

    # Linux / fallback
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024**2)
    except Exception:
        pass

    return 8.0


class MLXEngine:
    """
    MLXベースのLLM推論エンジン。

    M5 Pro 64GB — 60GB GPU + SSD 400GB 戦略。
    OS開発、デザイン、ネイティブアプリ開発を最大限支援。

    用途別プリセット:
    - architect:    70B 6bit — OS設計、カーネル、アーキテクチャ (ctx 4K)
    - developer:    70B 5bit — 開発メイン (ctx 8K)
    - longcontext:  70B 4bit — 大規模コード解析 (ctx 16K)
    - million:      8B + 1Mコンテキスト — プロジェクト丸ごと理解
    - hybrid_1m:    8B(1M) ↔ Coder32B 切替 — 読む→書く
    - balanced:     Coder 32B + 8B — コード特化 + 記憶
    - swift_dev:    33B + 8B — Swift/SwiftUI開発
    - lightweight:  8B — Xcode + Figma同時起動OK

    M5 Pro (273 GB/s) での推論速度:
    - 70B 6bit: ~5-8 tok/s   (設計議論に十分)
    - 70B 5bit: ~7-10 tok/s  (開発に快適)
    - 70B 4bit: ~8-12 tok/s  (サクサク)
    - 33B 4bit: ~15-25 tok/s (爆速)
    - 8B 4bit:  ~40-60 tok/s (瞬殺)
    - 8B 1M ctx: ~20-30 tok/s (KV cache量子化時)

    SSD活用:
    - 全モデル保管 (~202GB / 400GB)
    - KV cache overflow (50GB)
    - FineTuning用データ (50GB)
    - 余り ~98GB (新モデル追加用)
    """

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = get_hardware_config()

        self.config = config
        self.primary_name = config["primary_model"]
        self.secondary_name = config.get("secondary_model")
        self.primary_context = config.get("primary_context", 4096)
        self.simultaneous = config.get("simultaneous", False)

        self._primary_model = None
        self._primary_tokenizer = None
        self._secondary_model = None
        self._secondary_tokenizer = None
        self._active = "primary"

    @property
    def model_name(self) -> str:
        if self._active == "secondary" and self.secondary_name:
            return self.secondary_name
        return self.primary_name

    def _load_model(self, which: str = "primary"):
        """
        モデルの遅延ロード。

        simultaneous=True: 両方同時にメモリに載せる
        simultaneous=False: 切り替え式 (片方アンロード→gc.collect→ロード)
        """
        try:
            from mlx_lm import load
        except ImportError:
            raise RuntimeError(
                "mlx-lm がインストールされていません。\n"
                "pip install mlx mlx-lm\n"
                "※ Apple Silicon Mac が必要です"
            )

        if which == "primary":
            if self._primary_model is not None:
                return
            if not self.simultaneous and self._secondary_model is not None:
                print("[iHax] Unloading secondary for primary...")
                self._secondary_model = None
                self._secondary_tokenizer = None
                gc.collect()

            print(f"[iHax] Loading primary: {self.primary_name}")
            self._primary_model, self._primary_tokenizer = load(self.primary_name)
            self._active = "primary"

        elif which == "secondary" and self.secondary_name:
            if self._secondary_model is not None:
                return
            if not self.simultaneous and self._primary_model is not None:
                print("[iHax] Unloading primary for secondary...")
                self._primary_model = None
                self._primary_tokenizer = None
                gc.collect()

            print(f"[iHax] Loading secondary: {self.secondary_name}")
            self._secondary_model, self._secondary_tokenizer = load(
                self.secondary_name
            )
            self._active = "secondary"

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 40,
        use_secondary: bool = False,
    ) -> str:
        """
        テキスト生成。

        use_secondary=True: 8Bモデルで生成 (要約・分類用)
        use_secondary=False: プライマリで生成 (メイン推論)
        """
        if use_secondary and self.secondary_name:
            self._load_model("secondary")
            model = self._secondary_model
            tokenizer = self._secondary_tokenizer
        else:
            self._load_model("primary")
            model = self._primary_model
            tokenizer = self._primary_tokenizer

        try:
            from mlx_lm import generate as mlx_generate

            result = mlx_generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
                top_p=0.9,
            )
            return result.strip()
        except Exception as e:
            raise RuntimeError(f"生成エラー: {e}")

    def summarize(self, text: str, max_tokens: int = 256) -> str:
        """テキスト要約 (8Bモデル使用)。記憶の圧縮に使う。"""
        prompt = (
            "Summarize the following conversation concisely in Japanese. "
            "Focus on key facts and decisions.\n\n"
            f"{text}\n\nSummary:"
        )
        return self.generate(
            prompt, max_tokens=max_tokens, temperature=0.3, use_secondary=True,
        )

    def classify(self, text: str, categories: list[str]) -> str:
        """テキスト分類 (8Bモデル使用)。記憶のカテゴリ分けに。"""
        cats = ", ".join(categories)
        prompt = (
            f"Classify the following text into one of these categories: {cats}\n"
            f"Text: {text}\nCategory:"
        )
        return self.generate(
            prompt, max_tokens=10, temperature=0.1, use_secondary=True,
        )

    def extract_facts(self, conversation: str) -> str:
        """会話から事実を抽出 (8Bモデル使用)。Long-term memory 用。"""
        prompt = (
            "Extract key facts from this conversation as a JSON array of strings. "
            "Focus on: user preferences, mistakes made, topics discussed, "
            "knowledge gaps.\n\n"
            f"Conversation:\n{conversation}\n\nFacts (JSON array):"
        )
        return self.generate(
            prompt, max_tokens=512, temperature=0.2, use_secondary=True,
        )

    def switch_preset(self, preset: str):
        """プリセット切り替え (全モデルアンロード → 再構成)"""
        self._primary_model = None
        self._primary_tokenizer = None
        self._secondary_model = None
        self._secondary_tokenizer = None
        gc.collect()

        new_config = get_hardware_config(preset)
        self.__init__(new_config)
        print(f"[iHax] Switched to preset: {preset}")

    def get_status(self) -> dict:
        return {
            "primary": {
                "name": self.primary_name,
                "loaded": self._primary_model is not None,
            },
            "secondary": {
                "name": self.secondary_name,
                "loaded": self._secondary_model is not None,
            },
            "active": self._active,
            "simultaneous": self.simultaneous,
            "preset": self.config.get("preset", "unknown"),
            "gpu_budget_gb": self.config.get("gpu_budget_gb", 0),
        }


# ---------------------------------------------------------------------------
# CLI テスト
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  iHax MLX Engine — M5 Pro 64GB / 60GB GPU")
    print("=" * 60)
    print()

    # GPU割り当てチェック
    gpu_ready = setup_gpu_allocation()
    if not gpu_ready:
        print("[!] GPU制限の引き上げが必要です:")
        print(f"    sudo sysctl iogpu.wired_limit_mb={GPU_WIRED_LIMIT_MB}")
        print()

    config = get_hardware_config()
    print(f"Chip:         {config['chip']}")
    print(f"Physical RAM: {config['memory_gb']:.0f} GB")
    print(f"GPU Budget:   {config['gpu_budget_gb']} GB")
    print(f"OS Reserved:  {config['os_reserved_gb']} GB")
    print(f"SSD Swap:     {config['swap_gb']} GB (保険)")
    print()
    print(f"Preset:       {config['preset']}")
    print(f"              {config['preset_description']}")
    print(f"Primary:      {config['primary_model']}")
    print(f"              {config.get('primary_use_case', '')}")
    print(f"Secondary:    {config.get('secondary_model', 'None')}")
    print(f"Context:      {config['primary_context']} tokens")
    print(f"Usage:        ~{config['estimated_usage_gb']} / {config['gpu_budget_gb']} GB")
    print(f"Remaining:    ~{config['remaining_gpu_gb']:.0f} GB")
    print()

    print("--- Available presets ---")
    for name, p in PRESETS.items():
        cat = MODEL_CATALOG[p["primary"]]
        ctx = cat["max_context"]
        print(f"  {name:15s} | {p['description']:55s} | ctx {ctx:>5} | ~{cat['total_gb']}GB")
    print()

    if platform.system() == "Darwin":
        print("--- 推論テスト ---")
        engine = MLXEngine(config)
        result = engine.generate(
            "Write a Swift struct for a Todo app model with Codable conformance:",
            max_tokens=256,
            temperature=0.3,
        )
        print(f"Result:\n{result}")
    else:
        print("[SKIP] Apple Silicon Mac ではないため推論テストをスキップ")
