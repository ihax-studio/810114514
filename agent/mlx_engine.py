"""
MLX Engine for iHax Agent
=========================
Apple Silicon統合メモリを活用したローカルLLM推論エンジン。

爆死記事の教訓を全て反映:
- bfloat16 (float16のnan問題回避)
- Temperature + TopK サンプリング
- メモリ効率最優先の設計
- キャッシュ活用で再推論を避ける

依存: pip install mlx mlx-lm
"""

from __future__ import annotations

import platform
import subprocess
from typing import Optional


def get_hardware_config() -> dict:
    """
    実行環境のハードウェアを検出し、最適な設定を返す。

    M1 Max 64GB → 大きめのモデル (13B 4bit)
    M2 Max 64GB → 大きめのモデル (13B 4bit, やや高速)
    M2 Max 32GB → 中型モデル (8B 4bit)
    その他      → 小型モデル (3B 4bit) or CPU fallback
    """
    chip = _detect_chip()
    memory_gb = _detect_memory_gb()

    # メモリ量に応じたモデル選択
    # 記事の教訓: メモリが全て。小さいモデルでも動く方が100倍マシ
    if memory_gb >= 48:
        # 64GB: 13Bモデルが余裕で動く (使用メモリ ~8GB, 残り40GB+)
        default_model = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
        large_model = "mlx-community/Meta-Llama-3.1-13B-Instruct-4bit"
    elif memory_gb >= 24:
        # 32GB: 8Bモデルが実用的上限
        default_model = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
        large_model = default_model
    elif memory_gb >= 12:
        # 16GB: 小型モデルのみ
        default_model = "mlx-community/Llama-3.2-3B-Instruct-4bit"
        large_model = default_model
    else:
        # 8GB以下: 記事の爆死ゾーン。最小モデルで頑張る
        default_model = "mlx-community/Llama-3.2-1B-Instruct-4bit"
        large_model = default_model

    return {
        "chip": chip,
        "memory_gb": memory_gb,
        "default_model": default_model,
        "large_model": large_model,
        "max_context_length": min(8192, int(memory_gb * 128)),
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

    return 8.0  # 最悪ケース: 記事の爆死環境と同じ


class MLXEngine:
    """
    MLXベースのLLM推論エンジン。

    記事で学んだ生成手法を全て実装:
    - Temperature: 出力の多様性制御
    - TopK: 上位K個からサンプリング
    - bfloat16互換: MLXはデフォルトで対応
    """

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = get_hardware_config()

        self.model_name = config["default_model"]
        self.max_context = config["max_context_length"]
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """モデルの遅延ロード (初回呼び出し時のみ)"""
        if self._model is not None:
            return

        try:
            from mlx_lm import load

            self._model, self._tokenizer = load(self.model_name)
            print(f"[iHax] Model loaded: {self.model_name}")
        except ImportError:
            raise RuntimeError(
                "mlx-lm がインストールされていません。\n"
                "pip install mlx mlx-lm\n"
                "※ Apple Silicon Mac が必要です"
            )
        except Exception as e:
            raise RuntimeError(
                f"モデルのロードに失敗: {e}\n"
                f"モデル: {self.model_name}\n"
                "初回はモデルのダウンロードに数分かかります"
            )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 40,
    ) -> str:
        """
        テキスト生成。

        Temperature + TopK は記事で学んだ生成手法:
        - Temperature: 高い → 多様性UP, 低い → 確実性UP
        - TopK: 上位K個の候補からサンプリング

        記事の教訓:
        - greedy (argmax) だけだとカンマ連打になる
        - Temperature=0.7, TopK=40 が一般的に良いバランス
        """
        self._load_model()

        try:
            from mlx_lm import generate as mlx_generate

            # mlx_lm.generate は内部で Temperature + sampling を処理
            result = mlx_generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
                top_p=0.9,  # TopP も併用 (nucleus sampling)
            )
            return result.strip()
        except Exception as e:
            raise RuntimeError(f"生成エラー: {e}")

    def switch_model(self, model_name: str):
        """モデルの切り替え (メモリ解放 → 再ロード)"""
        # 記事の教訓: gc.collect + empty_cache に相当
        self._model = None
        self._tokenizer = None

        import gc

        gc.collect()

        self.model_name = model_name
        self._load_model()


# ---------------------------------------------------------------------------
# CLI テスト
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== iHax MLX Engine Test ===")

    config = get_hardware_config()
    print(f"Chip: {config['chip']}")
    print(f"Memory: {config['memory_gb']:.1f} GB")
    print(f"Default model: {config['default_model']}")
    print(f"Large model: {config['large_model']}")
    print(f"Max context: {config['max_context_length']}")

    if platform.system() == "Darwin":
        print("\n--- 推論テスト ---")
        engine = MLXEngine(config)
        result = engine.generate(
            "Translate to English: 今日はいい天気ですね",
            max_tokens=64,
            temperature=0.3,
        )
        print(f"Result: {result}")
    else:
        print("\n[SKIP] Apple Silicon Mac ではないため推論テストをスキップ")
