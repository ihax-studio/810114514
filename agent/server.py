"""
iHax Agent Server
=================
Apple Silicon (M1/M2 Max) + MLX でローカルLLMを動かし、
PWAアプリ (iN glish, 英検2級, 数学) にAI機能を提供する。

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

from mlx_engine import MLXEngine, get_hardware_config

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="iHax Agent",
    description="ローカルLLM API for iHax educational platform",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # PWAからのlocalhost接続を許可
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Engine (遅延初期化)
# ---------------------------------------------------------------------------
engine: Optional[MLXEngine] = None


def get_engine() -> MLXEngine:
    global engine
    if engine is None:
        config = get_hardware_config()
        engine = MLXEngine(config)
    return engine


# ---------------------------------------------------------------------------
# Cache (簡易インメモリ)
# ---------------------------------------------------------------------------
_cache: dict[str, tuple[str, float]] = {}
CACHE_TTL = 3600  # 1時間


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
    model: str
    memory_gb: float
    chip: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    """ヘルスチェック + ハードウェア情報"""
    config = get_hardware_config()
    eng = get_engine()
    return HealthResponse(
        status="ok",
        model=eng.model_name,
        memory_gb=config["memory_gb"],
        chip=config["chip"],
    )


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
    """汎用テキスト生成 (Temperature + TopK)"""
    eng = get_engine()

    t0 = time.time()
    result = eng.generate(
        req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
    )
    elapsed = time.time() - t0
    n_tokens = len(result.split())  # 簡易トークン数推定

    return GenerateResponse(
        text=result,
        tokens_generated=n_tokens,
        tokens_per_second=n_tokens / max(elapsed, 0.001),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("  iHax Agent Server")
    print("  http://localhost:8000")
    print("  API docs: http://localhost:8000/docs")
    print("=" * 60)

    config = get_hardware_config()
    print(f"  Chip: {config['chip']}")
    print(f"  Memory: {config['memory_gb']} GB")
    print(f"  Model: {config['default_model']}")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000)
