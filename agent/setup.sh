#!/bin/bash
# ===========================================
# iHax Agent セットアップスクリプト
# Mac到着後にこれを実行すれば環境構築完了
# ===========================================

set -e

echo "=================================="
echo "  iHax Agent Setup"
echo "=================================="

# --- 1. Apple Silicon チェック ---
ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ]; then
    echo "[ERROR] Apple Silicon Mac が必要です (detected: $ARCH)"
    exit 1
fi

echo "[OK] Apple Silicon detected"

# --- 2. メモリチェック ---
MEM_BYTES=$(sysctl -n hw.memsize)
MEM_GB=$((MEM_BYTES / 1073741824))
echo "[INFO] Memory: ${MEM_GB} GB"

if [ "$MEM_GB" -lt 16 ]; then
    echo "[WARN] 16GB未満: 小型モデルのみ使用可能"
elif [ "$MEM_GB" -lt 32 ]; then
    echo "[OK] 16-32GB: 8Bモデルが実用的"
elif [ "$MEM_GB" -lt 48 ]; then
    echo "[OK] 32GB: 8B-13Bモデルが実用的"
else
    echo "[OK] 64GB: 33Bモデルまで対応可能"
fi

# --- 3. Python 環境 ---
echo ""
echo "--- Python環境構築 ---"

if ! command -v python3 &>/dev/null; then
    echo "[INFO] Python3 not found. Installing via Homebrew..."
    if ! command -v brew &>/dev/null; then
        echo "[INFO] Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    brew install python@3.11
fi

PYTHON_VERSION=$(python3 --version)
echo "[OK] $PYTHON_VERSION"

# --- 4. venv 作成 ---
echo ""
echo "--- 仮想環境 ---"

VENV_DIR="$(cd "$(dirname "$0")" && pwd)/.venv"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "[OK] venv created at $VENV_DIR"
else
    echo "[OK] venv already exists"
fi

source "$VENV_DIR/bin/activate"

# --- 5. 依存インストール ---
echo ""
echo "--- 依存パッケージ ---"

pip install --upgrade pip
pip install -r "$(dirname "$0")/requirements.txt"

echo "[OK] All dependencies installed"

# --- 6. モデルダウンロード ---
echo ""
echo "--- モデルダウンロード ---"
echo "[INFO] 初回はモデルのダウンロードに5-10分かかります"

python3 -c "
from mlx_lm import load
print('Downloading default model...')
model, tokenizer = load('mlx-community/Meta-Llama-3.1-8B-Instruct-4bit')
print('[OK] Model ready')
"

# --- 7. 起動テスト ---
echo ""
echo "=================================="
echo "  Setup Complete!"
echo "=================================="
echo ""
echo "起動方法:"
echo "  source $(dirname "$0")/.venv/bin/activate"
echo "  python $(dirname "$0")/server.py"
echo ""
echo "API: http://localhost:8000"
echo "Docs: http://localhost:8000/docs"
echo "=================================="
