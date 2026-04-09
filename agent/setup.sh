#!/bin/bash
# ===========================================
# iHax Agent セットアップスクリプト
# M5 Pro 64GB — 60GB GPU割り当て + 400GB SSD
#
# Mac到着後にこれを実行すれば環境構築完了
# ===========================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================"
echo "  iHax Agent Setup — M5 Pro 64GB / 60GB GPU / 400GB SSD"
echo "============================================================"

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
echo "[INFO] Physical Memory: ${MEM_GB} GB"

if [ "$MEM_GB" -lt 64 ]; then
    echo "[WARN] 64GB未満: 60GB GPU割り当ては不可"
    echo "[WARN] 利用可能: $((MEM_GB * 3 / 4)) GB (75%制限)"
else
    echo "[OK] 64GB: 60GB GPU割り当て対応"
fi

# --- 3. GPU メモリ制限の引き上げ (60GB) ---
echo ""
echo "--- GPU メモリ割り当て ---"

CURRENT_LIMIT=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo "0")
TARGET_LIMIT=61440  # 60GB

if [ "$MEM_GB" -ge 64 ]; then
    if [ "$CURRENT_LIMIT" -lt "$TARGET_LIMIT" ]; then
        echo "[INFO] GPU制限を ${CURRENT_LIMIT}MB → ${TARGET_LIMIT}MB (60GB) に引き上げます"
        echo "[INFO] macOS用: 4GB物理 + 20GB SSD swap (保険)"
        echo ""
        echo "  sudo sysctl iogpu.wired_limit_mb=${TARGET_LIMIT}"
        echo ""
        echo "[INFO] sudoパスワードが必要です。再起動で元に戻ります。"

        sudo sysctl iogpu.wired_limit_mb=${TARGET_LIMIT}

        # 仮想メモリ (swap) 設定確認
        echo ""
        echo "[INFO] 仮想メモリ (swap) 状態:"
        sysctl vm.swapusage
        echo "[OK] macOSは必要時にSSD上にswapを自動作成します (15GB/s)"
    else
        echo "[OK] GPU制限は既に ${CURRENT_LIMIT}MB に設定済み"
    fi
else
    echo "[SKIP] 64GB未満のためGPU制限引き上げをスキップ"
fi

# --- 4. Python 環境 ---
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

# --- 5. venv 作成 ---
echo ""
echo "--- 仮想環境 ---"

VENV_DIR="${SCRIPT_DIR}/.venv"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "[OK] venv created at $VENV_DIR"
else
    echo "[OK] venv already exists"
fi

source "$VENV_DIR/bin/activate"

# --- 6. 依存インストール ---
echo ""
echo "--- 依存パッケージ ---"

pip install --upgrade pip
pip install -r "${SCRIPT_DIR}/requirements.txt"

echo "[OK] All dependencies installed"

# --- 7. モデルダウンロード (SSD 400GB に保管) ---
echo ""
echo "--- モデルダウンロード (SSD 400GB計画) ---"
echo ""
echo "  全モデル合計: ~202GB / 400GB"
echo "  KV overflow:  ~50GB"
echo "  FineTune:     ~50GB"
echo "  余り:         ~98GB"
echo ""

# まず8B (最小・最速。動作確認用)
echo "[1/3] 8B model (6GB) — 動作確認 + 記憶管理用"
python3 -c "
from mlx_lm import load
model, tok = load('mlx-community/Meta-Llama-3.1-8B-Instruct-4bit')
print('[OK] 8B model ready')
"

# Coder 32B (コード特化)
echo ""
echo "[2/3] Coder 32B model (22GB) — Swift/コード生成用"
python3 -c "
from mlx_lm import load
model, tok = load('mlx-community/Qwen2.5-Coder-32B-Instruct-4bit')
print('[OK] Coder 32B model ready')
"

# 70B 4bit (メイン推論)
echo ""
echo "[3/3] 70B 4bit model (42GB) — メイン推論用"
echo "[INFO] これは大きいので30分程度かかる場合があります"
python3 -c "
from mlx_lm import load
model, tok = load('mlx-community/Meta-Llama-3.1-70B-Instruct-4bit')
print('[OK] 70B 4bit model ready')
"

echo ""
echo "残りのモデル (70B 5bit/6bit) は必要に応じてダウンロード:"
echo "  python3 -c \"from mlx_lm import load; load('mlx-community/Meta-Llama-3.1-70B-Instruct-5bit')\""
echo "  python3 -c \"from mlx_lm import load; load('mlx-community/Meta-Llama-3.1-70B-Instruct-6bit')\""

# --- 8. SSD空き容量チェック ---
echo ""
echo "--- SSD空き容量 ---"
df -h / | tail -1
echo ""

# --- 9. 完了 ---
echo "============================================================"
echo "  Setup Complete!"
echo "============================================================"
echo ""
echo "  GPU割り当て: 60GB / ${MEM_GB}GB"
echo "  macOS用:     4GB物理 + SSD swap"
echo "  モデル保管:  SSD上 (~200GB)"
echo ""
echo "起動方法:"
echo "  source ${SCRIPT_DIR}/.venv/bin/activate"
echo "  python ${SCRIPT_DIR}/server.py"
echo ""
echo "プリセット切り替え (API):"
echo "  curl -X POST localhost:8000/preset -d '{\"preset\":\"architect\"}'   # 70B 6bit OS設計"
echo "  curl -X POST localhost:8000/preset -d '{\"preset\":\"developer\"}'   # 70B 5bit 開発"
echo "  curl -X POST localhost:8000/preset -d '{\"preset\":\"million\"}'     # 8B 1Mコンテキスト"
echo "  curl -X POST localhost:8000/preset -d '{\"preset\":\"balanced\"}'    # Coder32B+8B"
echo ""
echo "API:  http://localhost:8000"
echo "Docs: http://localhost:8000/docs"
echo "============================================================"
