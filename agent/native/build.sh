#!/bin/bash
# ===========================================
# iHax Capture — ビルドスクリプト
# M5 Pro / M6 Pro 最適化ビルド
# ===========================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT="${SCRIPT_DIR}/iHaxCapture"

echo "=================================="
echo "  Building iHax Capture"
echo "=================================="

# Apple Silicon 最適化フラグ:
#   -O           : 最適化
#   -target      : arm64 Apple Silicon ネイティブ
#   -Xcc -mcpu   : M5/M6 Pro の CPU 最適化
swiftc \
    -O \
    -target arm64-apple-macos14.0 \
    -framework Cocoa \
    -framework ScreenCaptureKit \
    -framework Vision \
    -o "$OUTPUT" \
    "${SCRIPT_DIR}/iHaxCapture.swift"

echo "[OK] Built: $OUTPUT"
echo ""
echo "起動: $OUTPUT"
echo "ホットキー: ⌘+Shift+X"
echo ""
echo "必要な権限:"
echo "  - スクリーン収録 (System Settings > Privacy > Screen Recording)"
echo "  - アクセシビリティ (System Settings > Privacy > Accessibility)"
echo ""
echo "iHax Agent Server が localhost:8000 で起動している必要があります"
echo "=================================="
