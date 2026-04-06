#!/usr/bin/env bash
# aheui-wasm 빌드 스크립트
#
# 1. wasm32-unknown-unknown 타겟으로 aheui-wasm crate 을 빌드
# 2. wasm-bindgen 으로 `pkg/` (브라우저), `pkg-node/` (node 테스트) glue 생성
#
# 사용법:
#   ./build.sh              # release 빌드
#   ./build.sh --dev        # dev 빌드
#
# 요구 도구:
#   rustup target add wasm32-unknown-unknown
#   cargo install wasm-bindgen-cli --version 0.2.117

set -euo pipefail

cd "$(dirname "$0")/.."   # aheui workspace root

PROFILE="release"
CARGO_PROFILE_FLAG="--release"
if [[ "${1:-}" == "--dev" ]]; then
  PROFILE="debug"
  CARGO_PROFILE_FLAG=""
fi

echo ">>> cargo build --target wasm32-unknown-unknown -p aheui-wasm ${CARGO_PROFILE_FLAG}"
cargo build --target wasm32-unknown-unknown -p aheui-wasm ${CARGO_PROFILE_FLAG}

WASM="target/wasm32-unknown-unknown/${PROFILE}/aheui_wasm.wasm"
OUT_WEB="aheui-wasm/pkg"
OUT_NODE="aheui-wasm/pkg-node"

echo ">>> wasm-bindgen --target web -> ${OUT_WEB}"
wasm-bindgen "${WASM}" --out-dir "${OUT_WEB}" --target web --no-typescript

echo ">>> wasm-bindgen --target nodejs -> ${OUT_NODE}"
wasm-bindgen "${WASM}" --out-dir "${OUT_NODE}" --target nodejs --no-typescript

echo
echo "빌드 완료."
echo "  브라우저 데모: cd aheui-wasm && python3 -m http.server 8000, 이후 http://localhost:8000/web/"
echo "  node 테스트:   cd aheui-wasm && node test_e2e.mjs"
