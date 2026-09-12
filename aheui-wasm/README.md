# aheui-wasm

아희 인터프리터와 컴파일러를 브라우저/Node.js 에서 실행하기 위한
wasm-bindgen 라이브러리, 브라우저 데모, 그리고 `wasm32-wasip1` CLI 빌드
가이드가 들어 있는 크레이트입니다.

## 경로 두 갈래

이 크레이트는 **같은 Rust 코어를 두 가지 wasm 타겟으로** 빌드할 수
있게 해 줍니다:

| 타겟 | 사용처 | 빌드 방법 |
|---|---|---|
| `wasm32-wasip1` | `aheui` CLI 를 wasmtime / WASI shim 위에서 실행 | `cargo build --target wasm32-wasip1 -p aheui --no-default-features --features naive,malachite-bigint --release` |
| `wasm32-unknown-unknown` | JS 에서 직접 호출하는 라이브러리 (브라우저 데모) | `./aheui-wasm/build.sh` |

`wasm32-wasip1` 빌드는 기존 CLI 를 그대로 (`run`, `asm`, `build`
서브커맨드 포함) wasm 으로 옮깁니다. JIT 과 네이티브 링크를 쓰는
백엔드 (`--emit link` 의 rust/c/cranelift) 는 wasm 런타임 안에서
실행할 수 없으므로 빠집니다.

`wasm32-unknown-unknown` 라이브러리는 JS 에서 호출할 개별 함수
(`interpret`, `compile_to_wat_web`, `compile_to_wasm_web`, ...) 를
`#[wasm_bindgen]` 으로 노출합니다.

## 빌드

필수 도구:

```sh
rustup target add wasm32-unknown-unknown wasm32-wasip1
cargo install wasm-bindgen-cli --version 0.2.127
```

라이브러리 (+ 브라우저/Node glue) 한 번에 빌드:

```sh
./aheui-wasm/build.sh          # release
./aheui-wasm/build.sh --dev    # debug
```

산출물:

- `aheui-wasm/pkg/`        — `--target web` (브라우저 ES 모듈)
- `aheui-wasm/pkg-node/`   — `--target nodejs` (Node 테스트용)

## 브라우저 데모

빌드 후 저장소 루트를 정적 파일 서버로 서빙하면 됩니다. 웹 데모가 같은
checkout의 `snippets` submodule을 직접 읽기 때문에 서버 루트도 저장소
루트여야 합니다:

```sh
./aheui-wasm/build.sh
python3 -m http.server 8000
# http://localhost:8000/aheui-wasm/web/
```

데모는 다음을 지원합니다:

- 소스 textarea + stdin textarea
- `예제 고르기` 드롭다운: hello / hello-world / factorial / fibonacci /
  99bottles / logo (`snippets/` submodule에서 fetch)
- **인터프리터** 버튼: `interpret(source, stdin)` 호출
- **WAT → wasm 실행** 버튼: `compile_to_wasm_web(source)` 로 wasm
  바이너리 생성 → `WebAssembly.instantiate` → `run()` 직접 실행
- **WAT / Asm / C / Rust** 뷰 버튼: 각 백엔드가 생성한 소스를 출력창에
  표시

## Node.js 테스트

```sh
./aheui-wasm/build.sh
cd aheui-wasm
node test_e2e.mjs       # 핵심 API 왕복 테스트
node test_samples.mjs   # 샘플 실행 회귀 테스트
```

## wasm-bindgen API

```rust
pub fn interpret(source: &str, input: &str) -> String
pub fn compile_to_wat_web(source: &str) -> String
pub fn compile_to_wat_wasi(source: &str) -> String
pub fn compile_to_c(source: &str) -> String
pub fn compile_to_rust(source: &str, opt_level: u8) -> String
pub fn compile_to_rust_bigint(source: &str, opt_level: u8) -> String
pub fn compile_to_asm(source: &str, opt_level: u8) -> String
pub fn wat_to_wasm(wat: &str) -> Result<Vec<u8>, JsValue>
pub fn compile_to_wasm_web(source: &str) -> Result<Vec<u8>, JsValue>
```

`interpret` 는 `wasm32-unknown-unknown` 에 stdin/stdout 이 없기 때문에
`aheui_runtime::io::wasm_buf` 의 thread-local 버퍼를 통해 입력 문자열을
주입하고 출력 문자열을 회수합니다.

`compile_to_wasm_web` 이 반환하는 wasm 모듈은 두 import 를 요구합니다:

```js
const imports = {
  env: {
    write_byte: (b) => { /* 한 바이트 출력 */ },
    read_byte: () => { /* 한 바이트 입력, EOF 면 -1 */ },
  },
};
const instance = await WebAssembly.instantiate(bytes, imports);
const exitCode = instance.exports.run();
```

## wasm32-wasip1 CLI 사용

```sh
cargo build --target wasm32-wasip1 -p aheui \
    --no-default-features --features naive,malachite-bigint --release

# 파일 실행
wasmtime run --dir .::/d \
    target/wasm32-wasip1/release/aheui.wasm /d/hello.aheui

# ahsembly 출력
wasmtime run --dir .::/d \
    target/wasm32-wasip1/release/aheui.wasm asm /d/hello.aheui

# WAT 생성 (wasm-in-wasm 컴파일)
wasmtime run --dir .::/d \
    target/wasm32-wasip1/release/aheui.wasm \
    build --codegen wasm32-web --emit source /d/hello.aheui
```

브라우저에서 WASI 로 돌리려면 `@bjorn3/browser_wasi_shim` 같은 shim 을
붙이면 됩니다. 이 디렉터리에는 WASI shim 기반 데모는 포함하지 않았고,
대신 `wasm32-unknown-unknown` 기반 직접 호출 데모만 제공합니다.

## 샘플 출처

샘플 프로그램은 [`aheui/snippets`](https://github.com/aheui/snippets)
submodule을 그대로 사용합니다. GitHub Pages 배포 단계도 이 디렉터리를
사이트에 복사하므로 별도의 샘플 복사본을 관리하지 않습니다.

## 라이선스

AGPL-3.0-or-later. 생성된 wasm 바이너리는 `aheui-runtime` 의 AGPL
컴포넌트를 포함하므로 배포 시 라이선스 의무를 따라야 합니다.
