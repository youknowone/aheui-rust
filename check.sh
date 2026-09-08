#!/bin/bash
# compaheuiler regression check: rgen (rustc/LLVM) + cranelift backends
set -uo pipefail
export CARGO_BUILD_JOBS=1 RUST_TEST_THREADS=1 LLBC_PARALLEL_LAYOUTS=0
cd "$(dirname "$0")"
CHECK_TMP=$(mktemp -d "${TMPDIR:-/tmp}/aheui-check.XXXXXXXX") || exit 1
echo "Check artifacts: $CHECK_TMP"

if [ -n "${AHEUI_SNIPPETS:-}" ]; then
    SNIPPETS="$AHEUI_SNIPPETS"
elif [ -d snippets ]; then
    SNIPPETS="snippets"
elif [ -d rpaheui/snippets ]; then
    SNIPPETS="rpaheui/snippets"
else
    SNIPPETS=""
fi
if [ ! -d "$SNIPPETS" ]; then
    echo "snippet corpus not found; initialize the snippets submodule or set AHEUI_SNIPPETS" >&2
    exit 1
fi
PASS=0
FAIL=0
RED='\033[91m'
GREEN='\033[92m'
YELLOW='\033[93m'
CYAN='\033[96m'
RESET='\033[0m'

fail() { echo -e "  ${RED}FAIL${RESET}: $1"; FAIL=$((FAIL + 1)); }
pass() { echo -e "  ${GREEN}PASS${RESET}: $1"; PASS=$((PASS + 1)); }
# An absent optional input is not a regression. Counting it as a FAIL makes the
# exit code nonzero for a reason unrelated to what the script checks, which
# trains the reader to ignore the exit code entirely.
skip() { echo -e "  ${YELLOW}SKIP${RESET}: $1"; }

measure() {
    local bin="$1" stdin_file="${2:-}" n="${3:-3}"
    local times=()
    for i in $(seq 1 $n); do
        local ms
        if [ -n "$stdin_file" ]; then
            ms=$(python3 -c "
import subprocess, time
t0 = time.monotonic()
subprocess.run(['$bin'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=open('$stdin_file','rb'))
print(int((time.monotonic() - t0) * 1000))
")
        else
            ms=$(python3 -c "
import subprocess, time
t0 = time.monotonic()
subprocess.run(['$bin'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
print(int((time.monotonic() - t0) * 1000))
")
        fi
        times+=($ms)
    done
    MEASURE_TIMES="${times[*]}"
    MEASURE_MEDIAN=$(printf '%s\n' "${times[@]}" | sort -n | sed -n "$((($n+1)/2))p")
}

echo -e "${CYAN}Building compaheuiler (rgen + cranelift)...${RESET}"
cargo build -p compaheuiler --features cranelift --release || exit 1

echo ""
echo -e "${CYAN}══ RGEN BACKEND (rustc/LLVM) ══${RESET}"

RUSTC_OPT="-C opt-level=3 -C target-cpu=native"

echo ""
echo "═══ 1. Logo ═══"
LOGO_REF="$SNIPPETS/logo/logo.out"
# Generate .rs via cargo test, then recompile with O3
LOGO_BIN="$CHECK_TMP/logo"

if ! cargo test -p compaheuiler --release --test rgen_test -- test_logo_rs --test-threads=1 > "$CHECK_TMP/logo-build.log" 2>&1 ||
   ! rustc $RUSTC_OPT -o "$LOGO_BIN" target/codegen/aheui_logo.rs; then
    fail "logo: generation or compilation failed (see $CHECK_TMP)"
else
    if "$LOGO_BIN" > "$CHECK_TMP/logo.out" && cmp -s "$CHECK_TMP/logo.out" "$LOGO_REF"; then
        pass "logo correctness ($(wc -c < "$CHECK_TMP/logo.out" | tr -d ' ') bytes)"
    else
        fail "logo output mismatch"
    fi

    measure "$LOGO_BIN" "" 5
    echo -e "  ${YELLOW}logo median: ${MEASURE_MEDIAN}ms${RESET} (runs: ${MEASURE_TIMES})"
    if [ "$MEASURE_MEDIAN" -le 30 ]; then
        pass "logo performance: ${MEASURE_MEDIAN}ms <= 30ms"
    else
        fail "logo performance: ${MEASURE_MEDIAN}ms > 30ms threshold"
    fi
fi

echo ""
echo "═══ 2. aheui.aheui + quine40 ═══"
QUINE40_SRC="$SNIPPETS/quine/quine.puzzlet.40col.aheui"
QUINE40_REF="$SNIPPETS/quine/quine.puzzlet.40col.out"

AHEUI_BIN="$CHECK_TMP/self-interpreter"

if [ -z "${AHEUI_SELF_INTERP:-}" ]; then
    skip "aheui.aheui: self-interpreter unavailable (set AHEUI_SELF_INTERP)"
elif [ ! -r "$AHEUI_SELF_INTERP" ]; then
    fail "AHEUI_SELF_INTERP is not readable: $AHEUI_SELF_INTERP"
elif ! cargo test -p compaheuiler --release --test rgen_test -- test_aheui_self_interp --test-threads=1 > "$CHECK_TMP/self-build.log" 2>&1 ||
     ! rustc $RUSTC_OPT -o "$AHEUI_BIN" target/codegen/aheui_aheui_self.rs; then
    fail "self-interpreter: generation or compilation failed (see $CHECK_TMP)"
else
    if "$AHEUI_BIN" < "$QUINE40_SRC" > "$CHECK_TMP/quine40.out" &&
       cmp -s "$CHECK_TMP/quine40.out" "$QUINE40_REF"; then
        pass "aheui+quine40 correctness"
    else
        fail "aheui+quine40 output mismatch"
    fi

    echo -n "밝희" | "$AHEUI_BIN" > /dev/null 2>&1
    AHEUI_EXIT=$?
    if [ "$AHEUI_EXIT" = "7" ]; then pass "aheui(밝희) exit=7"; else fail "aheui(밝희) exit=$AHEUI_EXIT, expected 7"; fi

    measure "$AHEUI_BIN" "$QUINE40_SRC"
    echo -e "  ${YELLOW}aheui+quine40 median: ${MEASURE_MEDIAN}ms${RESET} (runs: ${MEASURE_TIMES})"
    if [ "$MEASURE_MEDIAN" -le 150 ]; then
        pass "aheui+quine40 performance: ${MEASURE_MEDIAN}ms <= 150ms"
    else
        fail "aheui+quine40 performance: ${MEASURE_MEDIAN}ms > 150ms threshold"
    fi
fi

echo ""
echo "═══ 3. Interpreter and compiler tests ═══"
for CRATE in aheuinterpreter compaheuiler; do
    LOG="/tmp/aheui_${CRATE}_test.log"
    if cargo test -p "$CRATE" --release < /dev/null > "$LOG" 2>&1; then
        pass "cargo test -p $CRATE: $(rg -c '^test .* ok$' "$LOG" || echo 0) tests passed"
    else
        fail "cargo test -p $CRATE failed (see $LOG)"
    fi
done

echo ""
echo -e "${CYAN}══ CRANELIFT BACKEND (JIT) ══${RESET}"

echo ""
echo "═══ 4. Cranelift unit tests ═══"
if cargo test -p compaheuiler --features cranelift --release --test cranelift_test -- --test-threads=1 --nocapture > /tmp/compaheuiler_cl.log 2>&1; then
    CL_PASS=$(grep -c '\.\.\..*ok' /tmp/compaheuiler_cl.log || echo 0)
    pass "cranelift_test: $CL_PASS tests passed"
    CL_ADD=$(grep 'cranelift add' /tmp/compaheuiler_cl.log | grep -oE '[0-9.]+ms' | head -1)
    CL_HELLO=$(grep 'cranelift hello' /tmp/compaheuiler_cl.log | grep -oE '[0-9.]+ms' | head -1)
    CL_LOGO=$(grep 'cranelift logo' /tmp/compaheuiler_cl.log | grep -oE '[0-9.]+ms' | head -1)
    echo -e "  ${YELLOW}add: ${CL_ADD:-?}, hello: ${CL_HELLO:-?}, logo: ${CL_LOGO:-?}${RESET}"
else
    fail "cranelift_test failed (see /tmp/compaheuiler_cl.log)"
    tail -10 /tmp/compaheuiler_cl.log
fi

echo ""
echo -e "${CYAN}══ MAJIT JIT ══${RESET}"

echo ""
echo "═══ 5. JIT stats floor + threshold sweep ═══"
# This is the only section that runs the majit-driven interpreter. Test its
# runtime and JIT crates directly before the end-to-end corpus gate: a defect
# in code that no corpus program reaches, such as storage GC helpers, is visible
# only to the crate tests. The corpus and jitstress baselines then gate the
# pinned `snippets/` submodule.
for CRATE in aheui-runtime aheui-jit; do
    LOG="/tmp/aheui_${CRATE}_test.log"
    if cargo test -p "$CRATE" --release < /dev/null > "$LOG" 2>&1; then
        pass "$CRATE: $(rg -c '^test .* ok$' "$LOG" || echo 0) tests passed"
    else
        fail "$CRATE tests failed (see $LOG)"
    fi
done

if ! cargo build -p aheui --release 2>/tmp/aheui_jit_build.log; then
    fail "aheui (majit) build failed (see /tmp/aheui_jit_build.log)"
else
    if python3 scripts/jitstats.py check; then
        pass "jit-stats floor + un-compiled A/B"
    else
        fail "jit-stats floor (record with scripts/jitstats.py record)"
    fi

    # `check` runs one threshold, the recorded one. Compilation shape is a
    # function of the threshold — which loops get hot, in which order, and
    # which of them a later trace closes into — so a threshold the baseline
    # does not name is a whole region of that space nothing runs. The sweep
    # is the same A/B against the un-compiled run at several thresholds, with
    # no baseline of its own: nothing is recorded and nothing can be blessed.
    if python3 scripts/jitstats.py sweep; then
        pass "jit A/B across thresholds"
    else
        fail "jit A/B across thresholds"
    fi

    # Both axes above count events — loops compiled, guards failed, traces
    # aborted. A codegen change moves none of them: the same loop compiles,
    # the same guards fail, the same bytes come out, and only the size of the
    # trace differs, by as much as a factor of seven. The op census is the
    # field that notices, and it counts ops rather than timing the run because
    # this host builds siblings concurrently and wall time swings with them.
    if python3 scripts/opcensus.py check; then
        pass "backend op census"
    else
        fail "backend op census (record with scripts/opcensus.py record)"
    fi
fi

echo ""
echo "════════════════════════════════"
TOTAL=$((PASS + FAIL))
if [ "$FAIL" -eq 0 ]; then
    echo -e "${GREEN}All $TOTAL checks passed!${RESET}"
else
    echo -e "${RED}$FAIL/$TOTAL checks failed${RESET}"
fi
exit $FAIL
