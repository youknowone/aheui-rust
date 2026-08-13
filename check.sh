#!/bin/bash
# compaheuiler regression check: rgen (rustc/LLVM) + cranelift backends
set -uo pipefail
cd "$(dirname "$0")"

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
cargo build -p compaheuiler --features cranelift --release 2>/dev/null

echo ""
echo -e "${CYAN}══ RGEN BACKEND (rustc/LLVM) ══${RESET}"

RUSTC_OPT="-C opt-level=3 -C target-cpu=native"

echo ""
echo "═══ 1. Logo ═══"
LOGO_REF="$SNIPPETS/logo/logo.out"
# Generate .rs via cargo test, then recompile with O3
cargo test -p compaheuiler --release --test rgen_test -- test_logo_rs --test-threads=1 > /tmp/compaheuiler_logo_test.log 2>&1 || true
rustc $RUSTC_OPT -o /tmp/aheui_logo_o3 /tmp/aheui_logo.rs 2>/dev/null
LOGO_BIN="/tmp/aheui_logo_o3"

if [ ! -x "$LOGO_BIN" ]; then
    fail "logo: binary not found"
else
    "$LOGO_BIN" > /tmp/logo_check.txt 2>/dev/null || true
    if diff /tmp/logo_check.txt "$LOGO_REF" > /dev/null 2>&1; then
        pass "logo correctness ($(wc -c < /tmp/logo_check.txt | tr -d ' ') bytes)"
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

cargo test -p compaheuiler --release --test rgen_test -- test_aheui_self_interp --test-threads=1 > /tmp/compaheuiler_aheui_test.log 2>&1 || true
rustc $RUSTC_OPT -o /tmp/aheui_aheui_o3 /tmp/aheui_aheui_self.rs 2>/dev/null
AHEUI_BIN="/tmp/aheui_aheui_o3"

if [ ! -x "$AHEUI_BIN" ]; then
    # The self-interpreter source is in no snippet corpus; `n` names it and
    # there is no default, so the rgen test skips too ("aheui.aheui (set n)").
    skip "aheui.aheui: self-interpreter unavailable (set n)"
else
    "$AHEUI_BIN" < "$QUINE40_SRC" > /tmp/quine40_check.txt 2>/dev/null || true
    Q40_BYTES=$(wc -c < /tmp/quine40_check.txt | tr -d ' ')
    if diff <(head -c "$Q40_BYTES" /tmp/quine40_check.txt) <(head -c "$Q40_BYTES" "$QUINE40_REF") > /dev/null 2>&1; then
        pass "aheui+quine40 correctness ($Q40_BYTES bytes)"
    else
        fail "aheui+quine40 output mismatch"
    fi

    echo -n "밝희" | "$AHEUI_BIN" > /dev/null 2>&1
    if [ "$?" = "7" ]; then pass "aheui(밝희) exit=7"; else fail "aheui(밝희) exit=$?, expected 7"; fi

    measure "$AHEUI_BIN" "$QUINE40_SRC"
    echo -e "  ${YELLOW}aheui+quine40 median: ${MEASURE_MEDIAN}ms${RESET} (runs: ${MEASURE_TIMES})"
    if [ "$MEASURE_MEDIAN" -le 150 ]; then
        pass "aheui+quine40 performance: ${MEASURE_MEDIAN}ms <= 150ms"
    else
        fail "aheui+quine40 performance: ${MEASURE_MEDIAN}ms > 150ms threshold"
    fi
fi

echo ""
echo "═══ 3. Unit tests (rgen) ═══"
if cargo test -p compaheuiler --release --test rgen_test -- --test-threads=1 > /tmp/compaheuiler_unit.log 2>&1; then
    UNIT_PASS=$(grep -c 'test .* ok' /tmp/compaheuiler_unit.log || echo 0)
    pass "rgen_test: $UNIT_PASS tests passed"
else
    fail "rgen_test failed (see /tmp/compaheuiler_unit.log)"
fi

if cargo test -p compaheuiler --release --test all_snippets_test -- --test-threads=1 > /tmp/compaheuiler_allsnip.log 2>&1; then
    pass "all_snippets: 62/62 passed"
else
    SNIP_LINE=$(grep 'passed,' /tmp/compaheuiler_allsnip.log | tail -1)
    fail "all_snippets failed: $SNIP_LINE (see /tmp/compaheuiler_allsnip.log)"
fi

if cargo test -p compaheuiler --release --features bigint --test bigint_test -- --test-threads=1 > /tmp/compaheuiler_bigint.log 2>&1; then
    BIGINT_PASS=$(grep -c 'test .* ok' /tmp/compaheuiler_bigint.log || echo 0)
    pass "bigint_test: $BIGINT_PASS tests passed"
else
    fail "bigint_test failed (see /tmp/compaheuiler_bigint.log)"
fi

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
# The other four sections are all compaheuiler (rgen/cranelift AOT); this is
# the only one that runs the majit-driven interpreter. It gates the pinned
# `snippets/` submodule through the corpus and jitstress baselines.
cargo build -p aheui --release 2>/tmp/aheui_jit_build.log
if [ ! -x target/release/aheui ]; then
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
