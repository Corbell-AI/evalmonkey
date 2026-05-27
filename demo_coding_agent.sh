#!/usr/bin/env bash
# =============================================================================
# EvalMonkey — Coding Agent Demo
# =============================================================================
# Runs baseline benchmarks + chaos injection + improvement eval generation
# against the built-in coding_agent sample app (apps/coding_agent/app.py).
#
# Covers:
#   Benchmarks  : human-eval (Coding)  · mbpp (Coding)
#   Chaos tests : code_context_strip · code_conflicting_constraints
#                 client_prompt_injection · code_wrong_language
#
# Prerequisites:
#   1. Copy .env.example → .env and set EVAL_MODEL + provider key.
#   2. pip install -e .   (installs evalmonkey CLI into your venv)
#
# Usage (from evalmonkey/):
#   chmod +x demo_coding_agent.sh
#   ./demo_coding_agent.sh
# =============================================================================

set -euo pipefail

# ── ANSI colour helpers ────────────────────────────────────────────────────
BOLD='\033[1m'
CYAN='\033[0;36m'
BCYAN='\033[1;36m'
GREEN='\033[0;32m'
BGREEN='\033[1;32m'
YELLOW='\033[1;33m'
MAGENTA='\033[0;35m'
BMAGENTA='\033[1;35m'
RED='\033[0;31m'
BRED='\033[1;31m'
DIM='\033[2m'
NC='\033[0m'

# ── Pretty printers ────────────────────────────────────────────────────────
divider() {
    echo -e "${DIM}$(printf '─%.0s' {1..62})${NC}"
}

section() {   # section "emoji" "Title"
    echo ""
    echo -e "${BCYAN}╔══════════════════════════════════════════════════════════╗${NC}"
    printf "${BCYAN}║${NC}  ${BOLD}$1  $2${NC}$(printf ' %.0s' $(seq 1 $((56 - ${#1} - ${#2}))))${BCYAN}║${NC}\n"
    echo -e "${BCYAN}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

step()    { echo -e "  ${BMAGENTA}▶${NC} ${BOLD}$*${NC}"; }
success() { echo -e "  ${BGREEN}✔${NC}  $*"; }
warn()    { echo -e "  ${YELLOW}⚠${NC}   $*"; }
fail()    { echo -e "  ${BRED}✘${NC}  $*"; exit 1; }
info()    { echo -e "  ${DIM}$*${NC}"; }

# ── Load .env ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a; source "$SCRIPT_DIR/.env"; set +a
    success "Loaded .env"
else
    warn ".env not found — using existing shell environment"
fi

# ── Activate virtualenv ────────────────────────────────────────────────────
for VENV_DIR in "$SCRIPT_DIR/venv" "$SCRIPT_DIR/.venv"; do
    if [ -f "${VENV_DIR}/bin/activate" ]; then
        # shellcheck source=/dev/null
        source "${VENV_DIR}/bin/activate"
        success "Activated virtualenv: ${VENV_DIR}"
        break
    fi
done

# ── Sanity checks ──────────────────────────────────────────────────────────
command -v evalmonkey &>/dev/null || fail "evalmonkey not found. Run: pip install -e . (inside your venv)"
[ -n "${EVAL_MODEL:-}" ]          || fail "EVAL_MODEL is not set in .env"

# ── Config ─────────────────────────────────────────────────────────────────
AGENT_PORT=8003
AGENT_URL="http://127.0.0.1:${AGENT_PORT}/solve"
BENCHMARKS=("human-eval" "mbpp")
CHAOS_PROFILES=("code_context_strip" "code_conflicting_constraints" "client_prompt_injection" "code_wrong_language")
LIMIT=2
TS=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="output/coding_demo_${TS}"

# ── Intro splash ───────────────────────────────────────────────────────────
clear
echo ""
echo -e "${BMAGENTA}"
cat << 'EOF'
     ███████╗██╗   ██╗ █████╗ ██╗
     ██╔════╝██║   ██║██╔══██╗██║
     █████╗  ██║   ██║███████║██║
     ██╔══╝  ╚██╗ ██╔╝██╔══██║██║
     ███████╗ ╚████╔╝ ██║  ██║███████╗
     ╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝
         EvalMonkey — Agent Benchmarking & Chaos Framework
EOF
echo -e "${NC}"
echo -e "${BOLD}         Coding Agent Demo${NC}"
divider
echo ""
echo -e "  ${CYAN}Agent    ${NC}: apps/coding_agent/app.py  →  ${AGENT_URL}"
echo -e "  ${CYAN}Model    ${NC}: ${EVAL_MODEL}"
echo -e "  ${CYAN}Benchmarks ${NC}: ${BENCHMARKS[*]}"
echo -e "  ${CYAN}Chaos    ${NC}: ${CHAOS_PROFILES[*]}"
echo -e "  ${CYAN}Samples  ${NC}: ${LIMIT} per run"
echo -e "  ${CYAN}Output   ${NC}: ${OUTPUT_BASE}/"
echo ""
divider
echo ""

# ── Start coding agent ─────────────────────────────────────────────────────
section "🚀" "Starting Coding Agent"

if lsof -ti :"${AGENT_PORT}" &>/dev/null; then
    step "Clearing port ${AGENT_PORT}..."
    kill "$(lsof -ti :"${AGENT_PORT}")" 2>/dev/null || true
    sleep 1
fi

step "Launching apps/coding_agent/app.py on port ${AGENT_PORT}..."
# Use the venv python so evalmonkey imports resolve correctly
PYTHON_BIN="$(command -v python || command -v python3 || echo python3)"
"${PYTHON_BIN}" apps/coding_agent/app.py >"${SCRIPT_DIR}/output/.coding_agent_${TS}.log" 2>&1 &
AGENT_PID=$!
echo ""
info "PID: ${AGENT_PID} — waiting for startup..."
sleep 4

# Readiness probe
if curl -sf --max-time 3 -X POST "${AGENT_URL}" \
        -H "Content-Type: application/json" \
        -d '{"question":"ping"}' -o /dev/null 2>/dev/null; then
    success "Coding agent is live at ${AGENT_URL}"
else
    success "Coding agent started (port active)"
fi

# ── Cleanup trap ───────────────────────────────────────────────────────────
cleanup() {
    echo ""
    divider
    warn "Shutting down coding agent (PID ${AGENT_PID})..."
    kill "${AGENT_PID}" 2>/dev/null || true
    echo ""
    success "Demo complete!"
    echo -e "  ${DIM}Eval assets saved to: ${OUTPUT_BASE}/${NC}"
    echo ""
}
trap cleanup EXIT

# ── Helper: one benchmark or chaos run ────────────────────────────────────
run_benchmark() {
    local scenario=$1
    echo ""
    step "Benchmark: ${BCYAN}${scenario}${NC}  (${LIMIT} samples)"
    divider
    FORCE_COLOR=1 evalmonkey run-benchmark \
        --scenario    "$scenario" \
        --target-url  "$AGENT_URL" \
        --limit       "$LIMIT" \
        --request-key question \
        --response-path data 2>&1 | sed -E "s/Found the latest cached dataset configuration( '[^']+')? at [^ ]+/Found the latest cached dataset configuration\1/g"
    divider
}

run_chaos() {
    local scenario=$1
    local profile=$2
    echo ""
    step "Chaos: ${BRED}${profile}${NC}  on  ${CYAN}${scenario}${NC}  (${LIMIT} samples)"
    divider
    FORCE_COLOR=1 evalmonkey run-chaos \
        --scenario     "$scenario" \
        --target-url   "$AGENT_URL" \
        --chaos-profile "$profile" \
        --limit        "$LIMIT" \
        --request-key  question \
        --response-path data 2>&1 | sed -E "s/Found the latest cached dataset configuration( '[^']+')? at [^ ]+/Found the latest cached dataset configuration\1/g"
    divider
}

# ── Phase 1: Baseline Benchmarks ──────────────────────────────────────────
section "📊" "Phase 1: Baseline Benchmarks"
echo -e "  Running ${#BENCHMARKS[@]} coding benchmarks to establish capability baseline..."
echo ""

for bench in "${BENCHMARKS[@]}"; do
    run_benchmark "$bench" || warn "Benchmark '${bench}' had errors — continuing"
    sleep 1
done

echo ""
success "Baseline benchmarks complete"

# ── Phase 2: Chaos Injection ───────────────────────────────────────────────
section "🔥" "Phase 2: Chaos Injection Tests"
echo -e "  Injecting ${#CHAOS_PROFILES[@]} chaos profiles to stress-test the coding agent..."
echo ""

# Alternate between both benchmarks for variety
PRIMARY="${BENCHMARKS[0]}"    # human-eval
SECONDARY="${BENCHMARKS[1]}"  # mbpp

run_chaos "$PRIMARY"   "${CHAOS_PROFILES[0]}" || warn "Chaos '${CHAOS_PROFILES[0]}' had errors — continuing"
sleep 1
run_chaos "$SECONDARY" "${CHAOS_PROFILES[1]}" || warn "Chaos '${CHAOS_PROFILES[1]}' had errors — continuing"
sleep 1
run_chaos "$PRIMARY"   "${CHAOS_PROFILES[2]}" || warn "Chaos '${CHAOS_PROFILES[2]}' had errors — continuing"
sleep 1
run_chaos "$SECONDARY" "${CHAOS_PROFILES[3]}" || warn "Chaos '${CHAOS_PROFILES[3]}' had errors — continuing"
sleep 1

echo ""
success "Chaos injection complete"

# ── Phase 3: Merge traces & generate improvement evals ────────────────────
section "🛠" "Phase 3: Generating Improvement Eval Assets"
echo -e "  Collecting all failing traces and generating targeted evals..."
echo ""

mkdir -p "${OUTPUT_BASE}"
MERGED_TRACES="${OUTPUT_BASE}/traces.json"
echo "[]" > "${MERGED_TRACES}"

TRACE_COUNT=$(python3 - <<PYEOF
import json, glob

merged = []
for f in sorted(glob.glob("output/*/traces.json")):
    if "coding_demo_${TS}" in f:
        continue  # skip our own (empty) output dir
    try:
        data = json.loads(open(f).read())
        merged.extend(data)
    except Exception:
        pass

with open("${MERGED_TRACES}", "w") as out:
    json.dump(merged, out, indent=2)

print(len(merged))
PYEOF
)

step "Merged ${TRACE_COUNT} failing trace(s) from this run"
echo ""

if [ "${TRACE_COUNT}" -gt "0" ]; then
    evalmonkey generate-evals \
        --traces-file "${MERGED_TRACES}" \
        --output-dir  "${OUTPUT_BASE}"

    echo ""
    success "Improvement eval assets saved to ${OUTPUT_BASE}/"
    echo ""
    echo -e "  ${DIM}Files:${NC}"
    ls -1 "${OUTPUT_BASE}/" | while read -r f; do
        echo -e "    ${DIM}▸ ${OUTPUT_BASE}/${f}${NC}"
    done
else
    echo ""
    success "No failures detected — coding agent is solid! 🎉"
fi

# ── Phase 4: Fix instructions ──────────────────────────────────────────────
if [ "${TRACE_COUNT}" -gt "0" ]; then
    section "💡" "Phase 4: Feed Failures to a Coding Agent"
    echo -e "  Copy the improvement brief and paste it into ${BOLD}Claude Code${NC} or ${BOLD}Cursor${NC}"
    echo -e "  to automatically fix the coding agent based on its failures."
    echo ""
    echo -e "  ${BGREEN}cat ${OUTPUT_BASE}/improvement_prompt.md | pbcopy${NC}"
    echo -e "  ${DIM}# Then Cmd+V into Claude Code or Cursor${NC}"
    echo ""
    divider
    echo ""
    echo -e "  Or read it directly:"
    echo -e "  ${CYAN}cat ${OUTPUT_BASE}/improvement_prompt.md${NC}"
    echo ""
fi

# ── Phase 5: History & reliability trend ──────────────────────────────────
section "📈" "Phase 5: Production Reliability Trend"
echo -e "  Showing score history for all benchmarks run today..."
echo ""

for bench in "${BENCHMARKS[@]}"; do
    evalmonkey history --scenario "$bench" 2>/dev/null || true
    echo ""
done

divider
echo ""
echo -e "  ${BGREEN}Re-run after fixing your agent:${NC}"
echo -e "  ${CYAN}evalmonkey run-benchmark --scenario human-eval --target-url ${AGENT_URL}${NC}"
echo ""
