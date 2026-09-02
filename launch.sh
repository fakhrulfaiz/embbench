#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
    echo "Do not run with sudo. Use: bash $0"
    exit 1
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$REPO/docker-compose.yml"
UV="${UV:-$HOME/.local/bin/uv}"
if [[ ! -x "$UV" ]]; then
    UV="$(command -v uv 2>/dev/null || true)"
fi
if [[ ! -x "$UV" ]]; then
    echo "uv not found (tried \$HOME/.local/bin/uv). Install uv or set UV=."
    exit 1
fi

dc() {
    docker compose -f "$COMPOSE_FILE" --project-directory "$REPO" "$@"
}

dc_chat() {
    dc --profile chat "$@"
}

load_env() {
    if [[ ! -f "$REPO/.env" ]]; then
        echo "Missing $REPO/.env (copy .env.example)."
        exit 1
    fi
    set -a
    # shellcheck disable=SC1091
    source "$REPO/.env"
    set +a
    LLM_PORT="${LLM_PORT:-8000}"
    EMBED_PORT="${EMBED_PORT:-8001}"
    HEALTH_URL="http://127.0.0.1:${LLM_PORT}/health"
    EMBED_HEALTH_URL="http://127.0.0.1:${EMBED_PORT}/health"
    GEN_COUNT="${GEN_COUNT:-200}"
    GEN_LANGUAGE="${GEN_LANGUAGE:-en}"
    GEN_CONCURRENCY="${GEN_CONCURRENCY:-${LLM_MAX_REQ:-4}}"
    RETRIEVAL_NAME="${RETRIEVAL_NAME:-sop-handbook-v1}"
    STS_NAME="${STS_NAME:-sop-sts-v1}"
    EMBEDDING_MODEL="${EMBEDDING_MODEL:-bce-embedding-base_v1}"
    EMBBENCH_ROOT="$REPO"
    MODELS_YAML="$EMBBENCH_ROOT/configs/models.yaml"
    DATA_DIR="${EXPORT_DIR:-$EMBBENCH_ROOT/data}"
    EMBED_MODELS_ROOT="${EMBED_MODELS_ROOT:-/usr/local/models}"
    EMBED_HF_CACHE="${EMBED_HF_CACHE:-$HOME/.cache/huggingface}"
    DASHBOARD_PORT="${DASHBOARD_PORT:-8501}"
    LOG_DIR="$REPO/logs"
    mkdir -p "$LOG_DIR"
    LAUNCH_LOG="$LOG_DIR/launch.log"
    DASH_PID_FILE="$LOG_DIR/dashboard.pid"
    DASH_PORT_FILE="$LOG_DIR/dashboard.port"
}

log() {
    printf '%s %s\n' "$(date -Iseconds)" "$*" >> "${LAUNCH_LOG:-/dev/null}"
}

healthy() {
    curl -sf -o /dev/null --max-time 2 "$1"
}

# Stop any container publishing this host port (old compose project names included).
free_port() {
    local port="$1"
    local ids names
    ids="$(docker ps -q --filter "publish=${port}" 2>/dev/null || true)"
    [[ -z "$ids" ]] && return 0
    names="$(docker ps --filter "publish=${port}" --format '{{.Names}}' | tr '\n' ' ')"
    echo "Port $port in use ($names)—stopping so this compose project can bind."
    # shellcheck disable=SC2086
    docker stop $ids >/dev/null
}

stop_vllm() {
    if [[ -n "$(dc_chat ps --status=running -q vllm 2>/dev/null || true)" ]]; then
        echo "Stopping chat vLLM (GPU for embeddings)."
        dc_chat stop vllm
    fi
    free_port "${LLM_PORT:-8000}"
}

stop_embed() {
    if [[ -n "$(dc --profile embed ps --status=running -q embed 2>/dev/null || true)" ]]; then
        echo "Stopping embedding vLLM (GPU for chat)."
        dc --profile embed stop embed
    fi
    free_port "${EMBED_PORT:-8001}"
}

wait_healthy() {
    local url="$1"
    local label="$2"
    local service="$3"
    local i
    echo "waiting for $label..."
    log "waiting for $label at $url"
    for i in $(seq 1 60); do
        if healthy "$url"; then
            echo "$label ready."
            log "$label ready after $i tries"
            return 0
        fi
        log "wait $label $i/60"
        sleep 5
    done
    echo "$label did not become healthy in time. See $LAUNCH_LOG"
    log "$label failed after 60 tries"
    dc --profile embed logs --tail 80 "$service" 2>/dev/null || dc_chat logs --tail 80 "$service" || true
    exit 1
}

ensure_vllm() {
    stop_embed
    if healthy "$HEALTH_URL"; then
        echo "chat vLLM already serving ($HEALTH_URL)."
        return 0
    fi
    echo "Starting chat vLLM..."
    dc_chat up -d vllm
    wait_healthy "$HEALTH_URL" "chat vLLM" vllm
}

model_hf_name() {
    local id="$1"
    awk -v want="$id" '
        /^  - id:/{cur=$3}
        cur == want && /hf_name:/{print $2; exit}
    ' "$MODELS_YAML"
}

# Local dir if it has config.json; otherwise Hub id (vLLM downloads via EMBED_HF_CACHE).
set_embed_flags() {
    local id="$1"
    EMBED_CONVERT=embed
    EMBED_DTYPE=float16
    EMBED_ENFORCE_EAGER=0
    EMBED_HF_OVERRIDES=
    EMBED_POOLER_CONFIG='{"pooling_type":"CLS"}'
    EMBED_PREFIX_CACHE=1
    EMBED_MAX_BATCHED_TOKENS=
    case "$id" in
        bce-embedding-base_v1)
            EMBED_DTYPE=float16
            EMBED_POOLER_CONFIG='{"pooling_type":"CLS"}'
            ;;
        bge-m3)
            EMBED_DTYPE=float16
            EMBED_POOLER_CONFIG='{"pooling_type":"CLS"}'
            ;;
        voyage-4-nano)
            EMBED_DTYPE=bfloat16
            EMBED_ENFORCE_EAGER=1
            EMBED_HF_OVERRIDES='{"architectures":["VoyageQwen3BidirectionalEmbedModel"]}'
            EMBED_POOLER_CONFIG='{"pooling_type":"MEAN"}'
            ;;
        harrier-oss-v1-0.6b)
            EMBED_DTYPE=bfloat16
            EMBED_POOLER_CONFIG='{"pooling_type":"LAST"}'
            # Causal + LAST pooling: keep every request in one prefill step, or
            # long document batches wedge the engine (see embed-serve.sh).
            EMBED_PREFIX_CACHE=0
            EMBED_MAX_BATCHED_TOKENS=$((EMBED_MAX_REQ * EMBED_MAX_LEN))
            ;;
        Qwen3-Embedding-0.6B)
            EMBED_DTYPE=bfloat16
            EMBED_POOLER_CONFIG='{"pooling_type":"LAST"}'
            EMBED_PREFIX_CACHE=0
            EMBED_MAX_BATCHED_TOKENS=$((EMBED_MAX_REQ * EMBED_MAX_LEN))
            ;;
        *)
            echo "No vLLM pooling preset for '$id'; using CLS float16."
            ;;
    esac
}

set_embed_env() {
    local id="$1"
    local hf
    hf="$(model_hf_name "$id")"
    if [[ -z "$hf" ]]; then
        echo "Unknown model id '$id' (not in $MODELS_YAML)."
        exit 1
    fi
    EMBED_NAME="$hf"
    local dir=""
    if [[ -f "$EMBED_MODELS_ROOT/$id/config.json" ]]; then
        dir="$EMBED_MODELS_ROOT/$id"
    elif [[ -f "$EMBED_MODELS_ROOT/${hf##*/}/config.json" ]]; then
        dir="$EMBED_MODELS_ROOT/${hf##*/}"
    fi
    if [[ -n "$dir" ]]; then
        EMBED_PATH="$dir"
        EMBED_SERVE="/models"
        echo "embed weights: $dir (served as $EMBED_NAME)"
    else
        EMBED_PATH="${EMBED_PATH:-$EMBED_MODELS_ROOT/bce-embedding-base_v1}"
        EMBED_SERVE="$hf"
        echo "embed weights: Hub $hf (served as $EMBED_NAME)"
    fi
    set_embed_flags "$id"
    echo "embed flags: dtype=$EMBED_DTYPE pooler=$EMBED_POOLER_CONFIG eager=$EMBED_ENFORCE_EAGER prefix_cache=$EMBED_PREFIX_CACHE max_batched_tokens=${EMBED_MAX_BATCHED_TOKENS:-default}"
    export EMBED_PATH EMBED_SERVE EMBED_NAME EMBED_PORT EMBED_MAX_LEN EMBED_GPU_MEMORY EMBED_MAX_REQ EMBED_HF_CACHE
    export EMBED_DTYPE EMBED_CONVERT EMBED_POOLER_CONFIG EMBED_HF_OVERRIDES EMBED_ENFORCE_EAGER
    export EMBED_PREFIX_CACHE EMBED_MAX_BATCHED_TOKENS
}

ensure_embed() {
    local id="$1"
    stop_vllm
    stop_embed
    set_embed_env "$id"
    echo "Starting embedding vLLM for $id..."
    dc --profile embed up -d --force-recreate --no-deps embed
    wait_healthy "$EMBED_HEALTH_URL" "embedding vLLM" embed
}

embbench() {
    (cd "$REPO" && "$UV" run embbench "$@")
}

ask() {
    local prompt="$1"
    local default="${2:-}"
    local value
    if [[ -n "$default" ]]; then
        read -r -p "$prompt [$default]: " value
        echo "${value:-$default}"
    else
        read -r -p "$prompt: " value
        echo "$value"
    fi
}

want_manual() {
    local reply
    read -r -p "Enter parameters manually? [y/N]: " reply
    [[ "${reply:-}" =~ ^[yY] ]]
}

list_model_ids() {
    awk '/^  - id:/{print $3}' "$MODELS_YAML"
}

list_local_tasks() {
    local f
    if [[ -d "$DATA_DIR/retrieval" ]]; then
        for f in "$DATA_DIR/retrieval"/*; do
            [[ -d "$f" && -f "$f/queries.jsonl" ]] || continue
            echo "Retrieval ${f##*/}"
        done
    fi
    if [[ -d "$DATA_DIR/sts" ]]; then
        for f in "$DATA_DIR/sts"/*; do
            [[ -d "$f" && -f "$f/pairs.jsonl" ]] || continue
            echo "STS ${f##*/}"
        done
    fi
}

# One model id per line. "all" = every id in models.yaml.
resolve_models() {
    local choice="$1"
    local -a ids=()
    mapfile -t ids < <(list_model_ids)
    if [[ ${#ids[@]} -eq 0 ]]; then
        echo "No models in $MODELS_YAML." >&2
        return 1
    fi
    if [[ -z "$choice" ]]; then
        echo "$EMBEDDING_MODEL"
        return
    fi
    if [[ "${choice,,}" == "all" ]]; then
        printf '%s\n' "${ids[@]}"
        return
    fi
    local tok
    choice="${choice//,/ }"
    for tok in $choice; do
        if [[ "$tok" =~ ^[0-9]+$ ]]; then
            if (( tok >= 1 && tok <= ${#ids[@]} )); then
                echo "${ids[tok-1]}"
            else
                echo "No model number $tok." >&2
                return 1
            fi
        else
            echo "$tok"
        fi
    done
}

# Lines of "Retrieval name" / "STS name". "all" = every local folder.
# A base name like sop-handbook-v1 also matches sop-handbook-v1-en/zh/ms.
resolve_tasks() {
    local choice="$1"
    local -a rows=()
    mapfile -t rows < <(list_local_tasks)
    if [[ ${#rows[@]} -eq 0 ]]; then
        echo "No local datasets under $DATA_DIR/retrieval or $DATA_DIR/sts." >&2
        return 1
    fi
    if [[ -z "$choice" || "${choice,,}" == "all" ]]; then
        printf '%s\n' "${rows[@]}"
        return
    fi
    local tok row kind name matched
    choice="${choice//,/ }"
    for tok in $choice; do
        matched=0
        if [[ "$tok" =~ ^[0-9]+$ ]]; then
            if (( tok >= 1 && tok <= ${#rows[@]} )); then
                echo "${rows[tok-1]}"
                matched=1
            fi
        else
            for row in "${rows[@]}"; do
                name="${row#* }"
                if [[ "$name" == "$tok" ]]; then
                    echo "$row"
                    matched=1
                fi
            done
            if [[ "$matched" -eq 0 ]]; then
                for row in "${rows[@]}"; do
                    name="${row#* }"
                    if [[ "$name" == "$tok-en" || "$name" == "$tok-zh" || "$name" == "$tok-ms" ]]; then
                        echo "$row"
                        matched=1
                    fi
                done
            fi
        fi
        if [[ "$matched" -eq 0 ]]; then
            echo "No local task '$tok'." >&2
            return 1
        fi
    done
}

pick_models() {
    local ids=()
    local id i=1
    echo >&2
    echo "Embedding models (one GPU: sequential):" >&2
    while read -r id; do
        [[ -z "$id" ]] && continue
        ids+=("$id")
        echo "  $i) $id" >&2
        i=$((i + 1))
    done < <(list_model_ids)
    echo "  a) all" >&2
    echo >&2
    local choice
    read -r -p "Model [blank = $EMBEDDING_MODEL / all / 1,3]: " choice
    if [[ -z "$choice" ]]; then
        resolve_models "$EMBEDDING_MODEL"
    elif [[ "${choice,,}" == "a" ]]; then
        resolve_models all
    else
        resolve_models "$choice"
    fi
}

pick_tasks() {
    local -a rows=()
    local row i=1
    mapfile -t rows < <(list_local_tasks)
    if [[ ${#rows[@]} -eq 0 ]]; then
        echo "No local datasets under $DATA_DIR/retrieval or $DATA_DIR/sts." >&2
        return 1
    fi
    echo >&2
    echo "Local tasks:" >&2
    for row in "${rows[@]}"; do
        echo "  $i) $row" >&2
        i=$((i + 1))
    done
    echo "  a) all" >&2
    echo >&2
    local choice
    read -r -p "Tasks [blank = all / 1,2]: " choice
    if [[ -z "$choice" || "${choice,,}" == "a" ]]; then
        resolve_tasks all
    else
        resolve_tasks "$choice"
    fi
}

pick_language() {
    local default="${1:-en}"
    echo >&2
    echo "Language (chunk.language in Postgres):" >&2
    echo "  1) en" >&2
    echo "  2) zh" >&2
    echo "  3) ms" >&2
    echo "  4) all" >&2
    echo >&2
    local choice
    read -r -p "Language [blank = $default]: " choice
    case "${choice:-}" in
        "") echo "$default" ;;
        1) echo en ;;
        2) echo zh ;;
        3) echo ms ;;
        4|all|ALL) echo all ;;
        *) echo "$choice" ;;
    esac
}

ask_count() {
    local default="${1:-200}"
    local value
    read -r -p "How many [all / $default]: " value
    value="${value:-$default}"
    if [[ "${value,,}" == "all" ]]; then
        echo all
    else
        echo "$value"
    fi
}

# blank/all = every remaining chunk; skip = do not generate this language.
ask_count_for_lang() {
    local lang="$1"
    local value
    read -r -p "$lang: how many [Enter=all / skip / number]: " value
    if [[ -z "$value" || "${value,,}" == "all" ]]; then
        echo all
    elif [[ "${value,,}" == "skip" ]]; then
        echo skip
    else
        echo "$value"
    fi
}

generate_retrieval_lang() {
    local lang="$1"
    local count="$2"
    local conc="$3"
    echo "Generating retrieval ($lang, $count). Detail: $LOG_DIR/generate.log"
    if [[ "${count,,}" == "all" ]]; then
        embbench retrieval generate --all --language "$lang" --concurrency "$conc"
    else
        embbench retrieval generate --count "$count" --language "$lang" --concurrency "$conc"
    fi
}

generate_sts_lang() {
    local lang="$1"
    local count="$2"
    echo "Generating STS ($lang, $count). Detail: $LOG_DIR/generate.log"
    if [[ "${count,,}" == "all" ]]; then
        embbench sts generate --all --language "$lang"
    else
        embbench sts generate --count "$count" --language "$lang"
    fi
}

FROM_MENU=0

maybe_manual() {
    [[ "$FROM_MENU" == 1 ]] && want_manual
}

run_retrieval() {
    local count="$GEN_COUNT" name="$RETRIEVAL_NAME" lang="$GEN_LANGUAGE" conc="$GEN_CONCURRENCY"
    local -a langs=()
    local -a counts=()
    if maybe_manual; then
        name="$(ask "Export name" "$name")"
        lang="$(pick_language "$lang")"
        conc="$(ask "vLLM concurrency" "$conc")"
        if [[ "${lang,,}" == "all" ]]; then
            local l c
            for l in en zh ms; do
                c="$(ask_count_for_lang "$l")"
                [[ "$c" == skip ]] && continue
                langs+=("$l")
                counts+=("$c")
            done
            if [[ ${#langs[@]} -eq 0 ]]; then
                echo "No languages selected."
                return 1
            fi
        else
            count="$(ask_count "$count")"
            langs=("$lang")
            counts=("$count")
        fi
    else
        langs=("$lang")
        counts=("$count")
        if [[ "${lang,,}" == "all" ]]; then
            langs=(en zh ms)
            counts=("$count" "$count" "$count")
        fi
    fi
    ensure_vllm
    local i folder
    for i in "${!langs[@]}"; do
        generate_retrieval_lang "${langs[i]}" "${counts[i]}" "$conc"
        if [[ "${lang,,}" == "all" ]]; then
            folder="${name}-${langs[i]}"
        else
            folder="$name"
        fi
        embbench retrieval export --name "$folder" --language "${langs[i]}"
        echo "Wrote $DATA_DIR/retrieval/$folder"
    done
}

run_sts() {
    local count="$GEN_COUNT" name="$STS_NAME" lang="$GEN_LANGUAGE"
    local -a langs=()
    local -a counts=()
    if maybe_manual; then
        name="$(ask "Export name" "$name")"
        lang="$(pick_language "$lang")"
        if [[ "${lang,,}" == "all" ]]; then
            local l c
            for l in en zh ms; do
                c="$(ask_count_for_lang "$l")"
                [[ "$c" == skip ]] && continue
                langs+=("$l")
                counts+=("$c")
            done
            if [[ ${#langs[@]} -eq 0 ]]; then
                echo "No languages selected."
                return 1
            fi
        else
            count="$(ask_count "$count")"
            langs=("$lang")
            counts=("$count")
        fi
    else
        langs=("$lang")
        counts=("$count")
        if [[ "${lang,,}" == "all" ]]; then
            langs=(en zh ms)
            counts=("$count" "$count" "$count")
        fi
    fi
    ensure_vllm
    local i folder
    for i in "${!langs[@]}"; do
        generate_sts_lang "${langs[i]}" "${counts[i]}"
        if [[ "${lang,,}" == "all" ]]; then
            folder="${name}-${langs[i]}"
        else
            folder="$name"
        fi
        embbench sts export --name "$folder" --language "${langs[i]}"
        echo "Wrote $DATA_DIR/sts/$folder"
    done
}

run_benchmark() {
    local model_arg="${1:-}"
    local -a models=()
    local -a task_rows=()
    local -a names=()
    local -A types=()
    local row kind name model types_csv names_csv
    if [[ -n "$model_arg" ]]; then
        mapfile -t models < <(resolve_models "$model_arg")
        mapfile -t task_rows < <(resolve_tasks "${BENCH_TASKS:-$RETRIEVAL_NAME}")
    elif maybe_manual; then
        mapfile -t models < <(pick_models)
        mapfile -t task_rows < <(pick_tasks)
    else
        mapfile -t models < <(resolve_models "${BENCH_MODELS:-$EMBEDDING_MODEL}")
        mapfile -t task_rows < <(resolve_tasks "${BENCH_TASKS:-$RETRIEVAL_NAME}")
    fi
    if [[ ${#models[@]} -eq 0 ]]; then
        echo "No models selected."
        return 1
    fi
    if [[ ${#task_rows[@]} -eq 0 ]]; then
        echo "No tasks selected."
        return 1
    fi
    for row in "${task_rows[@]}"; do
        kind="${row%% *}"
        name="${row#* }"
        types["$kind"]=1
        names+=("$name")
    done
    types_csv="$(IFS=,; echo "${!types[*]}")"
    names_csv="$(IFS=,; echo "${names[*]}")"
    echo "Benchmark models: ${models[*]}"
    echo "Tasks: ${task_rows[*]}"
    for model in "${models[@]}"; do
        ensure_embed "$model"
        echo "Running embbench model=$model tasks=$names_csv (vLLM $EMBED_HEALTH_URL)"
        embbench run \
            --model "$model" \
            --languages eng,cmn,zsm \
            --task-types "$types_csv" \
            --no-include-mteb \
            --task-names "$names_csv"
    done
}

run_dashboard() {
    local sub
    while true; do
        echo
        if dashboard_alive; then
            echo "Dashboard: running $(dashboard_url)"
        else
            echo "Dashboard: stopped"
        fi
        echo "  1) Start (background)"
        echo "  2) Stop"
        echo "  3) Back"
        echo
        read -r -p "Dashboard: " sub
        case "${sub:-}" in
            1) start_dashboard ;;
            2) stop_dashboard ;;
            3|"") return 0 ;;
            *) echo "Invalid choice." ;;
        esac
    done
}

dashboard_url() {
    local port="${DASHBOARD_PORT:-8501}"
    if [[ -f "$DASH_PORT_FILE" ]]; then
        port="$(cat "$DASH_PORT_FILE")"
    fi
    echo "http://127.0.0.1:${port}"
}

dashboard_alive() {
    local pid
    [[ -f "$DASH_PID_FILE" ]] || return 1
    pid="$(cat "$DASH_PID_FILE")"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

start_dashboard() {
    local port="${DASHBOARD_PORT:-8501}"
    if dashboard_alive; then
        echo "Already running at $(dashboard_url) (pid $(cat "$DASH_PID_FILE"))."
        return 0
    fi
    if [[ "$FROM_MENU" == 1 ]]; then
        port="$(ask "Dashboard port" "$port")"
    fi
    echo "Starting dashboard in background on http://127.0.0.1:$port"
    (
        cd "$REPO"
        exec "$UV" run embbench dashboard --port "$port"
    ) >>"$LOG_DIR/dashboard.log" 2>&1 &
    echo $! >"$DASH_PID_FILE"
    echo "$port" >"$DASH_PORT_FILE"
    disown $! 2>/dev/null || true
    local i
    for i in $(seq 1 20); do
        if healthy "http://127.0.0.1:${port}"; then
            echo "Dashboard ready: http://127.0.0.1:$port  (2 to stop)"
            log "dashboard pid=$(cat "$DASH_PID_FILE") port=$port"
            return 0
        fi
        sleep 1
    done
    echo "Dashboard did not answer yet. Log: $LOG_DIR/dashboard.log"
}

stop_dashboard() {
    local pid
    if dashboard_alive; then
        pid="$(cat "$DASH_PID_FILE")"
        echo "Stopping dashboard (pid $pid)..."
        kill "$pid" 2>/dev/null || true
        pkill -P "$pid" 2>/dev/null || true
        sleep 0.5
        kill -9 "$pid" 2>/dev/null || true
        log "dashboard stopped pid=$pid"
    else
        echo "Dashboard is not running."
    fi
    rm -f "$DASH_PID_FILE"
}

usage() {
    cat <<EOF
Usage: $0 [retrieval|sts|benchmark [model-id|all]|dashboard]

No args: menu. Defaults come from .env. Manual prompts only if you say yes.

  retrieval     start chat vLLM if needed, generate, export ($RETRIEVAL_NAME)
  sts           start chat vLLM if needed, generate, export ($STS_NAME)
  benchmark     stop chat vLLM, start pooling vLLM, score local folders
  dashboard     start Streamlit in the background (:${DASHBOARD_PORT:-8501}, no GPU)
  dashboard stop

Chat is :${LLM_PORT:-8000}. Embeddings are :${EMBED_PORT:-8001} (same GPU, never both).
One GPU: models run one after another. Several local tasks share one model load.

.env: BENCH_MODELS (id, comma list, or all; default EMBEDDING_MODEL)
      BENCH_TASKS  (folder names, comma list, or all; default RETRIEVAL_NAME)
      DASHBOARD_PORT (default 8501)

Example:

  $0 retrieval
  $0 benchmark all
  $0 dashboard
EOF
}

menu() {
    FROM_MENU=1
    while true; do
        echo
        echo "-------------------------------------"
        echo "  generation / embbench"
        echo "-------------------------------------"
        echo "1) Generate retrieval"
        echo "2) Generate STS"
        echo "3) Benchmark"
        echo "4) Dashboard"
        echo "5) Quit"
        echo
        local choice
        read -r -p "Choice: " choice
        case "$choice" in
            1) run_retrieval ;;
            2) run_sts ;;
            3) run_benchmark ;;
            4) run_dashboard ;;
            5) exit 0 ;;
            *) echo "Invalid choice." ;;
        esac
    done
}

load_env

case "${1:-}" in
    "") menu ;;
    -h|--help|help) usage ;;
    retrieval) run_retrieval ;;
    sts) run_sts ;;
    benchmark) run_benchmark "${2:-}" ;;
    dashboard)
        if [[ "${2:-}" == stop ]]; then
            stop_dashboard
        else
            start_dashboard
        fi
        ;;
    *) usage; exit 1 ;;
esac
