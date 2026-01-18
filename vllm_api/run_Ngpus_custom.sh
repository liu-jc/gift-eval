#!/bin/bash
# vLLM multi-GPU launcher with port-specific cleanup

set -euo pipefail

if [ $# -lt 4 ]; then
    echo "Usage: $0 <model_path> <gpu_list> <start_port> <mode>"
    echo ""
    echo "mode:"
    echo "  llm   -> legacy server, endpoint: POST /generate"
    echo "  vlm   -> OpenAI-compatible server, endpoint: POST /v1/chat/completions"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/llm 0 8000 llm"
    echo "  $0 /path/to/llm 4 8000 llm"
    echo "  $0 /path/to/vlm \"0,2,5\" 9000 vlm"
    exit 1
fi

MODEL_PATH=$1
GPU_LIST=$2
START_PORT=$3
MODE=$4

if [[ "$MODE" != "llm" && "$MODE" != "vlm" ]]; then
    echo "❌ ERROR: mode must be 'llm' or 'vlm', got: $MODE"
    exit 1
fi

echo "===== Model Path: $MODEL_PATH ====="
echo "===== GPU List: $GPU_LIST ====="
echo "===== Start Port: $START_PORT ====="
echo "===== Mode: $MODE ====="

# Parse GPU list
if [[ $GPU_LIST =~ ^[0-9]+$ ]]; then
    NGPU=$GPU_LIST
    GPU_ARRAY=()
    for i in $(seq 0 $((NGPU - 1))); do
        GPU_ARRAY+=($i)
    done
    echo "===== Using GPUs: ${GPU_ARRAY[@]} ====="
else
    IFS=',' read -ra GPU_ARRAY <<< "$GPU_LIST"
    NGPU=${#GPU_ARRAY[@]}
    echo "===== Using GPUs: ${GPU_ARRAY[@]} (total: $NGPU) ====="
fi

# Dependency check
if ! command -v lsof >/dev/null 2>&1; then
    echo "❌ ERROR: lsof not found. Please install lsof."
    exit 1
fi

HAVE_CURL=0
if command -v curl >/dev/null 2>&1; then
    HAVE_CURL=1
fi

# Smart cleanup: only clean processes on target ports
echo ""
echo "===== Checking for Port Conflicts ====="
PORTS_TO_CLEAN=()
for i in $(seq 0 $((NGPU - 1))); do
    PORT=$((START_PORT + i))
    
    # Check if port is occupied
    if lsof -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
        # Check if it's a vLLM/Python process
        if lsof -iTCP:$PORT -sTCP:LISTEN | grep -q "python"; then
            echo "⚠️  Port $PORT is occupied (will be cleaned)"
            PORTS_TO_CLEAN+=($PORT)
        else
            echo "❌ ERROR: Port $PORT is occupied by non-Python process!"
            lsof -iTCP:$PORT -sTCP:LISTEN
            exit 1
        fi
    fi
done

if [ ${#PORTS_TO_CLEAN[@]} -gt 0 ]; then
    echo ""
    echo "Cleaning up processes on ports: ${PORTS_TO_CLEAN[*]}"
    
    for PORT in "${PORTS_TO_CLEAN[@]}"; do
        # Get PIDs occupying this port
        PIDS=$(lsof -iTCP:$PORT -sTCP:LISTEN -t 2>/dev/null || true)
        if [ -n "$PIDS" ]; then
            echo "  Killing process(es) on port $PORT: $PIDS"
            kill $PIDS 2>/dev/null || true
        fi
    done
    
    sleep 3
    
    # Verify cleanup
    FAILED=()
    for PORT in "${PORTS_TO_CLEAN[@]}"; do
        if lsof -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
            FAILED+=($PORT)
        fi
    done
    
    if [ ${#FAILED[@]} -gt 0 ]; then
        echo "⚠️  Some ports still occupied: ${FAILED[*]}, force killing..."
        for PORT in "${FAILED[@]}"; do
            PIDS=$(lsof -iTCP:$PORT -sTCP:LISTEN -t 2>/dev/null || true)
            if [ -n "$PIDS" ]; then
                kill -9 $PIDS 2>/dev/null || true
            fi
        done
        sleep 2
    fi
    
    echo "✅ Target ports cleared"
else
    echo "✅ No port conflicts detected"
fi

# Readiness check endpoint by mode
if [ "$MODE" = "vlm" ]; then
    READY_PATH="/v1/models"
else
    READY_PATH="/health"
fi

check_pid_alive() { kill -0 "$1" >/dev/null 2>&1; }
check_listen() { lsof -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

check_ready() {
    local port="$1"

    # Fallback to port listening if curl unavailable
    if [ $HAVE_CURL -eq 0 ]; then
        check_listen "$port"
        return $?
    fi

    if curl -sS --max-time 2 "http://127.0.0.1:${port}${READY_PATH}" >/dev/null 2>&1; then
        return 0
    fi

    # Port listening but HTTP not ready: return 2 (semi-ready)
    if check_listen "$port"; then
        return 2
    fi
    return 1
}

# Launch servers
echo ""
echo "===== Starting vLLM Servers ====="

PIDS=()
for i in "${!GPU_ARRAY[@]}"; do
    GPU_ID=${GPU_ARRAY[$i]}
    PORT=$((START_PORT + i))

    echo "Starting vLLM server (mode=$MODE) on GPU $GPU_ID at port $PORT..."

    if [ "$MODE" = "vlm" ]; then
        # OpenAI-compatible server (supports multimodal /v1/chat/completions)
        CUDA_VISIBLE_DEVICES=$GPU_ID python -m vllm.entrypoints.openai.api_server \
            --model "$MODEL_PATH" \
            --host 127.0.0.1 \
            --port "$PORT" \
            --gpu-memory-utilization 0.2 \
            --max-model-len 16000 \
            --tensor-parallel-size 1 \
            > "vllm_log_gpu_${GPU_ID}_port_${PORT}.txt" 2>&1 &
    else
        # Legacy server (text-only /generate)
        CUDA_VISIBLE_DEVICES=$GPU_ID python -m vllm.entrypoints.api_server \
            --model "$MODEL_PATH" \
            --host 127.0.0.1 \
            --port "$PORT" \
            --gpu-memory-utilization 0.2 \
            --max-model-len 16000 \
            --max-num-seqs 200 \
            --tensor-parallel-size 1 \
            > "vllm_log_gpu_${GPU_ID}_port_${PORT}.txt" 2>&1 &
    fi

    PIDS+=($!)
    sleep 2
done

# Wait for servers to become ready
echo ""
echo "===== Waiting for Servers to Become Ready (max 5 min) ====="

TIMEOUT_SEC=300
INTERVAL_SEC=5
DEADLINE=$((SECONDS + TIMEOUT_SEC))

while true; do
    # Any process exits early -> fail immediately
    for idx in "${!PIDS[@]}"; do
        pid="${PIDS[$idx]}"
        if ! check_pid_alive "$pid"; then
            GPU_ID="${GPU_ARRAY[$idx]}"
            PORT=$((START_PORT + idx))
            echo "❌ ERROR: vLLM process exited early (GPU $GPU_ID, port $PORT, pid $pid)"
            echo "   Check log: vllm_log_gpu_${GPU_ID}_port_${PORT}.txt"
            exit 1
        fi
    done

    READY=0
    SEMI=0
    PENDING=()

    for i in "${!GPU_ARRAY[@]}"; do
        GPU_ID=${GPU_ARRAY[$i]}
        PORT=$((START_PORT + i))

        if check_ready "$PORT"; then
            READY=$((READY + 1))
        else
            rc=$?
            if [ "$rc" -eq 2 ]; then
                SEMI=$((SEMI + 1))
                PENDING+=("GPU $GPU_ID:Port $PORT (LISTEN ok, HTTP not ready)")
            else
                PENDING+=("GPU $GPU_ID:Port $PORT")
            fi
        fi
    done

    if [ "$READY" -eq "$NGPU" ]; then
        echo "✅ All $NGPU servers are READY!"
        break
    fi

    if [ "$SECONDS" -ge "$DEADLINE" ]; then
        echo "❌ TIMEOUT: Not all servers ready within ${TIMEOUT_SEC}s (ready: $READY/$NGPU, semi: $SEMI)"
        echo "Pending: ${PENDING[*]}"
        echo "Check logs:"
        for i in "${!GPU_ARRAY[@]}"; do
            GPU_ID=${GPU_ARRAY[$i]}
            PORT=$((START_PORT + i))
            echo "  - GPU $GPU_ID port $PORT -> vllm_log_gpu_${GPU_ID}_port_${PORT}.txt"
        done
        exit 1
    fi

    echo "⏳ Ready: $READY/$NGPU (semi: $SEMI). Recheck in ${INTERVAL_SEC}s"
    if [ ${#PENDING[@]} -gt 0 ]; then
        echo "   Pending: ${PENDING[*]}"
    fi
    sleep "$INTERVAL_SEC"
done

# Output summary
echo ""
echo "===== Configuration Summary ====="
echo "Mode: $MODE"
echo "GPUs used: ${GPU_ARRAY[@]}"
echo "Ports: $START_PORT to $((START_PORT + NGPU - 1))"

if [ "$MODE" = "vlm" ]; then
    echo "API: POST /v1/chat/completions"
    echo "Ready check: GET ${READY_PATH}"
else
    echo "API: POST /generate"
    echo "Ready check: GET ${READY_PATH}"
fi

echo ""
echo "✅ Done."
