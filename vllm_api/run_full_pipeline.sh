#!/bin/bash
# Main pipeline script: configure model and run inference

set -e

# ============================================================================
# Configuration - modify these parameters
# ============================================================================
MODEL_ID="qwen3-vl-8b-thinking-with-data"
                                       # Available models: 
                                       #   - qwen2.5-7b (text)
                                       #   - qwen2.5-vl-3b (VLM, image only)
                                       #   - qwen2.5-vl-3b-with-data (VLM, image + data)
                                       #   - qwen2.5-vl-7b (VLM, image only)
                                       #   - qwen2.5-vl-7b-with-data (VLM, image + data)
                                       #   - qwen3-235b (text)
                                       #   - qwen3-235b-thinking (text + reasoning)
                                       #   - qwen3-vl-235b (VLM, image only)
                                       #   - qwen3-vl-235b-thinking (VLM, image + reasoning)
                                       #   - qwen3-vl-235b-with-data (VLM, image + data)
                                       #   - qwen3-vl-235b-thinking-with-data (VLM, image + data + reasoning)
                                       #   - qwen3-vl-8b-thinking (VLM, image + reasoning)
                                       #   - qwen3-vl-8b-thinking-with-data (VLM, image + data + reasoning)

GPU_LIST="0,3"                         # GPU list (comma-separated)
START_PORT=9000                        # Starting port number

ENABLE_REASONING="true"                # Enable reasoning mode (true/false)
OUTPUT_MODEL_NAME="qwen3-vl-8b-thinking-image+data-vllm"  # Output folder name
# ============================================================================
# Environment configuration (usually no need to modify)
# ============================================================================
CONDA_BASE="/home/zhiyuan/yc/miniconda3"
VLLM_ENV="ts"                          # vLLM server environment
INFER_ENV="ts_infer"                   # Inference script environment

# ============================================================================
# Auto-derived configuration (no manual modification needed)
# ============================================================================
# Calculate number of GPUs
IFS=',' read -ra GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}

# Get model path and mode via Python
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_BASE/envs/$INFER_ENV"

echo "🔍 Parsing model configuration..."
PYTHON_OUTPUT=$(python -c "
import sys
sys.path.insert(0, '.')
from model_registry import get_model_path, get_model_mode

model_id = '$MODEL_ID'
try:
    model_path = get_model_path(model_id)
    mode = get_model_mode(model_id)
    print(f'{model_path}|{mode}')
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
")

if [ $? -ne 0 ]; then
    echo "❌ Failed to parse model configuration!"
    echo "$PYTHON_OUTPUT"
    exit 1
fi

MODEL_PATH=$(echo "$PYTHON_OUTPUT" | cut -d'|' -f1)
MODE=$(echo "$PYTHON_OUTPUT" | cut -d'|' -f2)

# ============================================================================
# Display configuration
# ============================================================================
echo ""
echo "================================"
echo "Launch Configuration"
echo "================================"
echo "Model ID:        $MODEL_ID"
echo "Model Path:      $MODEL_PATH"
echo "Mode:            $MODE"
echo "GPU List:        $GPU_LIST ($NUM_GPUS GPUs)"
echo "Port Range:      $START_PORT - $((START_PORT + NUM_GPUS - 1))"
echo "Reasoning:       $([ "$ENABLE_REASONING" = "true" ] && echo "Enabled" || echo "Disabled")"
echo "Output Name:     $OUTPUT_MODEL_NAME"
echo "================================"
echo ""

# ============================================================================
# Launch vLLM servers
# ============================================================================
echo "Launching vLLM servers..."
conda activate "$CONDA_BASE/envs/$VLLM_ENV"

bash run_Ngpus_custom.sh "$MODEL_PATH" "$GPU_LIST" "$START_PORT" "$MODE"


# ============================================================================
# Run inference script
# ============================================================================
echo ""
echo "🔮 Running inference script..."
conda activate "$CONDA_BASE/envs/$INFER_ENV"

# Build inference command
INFER_CMD="python llm_predictor_qwen_vllm_infer_only.py $NUM_GPUS --start-port $START_PORT --model-id $MODEL_ID --output-model-name $OUTPUT_MODEL_NAME"

if [ "$ENABLE_REASONING" = "true" ]; then
    INFER_CMD="$INFER_CMD --enable-reasoning"
fi

echo "Executing: $INFER_CMD"
echo ""

eval $INFER_CMD

echo ""
echo "Inference completed"
