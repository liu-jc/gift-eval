# vLLM-based Time Series Forecasting with LLM/VLM

A streamlined pipeline for time series forecasting using Large Language Models (LLM) and Vision-Language Models (VLM) via vLLM inference servers.

## Project Structure

```
vllm_api/
├── run_full_pipeline.sh              # Main entry script - configure and run
├── run_Ngpus_custom.sh               # vLLM server launcher (multi-GPU support)
├── llm_predictor_vllm_infer_only.py  # Inference script
├── model_registry.py                 # Model configurations and prompts
├── utils.py                          # Utility functions
├── cleanup_vllm.sh                   # Manual cleanup script for vLLM servers
└── logs/                             # Execution logs

../results/{model_name}/              # Output predictions (JSON, created automatically)
```

## Quick Start

### 1. Configure the Pipeline

Edit `run_full_pipeline.sh` to set your model and GPU configuration:

```bash
MODEL_ID="qwen3-vl-8b-thinking-with-data"     # Model to use
GPU_LIST="0,3"                                # GPUs (comma-separated)
START_PORT=9000                               # Starting port
ENABLE_REASONING="true"                       # Enable reasoning mode
OUTPUT_MODEL_NAME="qwen3-vl-8b-thinking-image+data-vllm"  # Output folder name
```

### 2. Run the Pipeline

```bash
bash run_full_pipeline.sh
```

This will:
1. Launch vLLM servers on specified GPUs
2. Run inference on all datasets
3. Save predictions to `../results/{OUTPUT_MODEL_NAME}/`

### 3. Results

Predictions are saved as JSON files in the parent directory:
```
../results/
└── qwen3-vl-8b-thinking-image+data-vllm/
    ├── intermediate_electricity_hourly_H_short.json
    ├── intermediate_traffic_H_short.json
    └── ...
```

## Available Models

### Text-Only Models
- `qwen2.5-7b` - Qwen2.5 7B Instruct
- `qwen3-235b` - Qwen3 235B Instruct
- `qwen3-235b-thinking` - Qwen3 235B Thinking

### Vision-Language Models (VLM)

**Qwen2.5 Series:**
- `qwen2.5-vl-3b` - 3B, image only
- `qwen2.5-vl-3b-with-data` - 3B, image + numerical data
- `qwen2.5-vl-7b` - 7B, image only
- `qwen2.5-vl-7b-with-data` - 7B, image + numerical data

**Qwen3 Series:**
- `qwen3-vl-235b` - 235B, image only
- `qwen3-vl-235b-thinking` - 235B, image + reasoning
- `qwen3-vl-235b-with-data` - 235B, image + data
- `qwen3-vl-235b-thinking-with-data` - 235B, image + data + reasoning
- `qwen3-vl-8b-thinking` - 8B, image + reasoning ✨
- `qwen3-vl-8b-thinking-with-data` - 8B, image + data + reasoning ✨

## Advanced Usage

### Manual vLLM Server Launch

```bash
bash run_Ngpus_custom.sh MODEL_PATH GPU_LIST START_PORT MODE
```

Example:
```bash
bash run_Ngpus_custom.sh "Qwen/Qwen3-VL-8B-Thinking" "0,3" 9000 vlm
```

### Python Script Direct Call

```bash
python llm_predictor_vllm_infer_only.py NUM_GPUS \
    --start-port 9000 \
    --model-id qwen3-vl-8b-thinking-with-data \
    --enable-reasoning true \
    --output-model-name my-experiment
```

### Adding New Models

Edit `model_registry.py`:

```python
class MyModelConfig(BaseModelConfig):
    model_name = "MyOrg/MyModel"
    model_type = "vlm"  # or "text"
    
    def create_prompt(self, ...):
        return "Your prompt template"

ModelRegistry.register("my-model", MyModelConfig())
```

## Cleanup

**Automatic Cleanup** (Smart):
- The script automatically cleans up only the vLLM servers it started
- Other vLLM servers on different ports are preserved
- Supports running multiple experiments in parallel

**Manual Cleanup** (All servers):
```bash
bash cleanup_vllm.sh  # Kills ALL vLLM servers
```

## Key Features

- **Multi-GPU Support**: Automatically distributes vLLM servers across GPUs
- **Resume Capability**: Automatically resumes from cached predictions
- **Smart Port Management**: Only cleans up conflicting ports on startup
- **Smart Cleanup**: Only terminates vLLM servers started by current script
- **Parallel Experiments**: Run multiple experiments on different ports simultaneously
- **VLM Support**: Plot time series as images for vision-language models
- **Flexible Model Registry**: Easy to add new models without changing main code

## Monitoring

Check logs during execution:
```bash
tail -f logs/qwen3-vl-8b-thinking-image+data-vllm.log
```

Check vLLM server status:
```bash
ps aux | grep vllm.entrypoints | grep -v grep
lsof -i :9000,9001  # Check specific ports
```

## Environment

- Python environment for vLLM servers: `ts`
- Python environment for inference: `ts_infer`
- vLLM backend: Supports both `api_server` and `openai.api_server`

## Output Format

Predictions are saved as `intermediate_*.json` files with metadata:

```json
{
  "item_id": "item_001",
  "series_idx": 0,
  "dataset": "electricity_hourly",
  "prediction": [1.23, 4.56, ...],
  "timestamp": "2026-01-18T12:00:00"
}
```

Files are saved to: `../results/{OUTPUT_MODEL_NAME}/intermediate_{dataset}_{freq}_{term}.json`



