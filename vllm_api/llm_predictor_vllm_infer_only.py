"""
LLM-based Time Series Predictor - Inference Only (No Metrics)

Pure inference, save predictions, no evaluation metrics
"""

import os
import sys
import json
import logging
import itertools
import requests
import signal
import subprocess
import argparse
import base64
from io import BytesIO
from typing import Iterator, List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Headless mode
import matplotlib.pyplot as plt

from gluonts.model.forecast import SampleForecast
from gift_eval.data import Dataset

# Import model registry
from model_registry import ModelRegistry, get_model_config, list_available_models

# Import utility functions
from utils import (
    create_time_series_plot,
    parse_llm_response,
    cleanup_vllm_servers,
    setup_signal_handlers,
    PortRotator
)

# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LLM Time Series Predictor - Inference Only")
    parser.add_argument("num_gpus", type=int, nargs="?", default=3, help="Number of GPUs")
    parser.add_argument("--start-port", type=int, default=8000, help="Starting port")
    parser.add_argument("--model-id", type=str, default=None, help="Model registry ID (overrides CONFIG)")
    parser.add_argument("--output-model-name", type=str, default=None, help="Output model name for results directory (overrides CONFIG)")
    parser.add_argument("--enable-reasoning", action="store_true", help="Enable reasoning in prompts (overrides CONFIG)")
    return parser.parse_args()

args = parse_args()

# Command-line arguments override CONFIG
if args.model_id:
    print(f"🔧 Using command-line model ID: {args.model_id}")
if args.output_model_name:
    print(f"🔧 Using command-line output name: {args.output_model_name}")
if args.enable_reasoning:
    print(f"🔧 Reasoning mode enabled")

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Data Configuration
    "data_path": "/home/zhiyuan/gift-eval-data",
    
    # vLLM Configuration
    "api_provider": "vllm",
    "num_gpus": args.num_gpus,
    "vllm_ports": list(range(args.start_port, args.start_port + args.num_gpus)),
    "vllm_host": "127.0.0.1",
    
    # Model selection (using Model registry ID)
    "model_id": "qwen2.5-vl-3b",  # Options: "qwen2.5-7b", "qwen2.5-vl-7b", "gpt-4-vision", etc.
    # "model_id": "qwen2.5-7b",  # Options: "qwen2.5-7b", "qwen2.5-vl-7b", "gpt-4-vision", etc.
    # Model Configuration
    "num_samples": 5,
    "temperature": 0.7,
    "max_tokens": 5000,
    "max_context_length": 4000,
    "enable_reasoning": False,
    "top_p": 1.0,
    "top_k": -1,
    
    # Multimodal config (auto-used by VLM models)
    "save_plots": True,  # Save time series plots (in VLM mode)
    
    # Inference Configuration  
    "output_model_name": "qwen2.5-3B-instruct-vlm-vllm",
    "save_predictions": True,  # Save predictions
    
    # Parallel Processing
    "enable_parallel": True,
    "max_workers": 16,
    
    # Cleanup Configuration
    "auto_cleanup_vllm": True,
    "cleanup_on_start": False,
    
    # Datasets to infer
    "test_dataset_configs": [
        "kdd_cup_2018_with_missing/D/short",
        "restaurant",
        "hierarchical_sales/W/short",
        "solar/W/short",
        "SZ_TAXI/H",
        "M_DENSE/D",
        "m4_hourly",
        "m4_weekly",
        "covid_deaths",
        "hospital",
    ],
}

# Command-line arguments override CONFIG
if args.model_id:
    CONFIG["model_id"] = args.model_id
if args.output_model_name:
    CONFIG["output_model_name"] = args.output_model_name
if args.enable_reasoning:
    CONFIG["enable_reasoning"] = True


# ============================================================================
# LLM PREDICTOR CLASS
# ============================================================================

class LLMTimeSeriesPredictor:
    """LLM time series predictor (multi-model support)"""
    
    def __init__(
        self,
        model_id: str,  # Use Model registry ID
        prediction_length: int,
        num_samples: int = 20,
        temperature: float = 0.7,
        max_context_length: int = 4000,
        enable_reasoning: bool = False,
        domain: str = None,
        freq: str = None,
        save_intermediate: bool = False,
        vllm_ports: List[int] = None,
        vllm_host: str = "127.0.0.1",
        max_tokens: int = 5000,
        top_p: float = 1.0,
        top_k: int = -1,
        save_plots: bool = True,
    ):
        # Load model config from registry
        self.model_config = get_model_config(model_id)
        self.model_id = model_id
        self.model_name = self.model_config.model_name
        self.enable_vlm = (self.model_config.model_type == "vlm")
        
        # Inference parameters
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.temperature = temperature
        self.max_context_length = max_context_length
        self.enable_reasoning = enable_reasoning
        self.domain = domain
        self.freq = freq
        self.save_intermediate = save_intermediate
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.save_plots = save_plots
        
        # vLLM config
        self.vllm_ports = vllm_ports or [8000]
        self.vllm_host = vllm_host
        
        # Storage and locks
        self.intermediate_results = []
        self._file_lock = threading.Lock()
        
        # Logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        self.logger.info(f"Loaded model: {self.model_name} (ID: {self.model_id}, Type: {self.model_config.model_type})")
        
        # Initialize API client
        self._init_api_client()
    
    def _init_api_client(self):
        """Initialize vLLM API client"""
        import socket
        self.client = requests
        
        if not self.vllm_ports:
            self.logger.info("🔍 Auto-detecting vLLM servers...")
            self.vllm_ports = list(range(8000, 8011))
        
        available_ports = []
        for port in self.vllm_ports:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex((self.vllm_host, port))
                sock.close()
                if result == 0:
                    available_ports.append(port)
                    self.logger.info(f"vLLM server at port {port}")
            except Exception:
                pass
        
        if not available_ports:
            raise RuntimeError(f"No vLLM servers reachable! Checked: {self.vllm_ports}")
        
        self.vllm_ports = available_ports
        self.port_rotator = PortRotator(available_ports)  # Use PortRotator from utils
        self.logger.info(f"Connected to {len(self.vllm_ports)} vLLM server(s): {self.vllm_ports}")
    
    def _get_next_vllm_port(self) -> int:
        """Round-robin port selection (using utils.PortRotator)"""
        return self.port_rotator.get_next_port()
    
    def _predict_single_series(self, entry, idx):
        """Predict single sequence"""
        try:
            historical_data = entry["target"]
            start_date = entry["start"]
            item_id = entry.get("item_id", f"series_{idx}")
            
            # check cache
            if hasattr(self, 'cached_predictions') and idx in self.cached_predictions:
                cached_pred = self.cached_predictions[idx]
                if cached_pred.ndim == 1:
                    samples_array = np.tile(cached_pred, (self.num_samples, 1))
                else:
                    samples_array = np.tile(cached_pred[np.newaxis, :, :], (self.num_samples, 1, 1))
                
                if historical_data.ndim > 1:
                    length = historical_data.shape[1] if historical_data.shape[0] < historical_data.shape[1] else historical_data.shape[0]
                else:
                    length = len(historical_data)
                forecast_start_date = start_date + length
                
                return SampleForecast(samples=samples_array, start_date=forecast_start_date)
            
            # Generate multiple samples
            samples = []
            for sample_idx in range(self.num_samples):
                prediction = self._predict_single(historical_data, sample_idx, item_id, idx)
                samples.append(prediction)
            
            samples_array = np.array(samples)
            
            if historical_data.ndim > 1:
                length = historical_data.shape[1] if historical_data.shape[0] < historical_data.shape[1] else historical_data.shape[0]
            else:
                length = len(historical_data)
            forecast_start_date = start_date + length
            
            return SampleForecast(samples=samples_array, start_date=forecast_start_date)
            
        except Exception as e:
            self.logger.error(f"Error predicting series {idx}: {str(e)}")
            historical_data = entry["target"]
            if historical_data.ndim > 1:
                num_variates = historical_data.shape[0] if historical_data.shape[0] < historical_data.shape[1] else historical_data.shape[1]
                fallback_shape = (self.num_samples, self.prediction_length, num_variates)
                length = historical_data.shape[1] if historical_data.shape[0] < historical_data.shape[1] else historical_data.shape[0]
            else:
                fallback_shape = (self.num_samples, self.prediction_length)
                length = len(historical_data)
            
            forecast_start_date = entry["start"] + length
            return SampleForecast(samples=np.zeros(fallback_shape), start_date=forecast_start_date)
    
    def _load_cached_predictions(self):
        """Load cached predictions"""
        cached_predictions = {}
        if hasattr(self, 'intermediate_file') and self.intermediate_file and self.intermediate_file.exists():
            try:
                with open(self.intermediate_file, 'r') as f:
                    data = json.load(f)
                    predictions = data.get('predictions', [])
                    for pred in predictions:
                        series_idx = pred.get('series_idx')
                        prediction_list = pred.get('prediction')
                        if series_idx is not None and prediction_list is not None:
                            pred_array = np.array(prediction_list)
                            if not np.all(pred_array == 0):
                                cached_predictions[series_idx] = pred_array
            except Exception as e:
                self.logger.warning(f"Could not load cache: {e}")
        return cached_predictions
    
    def predict(self, test_data_input, enable_parallel=True, max_workers=5, global_start_idx=0, window_stride=1, **kwargs) -> Iterator[SampleForecast]:
        """generate predictions"""
        test_data_list = list(test_data_input)
        total_series = len(test_data_list)
        
        self.cached_predictions = self._load_cached_predictions()
        num_cached = len(self.cached_predictions)
        
        if num_cached > 0:
            print(f"  ⚡ Cached: {num_cached}/{total_series}")
        
        if enable_parallel and max_workers > 1:
            self.logger.info(f"🚀 Parallel inference: {total_series} series, {max_workers} workers")
            results = [None] * total_series
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(self._predict_single_series, entry, idx * window_stride + global_start_idx): idx
                    for idx, entry in enumerate(test_data_list)
                }
                
                with tqdm(total=total_series, desc="Predicting") as pbar:
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        results[idx] = future.result()
                        pbar.update(1)
            
            for result in results:
                yield result
        else:
            self.logger.info(f"Sequential inference: {total_series} series")
            for idx, entry in enumerate(tqdm(test_data_list, desc="Predicting")):
                yield self._predict_single_series(entry, idx * window_stride + global_start_idx)
    
    def _create_time_series_plot(self, historical_data: np.ndarray, series_idx: int = None) -> str:
        """
        Visualize time series as image, return base64-encoded string (using utils function)
        """
        # Optional: save image path
        save_path = None
        if self.save_plots and hasattr(self, 'plot_dir') and self.plot_dir and series_idx is not None:
            save_path = str(self.plot_dir / f"series_{series_idx}.png")
        
        # Use function from utils
        return create_time_series_plot(
            historical_data=historical_data,
            domain=self.domain,
            prediction_length=self.prediction_length,
            save_path=save_path
        )
    
    def _create_time_series_plot_old(self, historical_data: np.ndarray, series_idx: int = None) -> str:
        """
        Visualize time series as image, return base64-encoded string (deprecated)
        
        Args:
            historical_data: Historical data
            series_idx: Series index (for filename)
            
        Returns:
            base64-encoded PNG image string
        """
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Check if univariate or multivariate
            if historical_data.ndim == 1:
                # Univariate time series
                ax.plot(historical_data, linewidth=2, color='#2E86AB', label='Historical Data')
                ax.set_ylabel('Value', fontsize=12)
                ax.set_title(f'Time Series - {self.domain or "Unknown Domain"}', fontsize=14, fontweight='bold')
            else:
                # Multivariate time series
                if historical_data.shape[0] > historical_data.shape[1]:
                    data = historical_data.T  # Convert to (num_variates, time_steps)
                else:
                    data = historical_data
                
                num_variates = data.shape[0]
                colors = plt.cm.tab10(np.linspace(0, 1, min(num_variates, 10)))
                
                for i in range(num_variates):
                    ax.plot(data[i], linewidth=1.5, color=colors[i % 10], 
                           label=f'Variable {i+1}', alpha=0.8)
                
                ax.set_ylabel('Value', fontsize=12)
                ax.set_title(f'Multivariate Time Series ({num_variates} variables) - {self.domain or "Unknown"}', 
                           fontsize=14, fontweight='bold')
                
                if num_variates <= 10:
                    ax.legend(loc='best', fontsize=9)
            
            # General settings
            ax.set_xlabel('Time Step', fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add prediction area hint
            ax.axvline(x=len(historical_data[0] if historical_data.ndim > 1 else historical_data) - 1, 
                      color='red', linestyle='--', linewidth=2, alpha=0.5, 
                      label=f'Forecast {self.prediction_length} steps ahead')
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close(fig)
            
            # Optional: save image to file
            if self.save_plots and hasattr(self, 'plot_dir') and self.plot_dir:
                plot_file = self.plot_dir / f"series_{series_idx}.png"
                fig_save, ax_save = plt.subplots(figsize=(12, 6))
                
                if historical_data.ndim == 1:
                    ax_save.plot(historical_data, linewidth=2, color='#2E86AB')
                else:
                    data = historical_data.T if historical_data.shape[0] > historical_data.shape[1] else historical_data
                    for i in range(data.shape[0]):
                        ax_save.plot(data[i], linewidth=1.5, alpha=0.8)
                
                ax_save.set_xlabel('Time Step')
                ax_save.set_ylabel('Value')
                ax_save.set_title(f'Series {series_idx}')
                ax_save.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(plot_file, dpi=100, bbox_inches='tight')
                plt.close(fig_save)
            
            return image_base64
            
        except Exception as e:
            self.logger.error(f"Error creating plot: {e}")
            return None
    
    def _predict_single(self, historical_data: np.ndarray, sample_idx: int = 0, 
                        item_id: str = None, series_idx: int = None) -> np.ndarray:
        """Single prediction"""
        if historical_data.ndim > 1:
            num_variates = historical_data.shape[0] if historical_data.shape[0] < historical_data.shape[1] else historical_data.shape[1]
            if historical_data.shape[0] < historical_data.shape[1]:
                context_data = historical_data[:, -self.max_context_length:] if historical_data.shape[1] > self.max_context_length else historical_data
            else:
                context_data = historical_data[-self.max_context_length:, :] if historical_data.shape[0] > self.max_context_length else historical_data
        else:
            num_variates = 1
            context_data = historical_data[-self.max_context_length:] if len(historical_data) > self.max_context_length else historical_data
        
        # Create prompt using Model config
        prompt = self.model_config.create_prompt(
            historical_data=context_data,
            prediction_length=self.prediction_length,
            num_variates=num_variates,
            enable_reasoning=self.enable_reasoning,
            domain=self.domain,
            freq=self.freq,
        )
        
        # 🔥 Multimodal inference path
        if self.enable_vlm:
            image_base64 = self._create_time_series_plot(context_data, series_idx)
            response_text = self._call_api(prompt, image_base64=image_base64)
        else:
            response_text = self._call_api(prompt)
        
        prediction = self._parse_response(response_text, num_variates)
        
        if self.save_intermediate and sample_idx == 0:
            result_entry = {
                "item_id": item_id,
                "series_idx": series_idx,
                "sample_idx": sample_idx,
                "num_variates": num_variates,
                "context_length": context_data.shape[-1] if context_data.ndim > 1 else len(context_data),
                "prompt": prompt,
                "response": response_text,
                "prediction": prediction.tolist(),
                "mode": "vlm" if self.enable_vlm else "text",
            }
            
            # VLM mode: optionally save image path (skip base64 to save space)
            if self.enable_vlm and hasattr(self, 'plot_dir') and self.plot_dir:
                result_entry["plot_file"] = f"series_{series_idx}.png"
            
            self.intermediate_results.append(result_entry)
            
            if hasattr(self, 'intermediate_file') and self.intermediate_file:
                self._save_intermediate_incremental(result_entry)
        
        return prediction
    
    def _save_intermediate_incremental(self, result_entry):
        """Save prediction results in real-time"""
        try:
            with self._file_lock:
                if self.intermediate_file.exists():
                    with open(self.intermediate_file, 'r') as f:
                        data = json.load(f)
                else:
                    data = {
                        "dataset_config": getattr(self, 'dataset_config', 'unknown'),
                        "model": getattr(self, 'model_name_for_save', 'unknown'),
                        "enable_reasoning": self.enable_reasoning,
                        "num_samples": self.num_samples,
                        "predictions": []
                    }
                
                series_idx = result_entry.get('series_idx')
                existing_idx = None
                for i, pred in enumerate(data["predictions"]):
                    if pred.get('series_idx') == series_idx:
                        existing_idx = i
                        break
                
                if existing_idx is not None:
                    data["predictions"][existing_idx] = result_entry
                else:
                    data["predictions"].append(result_entry)
                
                with open(self.intermediate_file, 'w') as f:
                    json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save: {e}")
    
    # Prompt creation moved to Model config class in model_registry.py
    
    def _call_api(self, prompt: str, image_base64: Optional[str] = None, max_retries: int = 10) -> str:
        """
        Unified API call method (supports text and multimodal)
        
        Args:
            prompt: Text prompt
            image_base64: base64-encoded image (VLM mode usage)
            max_retries: Max retries (default 10)
            
        Returns:
            API response text
        """
        import time
        import re
        
        for attempt in range(max_retries):
            try:
                port = self._get_next_vllm_port()
                
                # Create payload using Model config
                url = f"http://{self.vllm_host}:{port}{self.model_config.api_endpoint}"
                payload = self.model_config.create_api_payload(
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    image_base64=image_base64,
                )
                
                # Send request
                timeout = 180 if image_base64 else 120
                response = self.client.post(url, json=payload, timeout=timeout)
                response.raise_for_status()
                response_json = response.json()
                
                # Extract response using Model config
                response_text = self.model_config.extract_response(response_json, prompt)
                
                # Validate response
                numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", response_text.strip())
                
                if response_text.strip() and len(numbers) > 0:
                    if attempt > 0:
                        self.logger.info(f"✓ Retry successful on attempt {attempt + 1}")
                    return response_text
                else:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Attempt {attempt+1}: Empty/invalid response, retrying...")
                        time.sleep(0.5)
                        continue
                    return ""
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Attempt {attempt+1}: {e}, retrying...")
                    time.sleep(0.5)
                    continue
                self.logger.error(f"API request failed after {max_retries} attempts: {e}")
                return ""
        
        return ""
    
    def _parse_response(self, response_text: str, num_variates: int = 1) -> np.ndarray:
        """Parse API response (using utils function)"""
        # Use function from utils
        return parse_llm_response(
            response_text=response_text,
            prediction_length=self.prediction_length,
            num_variates=num_variates,
            enable_reasoning=self.enable_reasoning
        )


# ============================================================================
# Cleanup and signal handling (using utils functions)
# ============================================================================
# cleanup_vllm_servers() and setup_signal_handlers() moved to utils.py


# ============================================================================
# Pure inference flow (no metrics)
# ============================================================================

def run_inference():
    """Run pure inference flow"""
    
    print("\n" + "="*80)
    print("🚀 LLM Time Series Predictor - Inference Only Mode")
    print("="*80)
    
    # Display model info (get from registry)
    model_config = get_model_config(CONFIG["model_id"])
    model_type_icon = "🎨" if model_config.model_type == "vlm" else "📝"
    model_type_name = "MULTIMODAL (Vision-Language)" if model_config.model_type == "vlm" else "TEXT-ONLY"
    
    print(f"  Mode: {model_type_icon} {model_type_name}")
    print(f"  Model ID: {CONFIG['model_id']}")
    print(f"  Model: {model_config.model_name}")
    if model_config.model_type == "vlm":
        print(f"  Save Plots: {CONFIG.get('save_plots', True)}")
    print(f"  GPUs: {CONFIG['num_gpus']}")
    print(f"  Ports: {CONFIG['vllm_ports']}")
    print(f"  Samples: {CONFIG['num_samples']}")
    print(f"  Temperature: {CONFIG['temperature']}")
    print("="*80 + "\n")
    
    if CONFIG.get("auto_cleanup_vllm", True):
        # Use function from utils - only clean up our ports
        setup_signal_handlers(
            cleanup_callback=lambda: cleanup_vllm_servers(ports=CONFIG.get("vllm_ports"))
        )
    
    # Set data path
    os.environ["GIFT_EVAL"] = CONFIG["data_path"]
    print(f"Dataset: {CONFIG['data_path']}\n")
    
    # Load dataproperties
    dataset_properties_map = json.load(
        open(Path(__file__).parent / "dataset_properties.json")
    )
    
    # Parse dataset config
    dataset_configs_to_run = []
    if CONFIG.get("test_dataset_configs"):
        for config_str in CONFIG["test_dataset_configs"]:
            parts = config_str.split("/")
            if len(parts) >= 3:
                ds_name = "/".join(parts[:-1])
                term = parts[-1]
                dataset_configs_to_run.append((ds_name, term))
            elif len(parts) == 2:
                dataset_configs_to_run.append((config_str, "short"))
            else:
                dataset_configs_to_run.append((config_str, "short"))
    else:
        dataset_configs_to_run = [("m4_weekly", "short"), ("m4_monthly", "short")]
    
    # Output directory (in parent's results folder)
    output_dir = Path(__file__).parent.parent / "results" / CONFIG["output_model_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run inference
    for config_idx, (ds_name, term) in enumerate(dataset_configs_to_run, 1):
        print(f"\n{'='*80}")
        print(f"[{config_idx}/{len(dataset_configs_to_run)}] {ds_name} - {term}")
        print(f"{'='*80}")
        
        # Dataset info
        if "/" in ds_name:
            ds_key = ds_name.split("/")[0].lower()
            ds_freq = ds_name.split("/")[1]
        else:
            ds_key = ds_name.lower()
            ds_freq = dataset_properties_map.get(ds_key, {}).get("frequency", "D")
        
        # Handle dataset names with suffixes (e.g. _with_missing, _short)
        # Try to match base name (remove common suffixes)
        if ds_key not in dataset_properties_map:
            # Try removing common suffixes
            for suffix in ["_with_missing", "_without_missing", "_short", "_long"]:
                if ds_key.endswith(suffix):
                    base_key = ds_key.replace(suffix, "")
                    if base_key in dataset_properties_map:
                        ds_key = base_key
                        break
        
        ds_config = f"{ds_key}/{ds_freq}/{term}"
        
        # Check if completed (cross-dataset recovery)
        pred_file = output_dir / f"intermediate_{ds_config.replace('/', '_')}.json"
        if pred_file.exists():
            try:
                with open(pred_file, 'r') as f:
                    data = json.load(f)
                    cached_preds = data.get('predictions', [])
                    
                    # Quick check: read dataset size
                    try:
                        temp_dataset = Dataset(name=ds_name, term=term, to_univariate=False)
                        try:
                            if temp_dataset.target_dim > 1:
                                temp_dataset = Dataset(name=ds_name, term=term, to_univariate=True)
                        except:
                            pass
                        
                        # Calculate total sequences (considering rolling windows)
                        test_data_list = list(temp_dataset.test_data.input)
                        total_series = len(test_data_list)
                        
                        # Check if all sequences predicted and non-zero
                        num_cached = len(cached_preds)
                        num_valid = sum(1 for p in cached_preds if p.get('prediction') and not np.all(np.array(p['prediction']) == 0))
                        
                        if num_cached >= total_series and num_valid >= total_series:
                            print(f"ALREADY COMPLETED ({num_valid}/{total_series} valid predictions)")
                            print(f"     File: {pred_file.name}")
                            continue  # Skip this dataset
                        elif num_cached > 0:
                            print(f"  ⚡ RESUME: Found {num_valid} valid predictions, continuing...")
                    except Exception as e:
                        print(f"Could not verify completion: {e}")
                        print(f"     Will continue with inference...")
            except Exception as e:
                print(f"Could not read cache file: {e}")
        
        try:
            # Load dataset
            to_univariate = False
            try:
                dataset = Dataset(name=ds_name, term=term, to_univariate=False)
                if dataset.target_dim > 1:
                    to_univariate = True
                    dataset = Dataset(name=ds_name, term=term, to_univariate=True)
            except:
                dataset = Dataset(name=ds_name, term=term, to_univariate=False)
            
            domain = dataset_properties_map.get(ds_key, {}).get("domain", "Unknown")
            
            print(f"  Domain: {domain}")
            print(f"  Frequency: {dataset.freq}")
            print(f"  Prediction length: {dataset.prediction_length}")
            print(f"  Windows: {dataset.windows}")
            
            # Initialize predictor (simplified - use Model registry ID)
            predictor = LLMTimeSeriesPredictor(
                model_id=CONFIG["model_id"],
                prediction_length=dataset.prediction_length,
                num_samples=CONFIG["num_samples"],
                temperature=CONFIG["temperature"],
                max_context_length=CONFIG["max_context_length"],
                enable_reasoning=CONFIG["enable_reasoning"],
                domain=domain,
                freq=ds_freq,
                save_intermediate=CONFIG["save_predictions"],
                vllm_ports=CONFIG.get("vllm_ports", [8000]),
                vllm_host=CONFIG.get("vllm_host", "127.0.0.1"),
                max_tokens=CONFIG.get("max_tokens", 5000),
                top_p=CONFIG.get("top_p", 1.0),
                top_k=CONFIG.get("top_k", -1),
                save_plots=CONFIG.get("save_plots", True),
            )
            
            # Set save file
            if CONFIG["save_predictions"]:
                predictor.intermediate_file = pred_file
                predictor.dataset_config = ds_config
                predictor.model_name_for_save = CONFIG.get("output_model_name", predictor.model_name)
                print(f"Saving predictions to: {pred_file.name}")
            
            # Set plot save directory (auto-used by VLM models)
            if predictor.enable_vlm and CONFIG.get("save_plots"):
                plot_dir = output_dir / "plots" / ds_config.replace('/', '_')
                plot_dir.mkdir(parents=True, exist_ok=True)
                predictor.plot_dir = plot_dir
                print(f"Saving plots to: {plot_dir}")
            
            # Start inference
            print(f"  Starting inference...")
            n_windows = dataset.windows
            
            forecast_windows = []
            for window_idx in range(n_windows):
                print(f"    Window {window_idx + 1}/{n_windows}")
                
                entries_window_k = list(
                    itertools.islice(dataset.test_data.input, window_idx, None, n_windows)
                )
                
                forecasts_window_k = list(predictor.predict(
                    entries_window_k,
                    enable_parallel=CONFIG.get("enable_parallel", True),
                    max_workers=CONFIG.get("max_workers", 5),
                    global_start_idx=window_idx,
                    window_stride=n_windows
                ))
                forecast_windows.append(forecasts_window_k)
            
            # Merge results
            forecasts = [item for items in zip(*forecast_windows) for item in items]
            
            print(f"Generated {len(forecasts)} forecasts")
            
            # Count zero predictions
            zero_count = sum(1 for f in forecasts if np.all(f.samples == 0))
            valid_count = len(forecasts) - zero_count
            print(f"Valid: {valid_count}/{len(forecasts)} ({valid_count/len(forecasts)*100:.1f}%)")
            
            if CONFIG["save_predictions"]:
                total_saved = len(json.load(open(predictor.intermediate_file))["predictions"]) if predictor.intermediate_file.exists() else 0
                print(f"{total_saved} predictions saved")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"Inference complete!")
    print(f"Predictions saved to: {output_dir}")
    print(f"{'='*80}\n")
    
    if CONFIG.get("auto_cleanup_vllm", True):
        # Only clean up vLLM servers on the ports we started
        cleanup_vllm_servers(ports=CONFIG.get("vllm_ports"))


if __name__ == "__main__":
    run_inference()

