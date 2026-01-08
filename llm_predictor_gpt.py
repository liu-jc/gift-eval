"""
LLM-based Time Series Predictor for GIFT-Eval Benchmark

This module provides a wrapper to evaluate LLM models (GPT-5, Gemini, etc.) 
on the GIFT-Eval benchmark using API calls.

Usage:
    1. Set your API configuration in the CONFIG section below
    2. Run the script: python llm_predictor.py
    3. Results will be saved to ../results/your_model_name/all_results.csv
"""

import os
import sys
import csv
import json
import logging
import itertools
from typing import Iterator, List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
import pandas as pd
from tqdm import tqdm

# GluonTS imports
from gluonts.model.forecast import SampleForecast, QuantileForecast
from gluonts.model import evaluate_model, evaluate_forecasts
from gluonts.time_feature import get_seasonality
from gluonts.ev.metrics import (
    MSE,
    MAE,
    MASE,
    MAPE,
    SMAPE,
    MSIS,
    RMSE,
    NRMSE,
    ND,
    MeanWeightedSumQuantileLoss,
)

# GIFT-Eval imports
from gift_eval.data import Dataset

# API imports (uncomment the one you need)
# import openai  # For OpenAI GPT models
# import anthropic  # For Claude models
# from google import generativeai as genai  # For Gemini models


# ============================================================================
# CONFIGURATION - MODIFY THIS SECTION
# ============================================================================

CONFIG = {
    # Data Configuration
    "data_path": "/home/zhiyuan/gift-eval-data",  # Path to GIFT-Eval dataset
    
    # API Configuration
    "api_provider": "openai",  # Options: "openai", "anthropic", "google"
    "api_key": "none",  # Replace with your actual API key
    "model_name": "gpt-5",  # Options: "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1-preview", "o1-mini"
    
    # Model Configuration
    "num_samples": 5,  # Number of samples for probabilistic forecasting (reduce for faster testing)
    "temperature": 0.7,  # Temperature for API calls (0.0-1.0, ignored for o1 models)
    "max_context_length": 4000,  # Maximum historical data points to include in prompt
    "enable_reasoning": False,  # If True, prompt LLM to show reasoning before prediction
    
    # Evaluation Configuration
    "batch_size": 8,  # Batch size for evaluation (reduce if API rate limits)
    "output_model_name": "gpt-5",  # Name to use in results file
    "model_type": "pretrained",  # Options: "statistical", "deep-learning", "pretrained"
    "model_dtype": "float32",
    "save_intermediate_results": True,  # Save prompt/response for debugging
    
    # 🔥 Parallel Processing Configuration
    "enable_parallel": True,  # Enable parallel API calls
    "max_workers": 8,  # Number of parallel threads (adjust based on API rate limits)
    
    # Dataset Configuration - Specify exact dataset configs to run
    # Format: "dataset_name/frequency/term" or just dataset names
    # Set to None to run default small subset
    # "test_dataset_configs": [
    #     # You can uncomment/add specific configs to test:
    #     "kdd_cup_2018_with_missing/D/short",
    #     "jena_weather/H/short",
    #     "bitbrains_rnd/H/short",
    #     "bizitobs_l2c/H/short",
    #     "restaurant/D/short",
    #     "hierarchical_sales/W/short",
    #     "ett1/H/short",
    #     "solar/W/short",
    #     "sz_taxi/H/short",
    #     "m_dense/D/short",
    #     "m4_hourly/H/short",
    #     "m4_weekly/W/short",
    #     "covid_deaths/D/short",
    #     "hospital/M/short",
    # ],
    "test_dataset_configs": [
    # ✅ 单变量数据集 (num_variates = 1)
        "kdd_cup_2018_with_missing/D/short",  # Nature, Daily (has freq subfolder)
        "restaurant",                    # Sales, Daily (no freq subfolder)
        "hierarchical_sales/W/short",          # Sales, Weekly (has freq subfolder)
        "solar/W/short",                       # Energy, Weekly (has freq subfolder)
        "sz_taxi/H/short",                     # Transport, Hourly (has freq subfolder)
        "m_dense/D/short",                     # Transport, Daily (has freq subfolder)
        "m4_hourly",                     # Econ/Fin, Hourly (no freq subfolder)
        "m4_weekly",                     # Econ/Fin, Weekly (no freq subfolder)
        "covid_deaths",                  # Healthcare, Daily (no freq subfolder)
        "hospital",                      # Healthcare, Monthly (no freq subfolder)
    ],
}


# ============================================================================
# LLM PREDICTOR CLASS
# ============================================================================

class LLMTimeSeriesPredictor:
    """
    Wrapper class for LLM-based time series prediction compatible with GluonTS evaluation.
    Supports both univariate and multivariate time series.
    """
    
    def __init__(
        self,
        api_provider: str,
        api_key: str,
        model_name: str,
        prediction_length: int,
        num_samples: int = 20,
        temperature: float = 0.7,
        max_context_length: int = 4000,
        enable_reasoning: bool = False,
        domain: str = None,
        freq: str = None,
        save_intermediate: bool = False,
    ):
        """
        Initialize the LLM predictor.
        
        Args:
            api_provider: API provider name ("openai", "anthropic", "google")
            api_key: API key for authentication
            model_name: Model name (e.g., "gpt-4o", "claude-3-opus", "gemini-pro")
            prediction_length: Number of future points to predict
            num_samples: Number of samples for probabilistic forecasting
            temperature: Temperature for API calls
            max_context_length: Maximum historical data points to use
            enable_reasoning: If True, prompt model to show reasoning before prediction
            domain: Domain of time series (e.g., "Sales", "Energy")
            freq: Frequency of time series (e.g., "H", "D", "M")
            save_intermediate: If True, save prompts and responses
        """
        self.api_provider = api_provider.lower()
        self.api_key = api_key
        self.model_name = model_name
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.temperature = temperature
        self.max_context_length = max_context_length
        self.enable_reasoning = enable_reasoning
        self.domain = domain
        self.freq = freq
        self.save_intermediate = save_intermediate
        
        # Storage for intermediate results
        self.intermediate_results = []
        
        # Thread safety
        self._file_lock = threading.Lock()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Initialize API client
        self._init_api_client()
    
    def _init_api_client(self):
        """Initialize the appropriate API client."""
        if self.api_provider == "openai":
            import openai
            openai.api_key = self.api_key
            self.client = openai
        elif self.api_provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        elif self.api_provider == "google":
            from google import generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model_name)
        else:
            raise ValueError(f"Unsupported API provider: {self.api_provider}")
    
    def _predict_single_series(self, entry, idx):
        """
        Predict all samples for a single series (helper for parallel processing).
        
        Args:
            entry: Data entry dict
            idx: Series index
            
        Returns:
            SampleForecast object
        """
        try:
            # Get historical data
            historical_data = entry["target"]
            start_date = entry["start"]
            item_id = entry.get("item_id", f"series_{idx}")
            
            # 🔥 Check cache first (for within-dataset resume capability)
            if hasattr(self, 'cached_predictions') and idx in self.cached_predictions:
                cached_pred = self.cached_predictions[idx]
                
                # Replicate cached prediction for all samples
                if cached_pred.ndim == 1:
                    # Univariate: (prediction_length,) -> (num_samples, prediction_length)
                    samples_array = np.tile(cached_pred, (self.num_samples, 1))
                else:
                    # Multivariate: (prediction_length, num_variates) -> (num_samples, prediction_length, num_variates)
                    samples_array = np.tile(cached_pred[np.newaxis, :, :], (self.num_samples, 1, 1))
                
                # Calculate forecast start date
                if historical_data.ndim > 1:
                    length = historical_data.shape[1] if historical_data.shape[0] < historical_data.shape[1] else historical_data.shape[0]
                else:
                    length = len(historical_data)
                forecast_start_date = start_date + length
                
                # Return cached forecast (no API call!)
                return SampleForecast(
                    samples=samples_array,
                    start_date=forecast_start_date
                )
            
            # Not in cache, need to predict via API
            # Determine if multivariate
            is_multivariate = historical_data.ndim > 1 and (
                historical_data.shape[0] > 1 if historical_data.shape[0] < historical_data.shape[1]
                else historical_data.shape[1] > 1
            )
            
            # Generate multiple samples for probabilistic forecasting
            samples = []
            for sample_idx in range(self.num_samples):
                prediction = self._predict_single(historical_data, sample_idx, item_id, idx)
                samples.append(prediction)
            
            # Convert to numpy array
            samples_array = np.array(samples)
            
            # Calculate forecast start date
            if historical_data.ndim > 1:
                length = historical_data.shape[1] if historical_data.shape[0] < historical_data.shape[1] else historical_data.shape[0]
            else:
                length = len(historical_data)
            forecast_start_date = start_date + length
            
            # Return SampleForecast object
            return SampleForecast(
                samples=samples_array,
                start_date=forecast_start_date
            )
            
        except Exception as e:
            self.logger.error(f"Error predicting series {idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Return zero forecast as fallback
            historical_data = entry["target"]
            if historical_data.ndim > 1:
                num_variates = historical_data.shape[0] if historical_data.shape[0] < historical_data.shape[1] else historical_data.shape[1]
                fallback_shape = (self.num_samples, self.prediction_length, num_variates)
                length = historical_data.shape[1] if historical_data.shape[0] < historical_data.shape[1] else historical_data.shape[0]
            else:
                fallback_shape = (self.num_samples, self.prediction_length)
                length = len(historical_data)
            
            forecast_start_date = entry["start"] + length
            return SampleForecast(
                samples=np.zeros(fallback_shape),
                start_date=forecast_start_date
            )
    
    def _load_cached_predictions(self):
        """
        Load cached predictions from intermediate file if it exists.
        Returns a dict mapping series_idx -> prediction array.
        Skips all-zero predictions (they will be retried).
        """
        cached_predictions = {}
        zero_predictions = []
        
        if hasattr(self, 'intermediate_file') and self.intermediate_file and self.intermediate_file.exists():
            try:
                with open(self.intermediate_file, 'r') as f:
                    data = json.load(f)
                    predictions = data.get('predictions', [])
                    
                    # Build cache: series_idx -> prediction array
                    for pred in predictions:
                        series_idx = pred.get('series_idx')
                        prediction_list = pred.get('prediction')
                        if series_idx is not None and prediction_list is not None:
                            pred_array = np.array(prediction_list)
                            
                            # Check if prediction is all zeros
                            if np.all(pred_array == 0):
                                zero_predictions.append(series_idx)
                                # Don't cache zero predictions - they will be retried
                                continue
                            
                            cached_predictions[series_idx] = pred_array
                    
                    total_found = len(cached_predictions) + len(zero_predictions)
                    if total_found > 0:
                        print(f"  📦 Found {total_found} predictions in cache:")
                        print(f"     → Valid (non-zero): {len(cached_predictions)} - will reuse")
                        if zero_predictions:
                            print(f"     → Zero predictions: {len(zero_predictions)} - will retry with new API calls")
            except Exception as e:
                self.logger.warning(f"Could not load cached predictions: {e}")
        
        return cached_predictions
    
    def predict(self, test_data_input, enable_parallel=True, max_workers=5, global_start_idx=0, window_stride=1, **kwargs) -> Iterator[SampleForecast]:
        """
        Generate predictions for test data (supports parallel processing).
        
        Args:
            test_data_input: Iterator of test data entries
            enable_parallel: Whether to use parallel processing
            max_workers: Number of parallel threads
            global_start_idx: Starting index for this window (for multi-window scenarios)
            window_stride: Stride between consecutive series in this window (for multi-window scenarios)
            
        Yields:
            SampleForecast objects for each test entry
        """
        test_data_list = list(test_data_input)
        total_series = len(test_data_list)
        
        # 🔥 Load cached predictions (for resume capability within a dataset)
        self.cached_predictions = self._load_cached_predictions()
        num_cached = len(self.cached_predictions)
        num_to_predict = total_series - num_cached
        
        if num_cached > 0:
            print(f"  ⚡ RESUME: {num_cached}/{total_series} series cached, will predict {num_to_predict} new ones")
        
        if enable_parallel and max_workers > 1:
            self.logger.info(f"🚀 Starting PARALLEL prediction on {total_series} series with {self.num_samples} samples each ({max_workers} workers)")
            
            # Use ThreadPoolExecutor for parallel API calls
            results = [None] * total_series  # Preserve order
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks (calculate global series_idx for correct caching)
                # Global idx = local_idx * window_stride + window_offset
                future_to_idx = {
                    executor.submit(self._predict_single_series, entry, idx * window_stride + global_start_idx): idx
                    for idx, entry in enumerate(test_data_list)
                }
                
                # Process completed tasks with progress bar
                with tqdm(total=total_series, desc="Parallel Predicting") as pbar:
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            result = future.result()
                            results[idx] = result
                        except Exception as e:
                            self.logger.error(f"Task {idx} failed: {e}")
                            # Fallback already handled in _predict_single_series
                            results[idx] = future.result()
                        pbar.update(1)
            
            # Yield results in order
            for result in results:
                yield result
        
        else:
            # Sequential processing (original behavior)
            self.logger.info(f"Starting SEQUENTIAL prediction on {total_series} series with {self.num_samples} samples each")
            
            for idx, entry in enumerate(tqdm(test_data_list, desc="Predicting")):
                yield self._predict_single_series(entry, idx * window_stride + global_start_idx)
        
        self.logger.info("Prediction completed.")
    
    def _predict_single(self, historical_data: np.ndarray, sample_idx: int = 0, 
                        item_id: str = None, series_idx: int = None) -> np.ndarray:
        """
        Generate a single prediction sample.
        
        Args:
            historical_data: Historical time series data (1D or 2D)
            sample_idx: Sample index (for logging)
            item_id: Item identifier
            series_idx: Series index in batch
            
        Returns:
            Predicted values as numpy array
        """
        # Determine number of variates
        if historical_data.ndim > 1:
            num_variates = historical_data.shape[0] if historical_data.shape[0] < historical_data.shape[1] else historical_data.shape[1]
            # Limit context length on time dimension
            if historical_data.shape[0] < historical_data.shape[1]:
                # Shape is (num_variates, time_steps)
                if historical_data.shape[1] > self.max_context_length:
                    context_data = historical_data[:, -self.max_context_length:]
                else:
                    context_data = historical_data
            else:
                # Shape is (time_steps, num_variates)
                if historical_data.shape[0] > self.max_context_length:
                    context_data = historical_data[-self.max_context_length:, :]
                else:
                    context_data = historical_data
        else:
            num_variates = 1
            # Limit context length (take most recent data points)
            if len(historical_data) > self.max_context_length:
                context_data = historical_data[-self.max_context_length:]
            else:
                context_data = historical_data
        
        # Create prompt
        prompt = self._create_prompt(context_data)
        
        # Call API
        response_text = self._call_api(prompt)
        
        # Parse response
        prediction = self._parse_response(response_text, num_variates)
        
        # Save intermediate results if enabled
        if self.save_intermediate and sample_idx == 0:  # Only save first sample to reduce storage
            result_entry = {
                "item_id": item_id,
                "series_idx": series_idx,
                "sample_idx": sample_idx,
                "num_variates": num_variates,
                "context_length": context_data.shape[-1] if context_data.ndim > 1 else len(context_data),
                "prompt": prompt,
                "response": response_text,
                "prediction": prediction.tolist(),
            }
            self.intermediate_results.append(result_entry)
            
            # 🔥 Real-time save: immediately write to file if intermediate_file is set
            if hasattr(self, 'intermediate_file') and self.intermediate_file:
                self._save_intermediate_incremental(result_entry)
        
        return prediction
    
    def _save_intermediate_incremental(self, result_entry):
        """
        Save a single prediction result incrementally to the intermediate file (thread-safe).
        If a prediction for the same series_idx already exists, it will be replaced (for retry support).
        """
        try:
            # Use lock to ensure thread-safe file access
            with self._file_lock:
                # Read existing data if file exists
                if self.intermediate_file.exists():
                    with open(self.intermediate_file, 'r') as f:
                        data = json.load(f)
                else:
                    # Create initial structure
                    data = {
                        "dataset_config": getattr(self, 'dataset_config', 'unknown'),
                        "model": getattr(self, 'model_name_for_save', 'unknown'),
                        "enable_reasoning": self.enable_reasoning,
                        "num_samples": self.num_samples,
                        "predictions": []
                    }
                
                # Check if prediction for this series_idx already exists
                series_idx = result_entry.get('series_idx')
                existing_idx = None
                for i, pred in enumerate(data["predictions"]):
                    if pred.get('series_idx') == series_idx:
                        existing_idx = i
                        break
                
                if existing_idx is not None:
                    # Replace existing prediction (retry case)
                    data["predictions"][existing_idx] = result_entry
                else:
                    # Append new result
                    data["predictions"].append(result_entry)
                
                # Write back
                with open(self.intermediate_file, 'w') as f:
                    json.dump(data, f, indent=2)
                    
        except Exception as e:
            self.logger.warning(f"Failed to save intermediate result: {e}")
    
    def _format_ts_data(self, target: np.ndarray, max_points: int = None) -> str:
        """
        Format time series data as string (supports univariate and multivariate).
        
        Args:
            target: Time series data (1D for univariate, 2D for multivariate)
            max_points: Maximum number of time points to include
            
        Returns:
            Formatted string representation
        """
        if max_points is None:
            max_points = self.max_context_length
        
        if target.ndim == 1:
            # Univariate: shape (time_steps,)
            context = target[-max_points:]
            return ", ".join([f"{x:.4f}" for x in context])
        else:
            # Multivariate: shape (num_variates, time_steps) or (time_steps, num_variates)
            # Normalize to (num_variates, time_steps)
            if target.shape[0] > target.shape[1]:
                # Assume (time_steps, num_variates) - transpose it
                target = target.T
            
            # Take last max_points time steps
            context = target[:, -max_points:]
            num_variates = context.shape[0]
            time_steps = context.shape[1]
            
            # Format as [[v1_t1, v2_t1, ...], [v1_t2, v2_t2, ...], ...]
            formatted_steps = []
            for t in range(time_steps):
                values = [f"{context[v, t]:.4f}" for v in range(num_variates)]
                formatted_steps.append(f"[{', '.join(values)}]")
            
            return ", ".join(formatted_steps)
    
    def _create_prompt(self, historical_data: np.ndarray) -> str:
        """
        Create prompt for the LLM with two modes: with/without reasoning.
        
        Args:
            historical_data: Historical time series data
            
        Returns:
            Prompt string
        """
        # Determine if multivariate
        is_multivariate = historical_data.ndim > 1 and historical_data.shape[0] > 1
        if is_multivariate:
            num_variates = historical_data.shape[0] if historical_data.shape[0] < historical_data.shape[1] else historical_data.shape[1]
            total_values_needed = self.prediction_length * num_variates
        else:
            num_variates = 1
            total_values_needed = self.prediction_length
        
        # Format historical data
        history_str = self._format_ts_data(historical_data)
        
        # Calculate basic statistics for univariate
        if not is_multivariate:
            recent_data = historical_data[-min(100, len(historical_data)):]
            mean_val = np.mean(recent_data)
            std_val = np.std(recent_data)
            trend = "increasing" if recent_data[-1] > recent_data[0] else "decreasing"
            has_stats = True
        else:
            has_stats = False
        
        # Domain and frequency context
        domain_info = f"in the {self.domain} domain" if self.domain else ""
        freq_info = f"with {self.freq} frequency" if self.freq else ""
        context_str = f"{domain_info} {freq_info}".strip()
        
        # Build prompt based on reasoning mode
        if self.enable_reasoning:
            # Prompt with reasoning - structured output
            if is_multivariate:
                prompt = f"""You are an expert in time series forecasting. Given a multivariate time series {context_str}, predict future values.

**Historical Data** ({num_variates} variables, most recent data shown):
{history_str}

**Task**: 
Forecast the next {self.prediction_length} time steps for all {num_variates} variables (total {total_values_needed} values).

**Instructions**:
1. Analyze the data and explain your reasoning
2. Provide predictions in the specified format

**Output Format** (strictly follow this):
Rationale: [Your analysis of trends, patterns, seasonality, etc.]
Prediction: [v1_t1, v2_t1, ..., v{num_variates}_t1], [v1_t2, v2_t2, ..., v{num_variates}_t2], ..., [v1_t{self.prediction_length}, v2_t{self.prediction_length}, ..., v{num_variates}_t{self.prediction_length}]

Important: 
- The Prediction line must contain exactly {total_values_needed} numbers
- Format as {self.prediction_length} groups of {num_variates} values each
- Use only numbers after "Prediction:", no text

Begin:"""
            else:
                stats_info = f"\n- Mean: {mean_val:.4f}, Std Dev: {std_val:.4f}, Trend: {trend}" if has_stats else ""
                prompt = f"""You are an expert in time series forecasting. Given a time series {context_str}, predict future values.

**Historical Data** (most recent {len(historical_data[-self.max_context_length:])} points):
{history_str}{stats_info}

**Task**: 
Predict the next {self.prediction_length} values.

**Instructions**:
1. Analyze patterns (trends, seasonality, cycles, anomalies)
2. Consider domain knowledge and frequency characteristics
3. Provide your reasoning and then predictions

**Output Format** (strictly follow this):
Rationale: [Your analysis here - consider trends, patterns, domain context]
Prediction: [val1, val2, val3, ..., val{self.prediction_length}]

Important:
- The Prediction line must contain exactly {self.prediction_length} comma-separated numbers
- Use only numbers after "Prediction:", no explanatory text

Begin:"""
        else:
            # Direct prediction mode - concise prompt
            if is_multivariate:
                prompt = f"""Time series {context_str} with {num_variates} variables: {history_str}

Forecast the next {self.prediction_length} time steps for all {num_variates} variables ({total_values_needed} total values).

Output format: [v1_t1, v2_t1, ...], [v1_t2, v2_t2, ...], ..., [v1_t{self.prediction_length}, v2_t{self.prediction_length}, ...]

Output only {total_values_needed} numbers in {self.prediction_length} groups:"""
            else:
                prompt = f"""Time series {context_str}: {history_str}

Predict the next {self.prediction_length} values. Output only {self.prediction_length} comma-separated numbers:"""
        
        return prompt
    
    def _call_api(self, prompt: str, max_retries: int = 5) -> str:
        """
        Call the LLM API with retry mechanism.
        
        Args:
            prompt: Input prompt
            max_retries: Maximum number of retries if response is empty or invalid
            
        Returns:
            Response text from the API
        """
        import time
        
        for attempt in range(max_retries):
            try:
                if self.api_provider == "openai":
                    # Build API params
                    api_params = {
                        "model": self.model_name,
                        "messages": [
                            {"role": "system", "content": "You are a time series forecasting expert."},
                            {"role": "user", "content": prompt}
                        ],
                        "max_completion_tokens": 5000,
                    }
                    
                    # Only add temperature if model supports it (not for o1 series)
                    if "o1" not in self.model_name.lower() and "gpt-5" not in self.model_name.lower():
                        api_params["temperature"] = self.temperature
                    
                    response = self.client.ChatCompletion.create(**api_params)
                    response_text = response.choices[0].message.content
                
                elif self.api_provider == "anthropic":
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=5000,
                        temperature=self.temperature,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    response_text = response.content[0].text
                
                elif self.api_provider == "google":
                    response = self.client.generate_content(
                        prompt,
                        generation_config={
                            "temperature": self.temperature,
                            "max_output_tokens": 5000,
                        }
                    )
                    response_text = response.text
                
                # Check if response is valid (not empty and contains numbers)
                import re
                numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", response_text.strip())
                
                if response_text.strip() and len(numbers) > 0:
                    # Valid response, return it
                    if attempt > 0:
                        print(f"  ✓ Retry successful on attempt {attempt + 1}")
                    return response_text
                else:
                    # Empty or no numbers found
                    if attempt < max_retries - 1:
                        print(f"  ⚠️ Attempt {attempt + 1}/{max_retries}: Empty/invalid response, retrying...")
                        time.sleep(1)  # Small delay before retry
                        continue
                    else:
                        print(f"  ❌ All {max_retries} attempts failed: No valid response")
                        return ""  # Return empty after all retries exhausted
                
            except Exception as e:
                # API request failed
                if attempt < max_retries - 1:
                    print(f"  ⚠️ Attempt {attempt + 1}/{max_retries}: API error, retrying...")
                    print(f"     Error: {str(e)}")
                    time.sleep(1)  # Small delay before retry
                    continue
                else:
                    # All retries exhausted
                    print(f"\n{'='*80}")
                    print(f"❌ API REQUEST FAILED AFTER {max_retries} ATTEMPTS")
                    print(f"{'='*80}")
                    print(f"Error: {str(e)}")
                    print(f"Model: {self.model_name}")
                    print(f"Provider: {self.api_provider}")
                    print(f"\nTraceback:")
                    import traceback
                    traceback.print_exc()
                    print(f"{'='*80}\n")
                    
                    self.logger.error(f"API request failed after {max_retries} attempts: {str(e)}")
                    return ""  # Return empty string after all retries exhausted
        
        return ""  # Should not reach here, but return empty as fallback
    
    def _parse_response(self, response_text: str, num_variates: int = 1) -> np.ndarray:
        """
        Parse the API response to extract numerical predictions.
        Supports both reasoning and direct output modes, and multivariate data.
        
        Args:
            response_text: Raw text response from API
            num_variates: Number of variables (1 for univariate, >1 for multivariate)
            
        Returns:
            Numpy array of predictions
            - For univariate: shape (prediction_length,)
            - For multivariate: shape (prediction_length, num_variates)
        """
        try:
            text = response_text.strip()
            
            # 首先检查：response是否为空？
            if not text:
                # 情况2a: API请求失败导致返回空字符串（上面的except已经打印过错误了）
                print(f"⚠️ Response is empty (API request failed above)")
                if num_variates > 1:
                    return np.zeros((self.prediction_length, num_variates))
                else:
                    return np.zeros(self.prediction_length)
            
            # If reasoning mode, extract only the Prediction part
            if self.enable_reasoning:
                # Look for "Prediction:" marker
                if "Prediction:" in text or "prediction:" in text.lower():
                    # Extract everything after "Prediction:"
                    parts = text.lower().split("prediction:")
                    if len(parts) > 1:
                        text = parts[1]
            
            # Remove markdown code blocks if present
            if "```" in text:
                # Extract content between code blocks
                code_blocks = text.split("```")
                if len(code_blocks) > 1:
                    text = code_blocks[1].strip()
            
            # Extract all numbers from the text
            import re
            numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
            
            total_values_needed = self.prediction_length * num_variates
            
            if len(numbers) >= total_values_needed:
                # Successfully extracted enough numbers
                predictions = [float(x) for x in numbers[:total_values_needed]]
            elif len(numbers) > 0:
                # Not enough numbers, pad with repetition
                predictions = [float(x) for x in numbers]
                self.logger.warning(
                    f"Only found {len(numbers)} values, expected {total_values_needed}. Padding..."
                )
                self.logger.warning(f"Response preview: {response_text[:200]}...")
                # Pad by repeating the last value
                while len(predictions) < total_values_needed:
                    predictions.append(predictions[-1] if predictions else 0.0)
            else:
                # 情况2b: API请求成功，但响应中没有数字
                print(f"\n{'='*80}")
                print(f"⚠️ NO NUMBERS FOUND IN RESPONSE")
                print(f"{'='*80}")
                print(f"API request succeeded, but response contains no parseable numbers.")
                print(f"\n📥 Response content (first 500 chars):")
                print(f"   {response_text[:500]}")
                print(f"\n💡 Possible reasons:")
                print(f"   1. Model refused to provide numerical predictions")
                print(f"   2. Prompt format doesn't match model's understanding")
                print(f"   3. Model doesn't understand the task")
                print(f"{'='*80}\n")
                
                self.logger.warning(f"No numbers found in response: {response_text[:500]}")
                predictions = [0.0] * total_values_needed
            
            # Convert to numpy array and reshape
            predictions = np.array(predictions[:total_values_needed])
            
            if num_variates > 1:
                # Reshape to (prediction_length, num_variates)
                predictions = predictions.reshape(self.prediction_length, num_variates)
            
            return predictions
        
        except Exception as e:
            self.logger.error(f"Failed to parse response: {str(e)}. Using zeros.")
            if num_variates > 1:
                return np.zeros((self.prediction_length, num_variates))
            else:
                return np.zeros(self.prediction_length)


# ============================================================================
# EVALUATION PIPELINE
# ============================================================================

def run_evaluation():
    """Run the full evaluation pipeline on GIFT-Eval benchmark."""
    
    # Set data path environment variable
    os.environ["GIFT_EVAL"] = CONFIG["data_path"]
    print(f"Using dataset from: {CONFIG['data_path']}")
    
    # Validate configuration
    if CONFIG["api_key"] == "YOUR_API_KEY_HERE":
        print("ERROR: Please set your API key in the CONFIG section!")
        return
    
    # Load dataset properties
    dataset_properties_map = json.load(
        open(Path(__file__).parent / "dataset_properties.json")
    )
    
    # Define metrics
    metrics = [
        MSE(forecast_type="mean"),
        MSE(forecast_type=0.5),
        MAE(),
        MASE(),
        MAPE(),
        SMAPE(),
        MSIS(),
        RMSE(),
        NRMSE(),
        ND(),
        MeanWeightedSumQuantileLoss(
            quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ),
    ]
    
    # Parse dataset configurations
    # Support format: "dataset_name/frequency/term" or parse from config
    dataset_configs_to_run = []
    
    if CONFIG.get("test_dataset_configs"):
        # User specified exact configs like "m4_weekly/W/short"
        for config_str in CONFIG["test_dataset_configs"]:
            parts = config_str.split("/")
            if len(parts) >= 3:
                # Full format: dataset_name/freq/term
                ds_name = "/".join(parts[:-1])  # Everything except last is dataset name
                term = parts[-1]
                dataset_configs_to_run.append((ds_name, term))
            elif len(parts) == 2:
                # Format: dataset_name/freq, use short term by default
                dataset_configs_to_run.append((config_str, "short"))
            else:
                # Just dataset name, use short term
                dataset_configs_to_run.append((config_str, "short"))
        
        print(f"Testing {len(dataset_configs_to_run)} specific dataset configurations")
    else:
        # Fallback to default small subset
        all_datasets = ["m4_weekly", "m4_monthly"]
        print(f"No specific datasets specified. Running on default subset: {all_datasets}")
        dataset_configs_to_run = [(ds, "short") for ds in all_datasets]
    
    # Setup output directory
    output_dir = Path(__file__).parent.parent / "results" / CONFIG["output_model_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file_path = output_dir / "all_results.csv"
    
    # Pretty names mapping
    pretty_names = {
        "saugeenday": "saugeen",
        "temperature_rain_with_missing": "temperature_rain",
        "kdd_cup_2018_with_missing": "kdd_cup_2018",
        "car_parts_with_missing": "car_parts",
    }
    
    # 🔥 Check for existing results (for resume capability)
    completed_datasets = set()
    csv_exists = csv_file_path.exists()
    
    if csv_exists:
        print(f"\n{'='*80}")
        print(f"📋 RESUME MODE: Found existing results")
        print(f"{'='*80}")
        print(f"File: {csv_file_path}")
        try:
            with open(csv_file_path, "r") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    completed_datasets.add(row["dataset"])
            
            print(f"✓ Found {len(completed_datasets)} completed datasets")
            if completed_datasets:
                sample_list = sorted(list(completed_datasets)[:3])
                print(f"  Examples: {', '.join(sample_list)}" + 
                      (f" ... (+{len(completed_datasets) - 3} more)" if len(completed_datasets) > 3 else ""))
            print(f"→ Will skip these and continue with remaining datasets")
            print(f"{'='*80}\n")
        except Exception as e:
            print(f"⚠️ Could not read existing CSV: {e}")
            print(f"→ Starting fresh.\n")
            completed_datasets = set()
            csv_exists = False
    
    # Initialize CSV file only if it doesn't exist
    if not csv_exists:
        print(f"📄 Creating new results file: {csv_file_path}\n")
        with open(csv_file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "dataset",
                "model",
                # Prediction statistics
                "total_predictions",
                "valid_predictions",
                "zero_predictions",
                "success_rate",
                # All predictions (including zeros)
                "eval_metrics/MSE[mean]_all",
                "eval_metrics/MAE[0.5]_all",
                "eval_metrics/RMSE[mean]_all",
                "eval_metrics/MAPE[0.5]_all",
                "eval_metrics/sMAPE[0.5]_all",
                # Valid predictions only (zeros filtered out)
                "eval_metrics/MSE[mean]_valid",
                "eval_metrics/MAE[0.5]_valid",
                "eval_metrics/RMSE[mean]_valid",
                "eval_metrics/MAPE[0.5]_valid",
                "eval_metrics/sMAPE[0.5]_valid",
                # Other metrics
                "eval_metrics/MASE[0.5]",
                "eval_metrics/MSIS",
                "eval_metrics/NRMSE[mean]",
                "eval_metrics/ND[0.5]",
                "eval_metrics/mean_weighted_sum_quantile_loss",
                "domain",
                "num_variates",
            ])
    
    # Run evaluation on each dataset configuration
    for config_idx, (ds_name, term) in enumerate(dataset_configs_to_run, 1):
        print(f"\n{'='*80}")
        print(f"Processing [{config_idx}/{len(dataset_configs_to_run)}]: {ds_name} - {term}")
        print(f"{'='*80}")
        
        # Get dataset key and frequency
        if "/" in ds_name:
            ds_key = ds_name.split("/")[0].lower()
            ds_freq = ds_name.split("/")[1]
        else:
            ds_key = ds_name.lower()
            ds_freq = dataset_properties_map.get(ds_key, {}).get("frequency", "D")
        
        ds_key = pretty_names.get(ds_key, ds_key)
        ds_config = f"{ds_key}/{ds_freq}/{term}"
        
        # 🔥 Skip if already completed
        if ds_config in completed_datasets:
            print(f"  ⏭️  SKIPPING (already completed)")
            continue
        
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
            
            season_length = get_seasonality(dataset.freq)
            
            # Get domain info
            domain = dataset_properties_map.get(ds_key, {}).get("domain", "Unknown")
            
            print(f"  Domain: {domain}")
            print(f"  Frequency: {dataset.freq}")
            print(f"  Prediction length: {dataset.prediction_length}")
            print(f"  Windows: {dataset.windows}")
            print(f"  Target dimension: {dataset.target_dim}")
            
            # Initialize predictor
            predictor = LLMTimeSeriesPredictor(
                api_provider=CONFIG["api_provider"],
                api_key=CONFIG["api_key"],
                model_name=CONFIG["model_name"],
                prediction_length=dataset.prediction_length,
                num_samples=CONFIG["num_samples"],
                temperature=CONFIG["temperature"],
                max_context_length=CONFIG["max_context_length"],
                enable_reasoning=CONFIG["enable_reasoning"],
                domain=domain,
                freq=ds_freq,
                save_intermediate=CONFIG["save_intermediate_results"],
            )
            
            # 🔥 Set up real-time intermediate file saving
            if CONFIG["save_intermediate_results"]:
                intermediate_file = output_dir / f"intermediate_{ds_config.replace('/', '_')}.json"
                predictor.intermediate_file = intermediate_file
                predictor.dataset_config = ds_config
                predictor.model_name_for_save = CONFIG["output_model_name"]
                print(f"  📝 Real-time saving to: {intermediate_file}")
            
            # 🔥 Critical: Handle rolling windows to avoid cross-batch leakage
            print(f"  Starting evaluation with {dataset.windows} rolling windows...")
            n_windows = dataset.windows
            
            # Generate forecasts for each window separately
            forecast_windows = []
            for window_idx in range(n_windows):
                print(f"    Processing window {window_idx + 1}/{n_windows}")
                
                # Extract entries for this window using stride
                # itertools.islice(iterable, start, stop, step)
                # This takes every n_windows-th item starting from window_idx
                entries_window_k = list(
                    itertools.islice(dataset.test_data.input, window_idx, None, n_windows)
                )
                
                print(f"      Window has {len(entries_window_k)} series to predict")
                
                # 🔥 Calculate correct global start index for this window
                # For window_idx=0: series 0, n_windows, 2*n_windows, ... -> global_start_idx = 0
                # For window_idx=1: series 1, 1+n_windows, 1+2*n_windows, ... -> global_start_idx = 1
                # But we need to map local idx to global: local_idx * n_windows + window_idx
                # So we pass window_idx and stride info to predict method
                
                # Generate predictions for this window (with parallel processing if enabled)
                forecasts_window_k = list(predictor.predict(
                    entries_window_k,
                    enable_parallel=CONFIG.get("enable_parallel", True),
                    max_workers=CONFIG.get("max_workers", 5),
                    global_start_idx=window_idx,  # Starting offset for this window
                    window_stride=n_windows  # Stride between series in this window
                ))
                forecast_windows.append(forecasts_window_k)
            
            # Interleave results from all windows back together
            # This reconstructs the original order of test instances
            forecasts = [item for items in zip(*forecast_windows) for item in items]
            
            print(f"  Generated {len(forecasts)} total forecasts")
            
            # 🔥 Detect zero predictions (API failures)
            print(f"  Analyzing forecast quality...")
            total_predictions = len(forecasts)
            zero_forecasts_indices = []
            
            for idx, forecast in enumerate(forecasts):
                # Check if all samples are zeros
                samples = forecast.samples  # shape: (num_samples, pred_length) or (num_samples, pred_length, num_variates)
                if np.all(samples == 0):
                    zero_forecasts_indices.append(idx)
            
            num_zeros = len(zero_forecasts_indices)
            num_valid = total_predictions - num_zeros
            success_rate = num_valid / total_predictions if total_predictions > 0 else 0
            
            print(f"  📊 Prediction Quality:")
            print(f"     Total: {total_predictions}")
            print(f"     Valid (non-zero): {num_valid} ({success_rate*100:.1f}%)")
            print(f"     Zero (failed): {num_zeros} ({(1-success_rate)*100:.1f}%)")
            
            # 🔥 Evaluation 1: All predictions (including zeros)
            print(f"  Evaluating all predictions (including zeros)...")
            res_all = evaluate_forecasts(
                forecasts,
                test_data=dataset.test_data,
                metrics=metrics,
                batch_size=1024,
                axis=None,
                mask_invalid_label=True,
                allow_nan_forecast=False,
                seasonality=season_length,
            )
            res_all = res_all.reset_index(drop=True).to_dict(orient="records")[0]
            
            # 🔥 Evaluation 2: Valid predictions only (zeros filtered out)
            if num_valid > 0 and num_zeros > 0:
                print(f"  Evaluating valid predictions only (zeros filtered out)...")
                
                # Filter out zero forecasts
                valid_forecasts = [f for i, f in enumerate(forecasts) if i not in zero_forecasts_indices]
                
                # Filter corresponding test data entries
                test_data_list = list(dataset.test_data.input)
                valid_test_entries = [entry for i, entry in enumerate(test_data_list) if i not in zero_forecasts_indices]
                
                # Create filtered test data
                from gluonts.dataset.common import ListDataset
                valid_test_dataset = ListDataset(
                    valid_test_entries,
                    freq=dataset.freq
                )
                
                res_valid = evaluate_forecasts(
                    valid_forecasts,
                    test_data=valid_test_dataset,
                    metrics=metrics,
                    batch_size=1024,
                    axis=None,
                    mask_invalid_label=True,
                    allow_nan_forecast=False,
                    seasonality=season_length,
                )
                res_valid = res_valid.reset_index(drop=True).to_dict(orient="records")[0]
                
                print(f"  ✅ All metrics: MSE={res_all['MSE[mean]']:.4f}, MAE={res_all['MAE[0.5]']:.4f}")
                print(f"  ✅ Valid only metrics: MSE={res_valid['MSE[mean]']:.4f}, MAE={res_valid['MAE[0.5]']:.4f}")
            else:
                # All predictions are valid or all are zeros
                res_valid = res_all
                if num_zeros == 0:
                    print(f"  ✅ All predictions are valid!")
                else:
                    print(f"  ⚠️  All predictions are zeros - metrics will be identical")
            
            # Save results (both all and valid metrics)
            with open(csv_file_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    ds_config,
                    CONFIG["output_model_name"],
                    # Prediction statistics
                    total_predictions,
                    num_valid,
                    num_zeros,
                    f"{success_rate:.4f}",
                    # All predictions (including zeros)
                    res_all["MSE[mean]"],
                    res_all["MAE[0.5]"],
                    res_all["RMSE[mean]"],
                    res_all["MAPE[0.5]"],
                    res_all["sMAPE[0.5]"],
                    # Valid predictions only
                    res_valid["MSE[mean]"],
                    res_valid["MAE[0.5]"],
                    res_valid["RMSE[mean]"],
                    res_valid["MAPE[0.5]"],
                    res_valid["sMAPE[0.5]"],
                    # Other metrics (from all predictions)
                    res_all["MASE[0.5]"],
                    res_all["MSIS"],
                    res_all["NRMSE[mean]"],
                    res_all["ND[0.5]"],
                    res_all["mean_weighted_sum_quantile_loss"],
                    dataset_properties_map.get(ds_key, {}).get("domain", "Unknown"),
                    dataset_properties_map.get(ds_key, {}).get("num_variates", 1),
                ])
            
            # Note: Intermediate results are now saved in real-time during prediction
            # Clear the in-memory list for next dataset
            if CONFIG["save_intermediate_results"]:
                predictor.intermediate_results = []
                total_saved = len(json.load(open(predictor.intermediate_file))["predictions"]) if predictor.intermediate_file.exists() else 0
                print(f"  ✓ Total {total_saved} predictions saved to {predictor.intermediate_file.name}")
            
            print(f"  ✓ Results saved for {ds_config}")
            print(f"    Success rate: {success_rate*100:.1f}% ({num_valid}/{total_predictions})")
            print(f"    MSE (all): {res_all['MSE[mean]']:.4f}, MAE (all): {res_all['MAE[0.5]']:.4f}")
            if num_zeros > 0:
                print(f"    MSE (valid): {res_valid['MSE[mean]']:.4f}, MAE (valid): {res_valid['MAE[0.5]']:.4f}")
            
        except Exception as e:
            print(f"  ✗ Error processing {ds_config}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Save config
    config_file_path = output_dir / "config.json"
    with open(config_file_path, "w") as f:
        json.dump({
            "model": CONFIG["output_model_name"],
            "model_type": CONFIG["model_type"],
            "model_dtype": CONFIG["model_dtype"],
            "api_provider": CONFIG["api_provider"],
            "base_model": CONFIG["model_name"],
            "num_samples": CONFIG["num_samples"],
            "temperature": CONFIG["temperature"],
            "max_context_length": CONFIG["max_context_length"],
            "enable_reasoning": CONFIG["enable_reasoning"],
            "save_intermediate_results": CONFIG["save_intermediate_results"],
            "test_dataset_configs": CONFIG.get("test_dataset_configs", []),
        }, f, indent=4)
    
    print(f"\n{'='*80}")
    print(f"Evaluation complete!")
    print(f"Results saved to: {csv_file_path}")
    print(f"Config saved to: {config_file_path}")
    print(f"{'='*80}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    run_evaluation()

