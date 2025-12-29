import json
import logging
import re
from typing import List

import numpy as np
import torch
from dotenv import load_dotenv
from gluonts.model import Forecast
from gluonts.model.forecast import QuantileForecast
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load environment variables
load_dotenv()

# Setup logger with console handler
logger = logging.getLogger("Qwen3 Predictor")
logger.setLevel(logging.INFO)

# Remove existing handlers to avoid duplicates
logger.handlers = []

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(console_handler)


class Qwen3Predictor:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-30B-A3B",
        prediction_length: int = None,
        batch_size: int = 1,
        quantile_levels: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        domain: str = None,
        freq: str = None,
        max_new_tokens: int = 32768,
        enable_thinking: bool = True,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        **kwargs
    ):
        """
        Initialize Qwen3 Predictor.
        
        Args:
            model_name: HuggingFace model name
            prediction_length: Forecast horizon
            batch_size: Batch size for generation
            quantile_levels: Quantile levels for forecast
            domain: Domain of the time series (e.g., "Sales", "Energy", "Econ/Fin")
            freq: Frequency of the time series (e.g., "M", "D", "H")
            max_new_tokens: Maximum tokens to generate
            enable_thinking: Enable thinking mode for Qwen3
            device_map: Device mapping for model
            torch_dtype: Torch dtype for model
        """
        self.model_name = model_name
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.quantile_levels = quantile_levels
        self.domain = domain
        self.freq = freq
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        
        # Load tokenizer and model
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Handle torch_dtype conversion
        if torch_dtype == "auto":
            dtype_kwarg = "auto"
        elif isinstance(torch_dtype, torch.dtype):
            dtype_kwarg = torch_dtype
        elif isinstance(torch_dtype, str):
            dtype_kwarg = getattr(torch, torch_dtype, None)
            if dtype_kwarg is None:
                logger.warning(f"Unknown torch_dtype {torch_dtype}, using auto")
                dtype_kwarg = "auto"
        else:
            dtype_kwarg = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype_kwarg,
            device_map=device_map,
            **kwargs
        )
        logger.info("Model loaded successfully")

    def _format_ts_data(self, target: np.ndarray) -> str:
        """Format time series data as a string for the prompt."""
        if target.ndim == 1:
            # Univariate: format as a simple list
            return str(target.tolist())
        else:
            # Multivariate: format as list of lists where each inner list is a time step
            # target shape is (num_variates, time_steps), we want (time_steps, num_variates)
            if target.shape[0] < target.shape[1]:
                # Assume (num_variates, time_steps) format
                target_transposed = target.T  # (time_steps, num_variates)
            else:
                # Assume (time_steps, num_variates) format
                target_transposed = target
            return str(target_transposed.tolist())

    def _create_prompt(self, ts_data: str, domain: str, freq: str, horizon: int, num_variates: int = 1) -> str:
        """Create prompt for the model."""
        if num_variates > 1:
            prompt = (
                f"We have a time series in {domain} domain with the {freq} frequency, "
                f"{ts_data}, please forecast the next {horizon} time steps for all {num_variates} variates "
                f"({horizon * num_variates} total values) and the output text should only include the {horizon * num_variates} numeric values."
            )
        else:
            prompt = (
                f"We have a time series in {domain} domain with the {freq} frequency, "
                f"{ts_data}, please forecast the next {horizon} values and the output text should only include the {horizon} numeric values."
            )
        return prompt

    def _parse_numeric_values(self, text: str, expected_count: int) -> np.ndarray:
        """
        Parse numeric values from text output.
        Tries multiple strategies to extract numbers.
        """
        # Remove thinking content markers if present
        text = text.strip()
        
        # Strategy 1: Extract all numbers from text
        numbers = re.findall(r'-?\d+\.?\d*', text)
        
        if len(numbers) >= expected_count:
            # Take the first expected_count numbers
            values = [float(n) for n in numbers[:expected_count]]
            return np.array(values)
        
        # Strategy 2: Look for list-like patterns [num1, num2, ...]
        list_pattern = r'\[([^\]]+)\]'
        matches = re.findall(list_pattern, text)
        if matches:
            # Try to parse the first list match
            numbers_str = matches[0]
            numbers = re.findall(r'-?\d+\.?\d*', numbers_str)
            if len(numbers) >= expected_count:
                values = [float(n) for n in numbers[:expected_count]]
                return np.array(values)
        
        # Strategy 3: Look for comma-separated values
        # Split by common separators and extract numbers
        parts = re.split(r'[,;\n]', text)
        numbers = []
        for part in parts:
            nums = re.findall(r'-?\d+\.?\d*', part)
            numbers.extend(nums)
            if len(numbers) >= expected_count:
                break
        
        if len(numbers) >= expected_count:
            values = [float(n) for n in numbers[:expected_count]]
            return np.array(values)
        
        # If we still don't have enough, pad with the last value or repeat
        if len(numbers) > 0:
            values = [float(n) for n in numbers]
            # Pad with last value
            while len(values) < expected_count:
                values.append(values[-1] if values else 0.0)
            return np.array(values[:expected_count])
        
        # Fallback: return zeros
        logger.warning(f"Could not parse enough values from text. Expected {expected_count}, got {len(numbers)}. Returning zeros.")
        return np.zeros(expected_count)

    def _generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate predictions for a batch of prompts."""
        # Prepare messages
        messages_list = [[{"role": "user", "content": prompt}] for prompt in prompts]
        
        # Apply chat template
        texts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking
            )
            for messages in messages_list
        ]
        
        # Tokenize
        model_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32768  # Reasonable max length
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Use greedy decoding for deterministic results
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode outputs
        outputs = []
        for i, generated_id in enumerate(generated_ids):
            # Get only the newly generated tokens
            input_length = model_inputs.input_ids[i].shape[0]
            output_ids = generated_id[input_length:].tolist()
            
            # Parse thinking content if enabled
            if self.enable_thinking:
                try:
                    # Token ID for </think> is 151668
                    if 151668 in output_ids:
                        index = len(output_ids) - output_ids[::-1].index(151668)
                        content = self.tokenizer.decode(
                            output_ids[index:], skip_special_tokens=True
                        ).strip("\n")
                    else:
                        content = self.tokenizer.decode(
                            output_ids, skip_special_tokens=True
                        ).strip("\n")
                except (ValueError, IndexError):
                    content = self.tokenizer.decode(
                        output_ids, skip_special_tokens=True
                    ).strip("\n")
            else:
                content = self.tokenizer.decode(
                    output_ids, skip_special_tokens=True
                ).strip("\n")
            
            outputs.append(content)
        
        return outputs

    def predict(self, test_data_input) -> List[Forecast]:
        """
        Generate forecasts for test data.
        
        Args:
            test_data_input: List of dictionaries with keys:
                - "target": numpy array of time series values
                - "start": start date/time
                - "item_id": (optional) item identifier
                - "freq": (optional) frequency
        
        Returns:
            List of Forecast objects
        """
        if self.prediction_length is None:
            raise ValueError("prediction_length must be set")
        if self.domain is None:
            raise ValueError("domain must be set")
        if self.freq is None:
            raise ValueError("freq must be set")
        
        # Prepare prompts for all items
        prompts = []
        for item in test_data_input:
            ts_data_str = self._format_ts_data(item["target"])
            num_variates = item["target"].shape[0] if item["target"].ndim > 1 else 1
            prompt = self._create_prompt(
                ts_data_str, self.domain, self.freq, self.prediction_length, num_variates
            )
            prompts.append(prompt)
        logger.info(f"Prompts: {prompts}")
        # Generate predictions in batches
        all_forecasts = []
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i:i + self.batch_size]
            batch_items = test_data_input[i:i + self.batch_size]
            
            logger.info(f"Processing batch {i // self.batch_size + 1}/{(len(prompts) + self.batch_size - 1) // self.batch_size}")
            
            try:
                batch_outputs = self._generate_batch(batch_prompts)
            except torch.cuda.OutOfMemoryError or torch.OutOfMemoryError:
                logger.error(f"OutOfMemoryError at batch_size {self.batch_size}, reducing to {self.batch_size // 2}")
                if self.batch_size == 1:
                    raise
                self.batch_size //= 2
                # Retry with smaller batch
                batch_outputs = self._generate_batch(batch_prompts)
            
            # Parse outputs and create forecasts
            for output_text, item in zip(batch_outputs, batch_items):
                # Parse numeric values
                # For multivariate, we need to predict all variates
                is_multivariate = item["target"].ndim > 1 and item["target"].shape[0] > 1
                if is_multivariate:
                    num_variates = item["target"].shape[0]
                    total_values_needed = self.prediction_length * num_variates
                    predicted_values = self._parse_numeric_values(
                        output_text, total_values_needed
                    )
                    # Reshape to (prediction_length, num_variates)
                    predicted_values = predicted_values.reshape(self.prediction_length, num_variates)
                else:
                    predicted_values = self._parse_numeric_values(
                        output_text, self.prediction_length
                    )
                
                # Create quantile forecast
                # Since we only have point forecasts, we'll use the same values for all quantiles
                # or create symmetric quantiles around the median
                forecast_arrays = []
                for q in self.quantile_levels:
                    if q == 0.5:
                        # Use predicted values as median
                        forecast_arrays.append(predicted_values)
                    else:
                        # Create symmetric quantiles: for q < 0.5, subtract; for q > 0.5, add
                        # Use a small percentage (5%) of the absolute value as uncertainty
                        uncertainty = np.abs(predicted_values) * 0.05 * abs(q - 0.5) * 2
                        if q < 0.5:
                            quantile_values = predicted_values - uncertainty
                        else:
                            quantile_values = predicted_values + uncertainty
                        forecast_arrays.append(quantile_values)
                
                # Stack quantiles
                # For univariate: shape should be (num_quantiles, prediction_length)
                # For multivariate: shape should be (num_quantiles, prediction_length, num_variates)
                forecast_array = np.stack(forecast_arrays, axis=0)
                
                # For univariate, ensure shape is (num_quantiles, prediction_length)
                if not is_multivariate:
                    if forecast_array.ndim > 2:
                        forecast_array = forecast_array.squeeze(-1)
                
                # Create forecast start date
                forecast_start_date = item["start"] + len(item["target"])
                
                # Create QuantileForecast
                forecast = QuantileForecast(
                    forecast_arrays=forecast_array,
                    forecast_keys=list(map(str, self.quantile_levels)),
                    start_date=forecast_start_date,
                )
                all_forecasts.append(forecast)
        
        return all_forecasts


def evaluate_qwen3_on_gift_eval():
    """
    Main evaluation function for Qwen3 on gift-eval benchmark.
    Similar to chronos-2.ipynb evaluation.
    """
    import itertools
    import pandas as pd
    from pathlib import Path
    
    from gluonts.model import evaluate_forecasts
    from gluonts.time_feature import get_seasonality
    
    from gift_eval.data import Dataset
    from gluonts.ev.metrics import (
        MAE,
        MAPE,
        MASE,
        MSE,
        MSIS,
        ND,
        NRMSE,
        RMSE,
        SMAPE,
        MeanWeightedSumQuantileLoss,
    )
    
    # Setup logging filter for warnings
    class WarningFilter(logging.Filter):
        def __init__(self, text_to_filter):
            super().__init__()
            self.text_to_filter = text_to_filter

        def filter(self, record):
            return self.text_to_filter not in record.getMessage()

    gts_logger = logging.getLogger("gluonts.model.forecast")
    gts_logger.addFilter(
        WarningFilter("The mean prediction is not stored in the forecast data")
    )
    
    # Load dataset properties
    # Try multiple possible locations
    possible_paths = [
        Path(__file__).parent.parent / "notebooks" / "dataset_properties.json",  # From LLMs/ to notebooks/
        Path(__file__).parent / "notebooks" / "dataset_properties.json",  # If in root
        Path(__file__).parent / "dataset_properties.json",  # If in same directory
    ]
    
    dataset_properties_path = None
    for path in possible_paths:
        if path.exists():
            dataset_properties_path = path
            break
    
    if dataset_properties_path is None:
        raise FileNotFoundError(
            f"Could not find dataset_properties.json. Tried: {possible_paths}"
        )
    
    dataset_properties_map = json.load(open(dataset_properties_path))
    
    # Instantiate the metrics
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
    
    # Configuration
    # model_name = "Qwen/Qwen3-30B-A3B"
    model_name = "Qwen/Qwen3-4B-Thinking-2507"
    # Output to results directory in the project root
    output_dir = Path(__file__).parent.parent / "results" / "qwen3-4b-thinking-2507" / "all_results.csv"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    
    pretty_names = {
        "saugeenday": "saugeen",
        "temperature_rain_with_missing": "temperature_rain",
        "kdd_cup_2018_with_missing": "kdd_cup_2018",
        "car_parts_with_missing": "car_parts",
    }
    pretty_model_name = {
        "Qwen/Qwen3-30B-A3B": "Qwen3-30B-A3B",
        "Qwen/Qwen3-4B-Thinking-2507": "Qwen3-4B-Thinking-2507",
    }
    
    # Dataset configuration - modify as needed
    SHORT_DATASETS = "m4_weekly"
    MED_LONG_DATASETS = "bizitobs_l2c/H"
    
    # Get union of short and med_long datasets
    all_datasets = list(set(SHORT_DATASETS.split() + MED_LONG_DATASETS.split()))
    
    def evaluate_on_dataset(
        model_name: str,
        ds_name: str,
        ds_term: str,
        batch_size: int,
        use_multivariate_data: bool = True,
        **predictor_kwargs,
    ):
        is_multivariate_source = (
            Dataset(
                name=ds_name,
                term=ds_term,
                to_univariate=False,
            ).target_dim
            > 1
        )

        dataset = Dataset(
            name=ds_name,
            term=ds_term,
            to_univariate=is_multivariate_source and not use_multivariate_data,
        )

        logger.info(f"Dataset size: {len(dataset.test_data)}")

        # Get domain and frequency for the dataset
        if "/" in ds_name:
            ds_key = ds_name.split("/")[0]
            ds_freq = ds_name.split("/")[1]
            ds_key = ds_key.lower()
            ds_key = pretty_names.get(ds_key, ds_key)
        else:
            ds_key = ds_name.lower()
            ds_key = pretty_names.get(ds_key, ds_key)
            ds_freq = dataset_properties_map[ds_key]["frequency"]
        
        domain = dataset_properties_map[ds_key]["domain"]
        
        predictor = Qwen3Predictor(
            model_name=model_name,
            prediction_length=dataset.prediction_length,
            batch_size=batch_size,
            domain=domain,
            freq=ds_freq,
            **predictor_kwargs,
        )
        
        # Avoid cross batch leakage of rolling evaluation by prediction of windows individually.
        forecast_windows = []
        n_windows = dataset.test_data.windows
        for window_idx in range(n_windows):
            entries_window_k = list(itertools.islice(dataset.test_data.input, window_idx, None, n_windows))
            forecasts_window_k = list(predictor.predict(entries_window_k))
            forecast_windows.append(forecasts_window_k)        

        forecasts = [item for items in zip(*forecast_windows) for item in items]  # interleave results again
        season_length = get_seasonality(dataset.freq)
        return evaluate_forecasts(
                forecasts,
                test_data=dataset.test_data,
                metrics=metrics,
                batch_size=1024,
                axis=None,
                mask_invalid_label=True,
                allow_nan_forecast=False,
                seasonality=season_length,
            ) \
        .reset_index(drop=True) \
        .to_dict(orient="records")

    all_results = []
    for ds_num, ds_name in enumerate(all_datasets):
        ds_key = ds_name.split("/")[0]
        logger.info(f"Processing dataset: {ds_name} ({ds_num + 1} of {len(all_datasets)})")
        print(f"Processing dataset: {ds_name} ({ds_num + 1} of {len(all_datasets)})")
        terms = ["short", "medium", "long"]
        for term in terms:
            logger.info(f"Processing term: {term}")
            if (term == "medium" or term == "long") and ds_name not in MED_LONG_DATASETS.split():
                continue

            if "/" in ds_name:
                ds_key = ds_name.split("/")[0]
                ds_freq = ds_name.split("/")[1]
                ds_key = ds_key.lower()
                ds_key = pretty_names.get(ds_key, ds_key)
            else:
                ds_key = ds_name.lower()
                ds_key = pretty_names.get(ds_key, ds_key)
                ds_freq = dataset_properties_map[ds_key]["frequency"]
            ds_config = f"{ds_key}/{ds_freq}/{term}"

            logger.info(f"Generating forecasts for {ds_config}")
            all_results.append(
                (
                    evaluate_on_dataset(
                        model_name=model_name,
                        ds_name=ds_name,
                        ds_term=term,
                        batch_size=4,  # Start with batch_size=1 for LLM
                        use_multivariate_data=True,
                        device_map="cuda",
                        # torch_dtype="float32",
                        torch_dtype="auto",
                    ),
                    ds_config,
                    dataset_properties_map[ds_key]["domain"],
                    dataset_properties_map[ds_key]["num_variates"],
                )
            )

    result_df_rows = []
    for result_metrics, ds_config, domain, num_variates in all_results:
        result_metrics = {f"eval_metrics/{k}": v for k, v in result_metrics[0].items()}

        result_df_rows.append(
            {
                "dataset": ds_config,
                "model": pretty_model_name.get(model_name, model_name),
                **result_metrics,
                "domain": domain,
                "num_variates": num_variates,
            }
        )
    results_df = pd.DataFrame(result_df_rows).sort_values(by="dataset")
    results_df.to_csv(output_dir, index=False)
    logger.info(f"Results have been written to {output_dir}.")


if __name__ == "__main__":
    # Run evaluation
    evaluate_qwen3_on_gift_eval()
