#!/usr/bin/env python3
"""
Re-calculate evaluation metrics from cached intermediate files.
Run this from the project root directory.

Usage:
    cd /Users/zhiyuan/Desktop/gift-eval-main
    python recompute_metrics.py gpt-5
"""

import os
import sys
import csv
import json
import numpy as np
from pathlib import Path

# Setup environment
os.environ["GIFT_EVAL"] = os.environ.get("GIFT_EVAL", "/home/zhiyuan/gift-eval-data")

from gluonts.model.forecast import SampleForecast
from gluonts.model import evaluate_forecasts
from gluonts.time_feature import get_seasonality
from gluonts.dataset.common import ListDataset
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
from gift_eval.data import Dataset

# Load dataset properties from JSON
dataset_properties_map = json.load(
    open(Path(__file__).parent / "dataset_properties.json")
)

# Dataset name mapping (JSON name -> actual directory name)
# Some datasets were saved with cleaned names but the actual directory has suffixes
DATASET_NAME_MAP = {
    'kdd_cup_2018': 'kdd_cup_2018_with_missing',  # JSON has cleaned name, but directory has suffix
}

# Dataset config rewrite rules (for datasets without frequency subfolders)
# Maps "dataset/freq/term" -> "dataset" (removes freq subfolder)
DATASETS_WITHOUT_FREQ_SUBFOLDER = {
    'restaurant',      # No /D/ subfolder
    'm4_hourly',       # No /H/ subfolder
    'm4_weekly',       # No /W/ subfolder
    'm4_daily',        # No /D/ subfolder
    'm4_monthly',      # No /M/ subfolder
    'm4_quarterly',    # No /Q/ subfolder
    'm4_yearly',       # No /Y/ subfolder
    'covid_deaths',    # No /D/ subfolder
    'hospital',        # No /M/ subfolder
    'car_parts_with_missing',  # No freq subfolder
    'bizitobs_application',    # No freq subfolder
    'bizitobs_service',        # No freq subfolder
}


def load_forecasts_from_json(json_file, dataset):
    """Load forecasts from intermediate JSON file."""
    print(f"Loading forecasts from {json_file.name}...")
    
    with open(json_file) as f:
        data = json.load(f)
    
    predictions = data.get('predictions', [])
    num_samples = data.get('num_samples', 5)
    
    print(f"  Found {len(predictions)} predictions")
    
    # Group by series_idx
    predictions_by_idx = {}
    for pred in predictions:
        series_idx = pred.get('series_idx')
        if series_idx is not None:
            predictions_by_idx[series_idx] = pred
    
    print(f"  Grouped into {len(predictions_by_idx)} unique series")
    
    # Get test data
    test_data_list = list(dataset.test_data.input)
    total_test = len(test_data_list)
    
    print(f"  Test data has {total_test} entries")
    
    # Check first test entry to see if it's multivariate
    if test_data_list:
        first_target = test_data_list[0]["target"]
        if first_target.ndim > 1:
            num_variates = first_target.shape[0] if first_target.shape[0] < first_target.shape[1] else first_target.shape[1]
            print(f"  ⚠️  Test data is MULTIVARIATE with {num_variates} variates")
            
            # Check if predictions match
            if predictions_by_idx:
                sample_pred = list(predictions_by_idx.values())[0]
                pred_array = np.array(sample_pred['prediction'])
                if pred_array.ndim == 1:
                    print(f"  ❌ ERROR: Predictions are UNIVARIATE but test data has {num_variates} variates!")
                    print(f"     This happens when llm_predictor.py only predicted one variate of a multivariate dataset.")
                    print(f"     Skipping this dataset...")
                    return None
    
    # Build forecasts in order
    forecasts = []
    missing_indices = []
    used_predictions = 0
    
    for idx, entry in enumerate(test_data_list):
        if idx in predictions_by_idx:
            pred_data = predictions_by_idx[idx]
            prediction_array = np.array(pred_data['prediction'])
            
            # Replicate for all samples
            if prediction_array.ndim == 1:
                samples = np.tile(prediction_array, (num_samples, 1))
            else:
                samples = np.tile(prediction_array[np.newaxis, :, :], (num_samples, 1, 1))
            
            # Calculate forecast start date
            historical_data = entry["target"]
            start_date = entry["start"]
            if historical_data.ndim > 1:
                length = historical_data.shape[1] if historical_data.shape[0] < historical_data.shape[1] else historical_data.shape[0]
            else:
                length = len(historical_data)
            forecast_start_date = start_date + length
            
            forecast = SampleForecast(samples=samples, start_date=forecast_start_date)
            forecasts.append(forecast)
            used_predictions += 1
        else:
            # Missing prediction, use zeros
            missing_indices.append(idx)
            historical_data = entry["target"]
            start_date = entry["start"]
            
            if historical_data.ndim > 1:
                num_variates = historical_data.shape[0] if historical_data.shape[0] < historical_data.shape[1] else historical_data.shape[1]
                fallback_shape = (num_samples, dataset.prediction_length, num_variates)
                length = historical_data.shape[1] if historical_data.shape[0] < historical_data.shape[1] else historical_data.shape[0]
            else:
                fallback_shape = (num_samples, dataset.prediction_length)
                length = len(historical_data)
            
            forecast_start_date = start_date + length
            forecast = SampleForecast(samples=np.zeros(fallback_shape), start_date=forecast_start_date)
            forecasts.append(forecast)
    
    if missing_indices:
        print(f"  ⚠️  Warning: {len(missing_indices)} predictions missing (using zeros)")
    
    # Check for extra predictions that were not used
    unused_predictions = len(predictions_by_idx) - used_predictions
    if unused_predictions > 0:
        print(f"  ⚠️  Warning: {unused_predictions} predictions in JSON were not used (idx out of range)")
    
    print(f"  ✅ Built {len(forecasts)} forecasts (matched with test data)")
    
    return forecasts


def calculate_quantile_loss(forecasts, test_entries, quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
    """
    Calculate Mean Weighted Sum Quantile Loss from sample forecasts.
    
    Args:
        forecasts: List of SampleForecast objects
        test_entries: List of test data entries
        quantile_levels: List of quantile levels to evaluate
    
    Returns:
        float: Mean weighted sum quantile loss
    """
    all_quantile_losses = []
    all_weights = []
    
    for forecast, test_entry in zip(forecasts, test_entries):
        # Get samples and target
        samples = forecast.samples  # (num_samples, prediction_length)
        full_target = test_entry['target']
        prediction_length = samples.shape[1]
        
        # Extract target for forecast horizon
        if len(full_target) >= prediction_length:
            target = full_target[-prediction_length:]
        else:
            target = full_target
        
        # Handle multivariate
        if target.ndim > 1:
            target = target.flatten()
        
        # Truncate to match
        min_len = min(len(target), prediction_length)
        target = target[:min_len]
        samples = samples[:, :min_len]
        
        # Skip if target contains NaN
        if np.any(np.isnan(target)):
            continue
        
        # Calculate quantile forecasts from samples
        quantile_forecasts = {}
        for q in quantile_levels:
            quantile_forecasts[q] = np.quantile(samples, q, axis=0)
        
        # Calculate quantile loss for each level
        for q in quantile_levels:
            q_forecast = quantile_forecasts[q]
            errors = target - q_forecast
            
            # Quantile loss: rho_q(error) = q * max(error, 0) + (1-q) * max(-error, 0)
            #              = q * error if error > 0, else (q-1) * error
            loss = np.where(errors >= 0, q * errors, (q - 1) * errors)
            
            # Sum over time steps
            total_loss = np.sum(loss)
            
            # Weight by absolute target sum (for normalization)
            weight = np.sum(np.abs(target))
            
            all_quantile_losses.append(total_loss)
            all_weights.append(weight)
    
    # Calculate weighted mean
    all_quantile_losses = np.array(all_quantile_losses)
    all_weights = np.array(all_weights)
    
    if len(all_weights) > 0 and np.sum(all_weights) > 0:
        # Sum of all quantile losses divided by sum of all weights
        mean_wql = np.sum(all_quantile_losses) / np.sum(all_weights) / len(quantile_levels)
        return float(mean_wql)
    else:
        return np.nan


def calculate_metrics_manually(forecasts, test_entries, season_length):
    """
    Manually calculate comprehensive metrics from forecasts and test entries.
    This avoids the complexity of GluonTS's evaluate_forecasts interface.
    
    Args:
        forecasts: List of SampleForecast objects
        test_entries: List of test data entries (dicts with 'target' key)
        season_length: Seasonal period for MASE calculation
    
    Returns:
        dict: Metrics compatible with evaluate_forecasts output format
    """
    all_errors_mean = []
    all_errors_median = []
    all_abs_errors_mean = []
    all_abs_errors_median = []
    all_targets = []
    all_predictions_mean = []
    all_predictions_median = []
    all_naive_errors = []  # For MASE
    
    print(f"    Calculating for {len(forecasts)} forecasts and {len(test_entries)} test entries...")
    
    for idx, (forecast, test_entry) in enumerate(zip(forecasts, test_entries)):
        # Get forecast samples (shape: num_samples × prediction_length [× num_variates])
        samples = forecast.samples
        prediction_length = samples.shape[1]  # Second dimension is prediction length
        
        # Get true values - extract only the forecast horizon
        # test_entry['target'] contains the full time series, we need the last `prediction_length` points
        full_target = test_entry['target']
        
        # Take the last prediction_length points (the forecast horizon)
        if len(full_target) >= prediction_length:
            target = full_target[-prediction_length:]
        else:
            # If target is shorter than prediction_length, pad with NaN
            target = full_target
        
        if idx == 0:
            print(f"    Debug - Full target shape: {full_target.shape}, samples shape: {samples.shape}")
            print(f"    Debug - Using target[-{prediction_length}:] for comparison")
        
        # Calculate mean and median predictions
        pred_mean = np.mean(samples, axis=0)  # Average across samples
        pred_median = np.median(samples, axis=0)  # Median across samples
        
        # Handle multivariate: flatten if needed
        if target.ndim > 1:
            target = target.flatten()
            pred_mean = pred_mean.flatten()
            pred_median = pred_median.flatten()
        
        # Ensure same length (truncate if needed)
        min_len = min(len(target), len(pred_mean))
        target = target[:min_len]
        pred_mean = pred_mean[:min_len]
        pred_median = pred_median[:min_len]
        
        # Calculate errors
        errors_mean = pred_mean - target
        errors_median = pred_median - target
        abs_errors_mean = np.abs(errors_mean)
        abs_errors_median = np.abs(errors_median)
        
        # Calculate naive forecast error for MASE (seasonal naive)
        # Use the last observed value before the forecast horizon
        if len(full_target) >= prediction_length + season_length:
            naive_forecast = full_target[-(prediction_length + season_length):-season_length]
            naive_errors = np.abs(naive_forecast - target[:len(naive_forecast)])
            all_naive_errors.extend(naive_errors)
        elif idx == 0:
            print(f"    Warning: Cannot calculate MASE - full_target length {len(full_target)} < {prediction_length + season_length}")
        
        all_errors_mean.extend(errors_mean)
        all_errors_median.extend(errors_median)
        all_abs_errors_mean.extend(abs_errors_mean)
        all_abs_errors_median.extend(abs_errors_median)
        all_targets.extend(target)
        all_predictions_mean.extend(pred_mean)
        all_predictions_median.extend(pred_median)
    
    # Convert to numpy arrays
    all_errors_mean = np.array(all_errors_mean)
    all_errors_median = np.array(all_errors_median)
    all_abs_errors_mean = np.array(all_abs_errors_mean)
    all_abs_errors_median = np.array(all_abs_errors_median)
    all_targets = np.array(all_targets)
    all_predictions_mean = np.array(all_predictions_mean)
    all_predictions_median = np.array(all_predictions_median)
    all_naive_errors = np.array(all_naive_errors)
    
    print(f"    Total data points: {len(all_targets)}")
    print(f"    Naive errors collected: {len(all_naive_errors)}")
    print(f"    Target range: [{np.nanmin(all_targets):.2f}, {np.nanmax(all_targets):.2f}]")
    print(f"    Pred range: [{np.min(all_predictions_mean):.2f}, {np.max(all_predictions_mean):.2f}]")
    
    # Check for NaN or Inf
    if np.any(np.isnan(all_targets)) or np.any(np.isnan(all_predictions_mean)):
        print(f"    ⚠️  Warning: Found NaN in data!")
        print(f"       NaN in targets: {np.sum(np.isnan(all_targets))}")
        print(f"       NaN in predictions: {np.sum(np.isnan(all_predictions_mean))}")
        # Remove NaN values
        valid_mask = ~(np.isnan(all_targets) | np.isnan(all_predictions_mean) | np.isnan(all_predictions_median))
        all_targets = all_targets[valid_mask]
        all_predictions_mean = all_predictions_mean[valid_mask]
        all_predictions_median = all_predictions_median[valid_mask]
        all_errors_mean = all_predictions_mean - all_targets
        all_errors_median = all_predictions_median - all_targets
        all_abs_errors_mean = np.abs(all_errors_mean)
        all_abs_errors_median = np.abs(all_errors_median)
        print(f"       Valid data points after filtering: {len(all_targets)}")
    
    # Calculate MSE
    mse_mean = np.mean(all_errors_mean ** 2)
    mse_median = np.mean(all_errors_median ** 2)
    
    # Calculate MAE
    mae_mean = np.mean(all_abs_errors_mean)
    mae_median = np.mean(all_abs_errors_median)
    
    # Calculate RMSE
    rmse_mean = np.sqrt(mse_mean)
    
    # Calculate MASE (Mean Absolute Scaled Error)
    # Filter NaN from naive errors too
    if len(all_naive_errors) > 0:
        all_naive_errors_filtered = all_naive_errors[~np.isnan(all_naive_errors)]
        if len(all_naive_errors_filtered) > 0:
            mae_naive = np.mean(all_naive_errors_filtered)
            print(f"    MASE calculation: MAE_model={mae_median:.4f}, MAE_naive={mae_naive:.4f}")
            if mae_naive > 0:
                mase_median = mae_median / mae_naive
            else:
                mase_median = np.nan
                print(f"    Warning: MAE_naive is zero, cannot calculate MASE")
        else:
            mase_median = np.nan
            print(f"    Warning: All naive errors are NaN, MASE=nan")
    else:
        mase_median = np.nan
        print(f"    Warning: No naive errors collected, MASE=nan (target series too short)")
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    non_zero_mask = all_targets != 0
    if np.sum(non_zero_mask) > 0:
        mape_median = np.mean(np.abs((all_targets[non_zero_mask] - all_predictions_median[non_zero_mask]) / all_targets[non_zero_mask]))
    else:
        mape_median = np.nan
    
    # Calculate sMAPE (Symmetric MAPE)
    denominator = (np.abs(all_targets) + np.abs(all_predictions_median))
    smape_mask = denominator != 0
    if np.sum(smape_mask) > 0:
        smape_median = np.mean(2.0 * np.abs(all_targets[smape_mask] - all_predictions_median[smape_mask]) / denominator[smape_mask])
    else:
        smape_median = np.nan
    
    # Calculate ND (Normalized Deviation)
    sum_abs_errors = np.sum(all_abs_errors_median)
    sum_targets = np.sum(np.abs(all_targets))
    if sum_targets > 0:
        nd_median = sum_abs_errors / sum_targets
    else:
        nd_median = np.nan
    
    # Calculate NRMSE (Normalized RMSE)
    mean_target = np.mean(all_targets)
    if mean_target != 0:
        nrmse_mean = rmse_mean / np.abs(mean_target)
    else:
        nrmse_mean = np.nan
    
    # Calculate Mean Weighted Sum Quantile Loss
    # This requires going back to the original forecasts with samples
    print(f"    Calculating quantile loss...")
    mean_wql = calculate_quantile_loss(forecasts, test_entries)
    print(f"    Mean weighted quantile loss: {mean_wql:.6f}" if not np.isnan(mean_wql) else "    Mean weighted quantile loss: nan")
    
    return {
        'MSE[mean]': float(mse_mean),
        'MSE[0.5]': float(mse_median),
        'MAE[0.5]': float(mae_median),
        'MASE[0.5]': float(mase_median),
        'MAPE[0.5]': float(mape_median),
        'sMAPE[0.5]': float(smape_median),
        'RMSE[mean]': float(rmse_mean),
        'NRMSE[mean]': float(nrmse_mean),
        'ND[0.5]': float(nd_median),
        'MSIS': np.nan,  # Requires specific quantile intervals, skip for now
        'mean_weighted_sum_quantile_loss': mean_wql,
    }


def evaluate_dataset(json_file, dataset, metrics):
    """Evaluate a single dataset from cached predictions."""
    print(f"\n{'='*80}")
    print(f"Evaluating: {json_file.stem.replace('intermediate_', '')}")
    print(f"{'='*80}")
    
    # Load forecasts
    forecasts = load_forecasts_from_json(json_file, dataset)
    
    # Check if loading failed (e.g., dimension mismatch)
    if forecasts is None:
        print(f"\n❌ Failed to load forecasts: {json_file.stem}")
        return None
    
    print(f"\nDataset info:")
    print(f"  Frequency: {dataset.freq}")
    print(f"  Prediction length: {dataset.prediction_length}")
    
    # Get seasonality
    season_length = get_seasonality(dataset.freq)
    
    # Analyze zeros
    print(f"\nAnalyzing forecast quality...")
    total_predictions = len(forecasts)
    zero_forecasts_indices = []
    
    # Debug: Check forecast shapes
    if forecasts:
        print(f"  Debug - First forecast samples shape: {forecasts[0].samples.shape}")
        if len(forecasts) > 1:
            print(f"  Debug - Second forecast samples shape: {forecasts[1].samples.shape}")
    
    for idx, forecast in enumerate(forecasts):
        if np.all(forecast.samples == 0):
            zero_forecasts_indices.append(idx)
    
    num_zeros = len(zero_forecasts_indices)
    num_valid = total_predictions - num_zeros
    success_rate = num_valid / total_predictions if total_predictions > 0 else 0
    
    print(f"  Total: {total_predictions}, Valid: {num_valid} ({success_rate*100:.1f}%), Zeros: {num_zeros}")
    
    # Evaluation 1: All
    print(f"\n  Evaluating all predictions...")
    try:
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
        print(f"  ✅ MSE={res_all['MSE[mean]']:.4f}, MAE={res_all['MAE[0.5]']:.4f}")
        print(f"  📋 All metrics from GluonTS: {list(res_all.keys())}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Evaluation 2: Valid only (manual calculation)
    if num_valid > 0 and num_zeros > 0:
        print(f"\n  Evaluating valid predictions only (manual calculation)...")
        try:
            # Get valid forecasts and test entries
            valid_forecasts = [f for i, f in enumerate(forecasts) if i not in zero_forecasts_indices]
            test_data_list = list(dataset.test_data.input)
            valid_test_entries = [entry for i, entry in enumerate(test_data_list) if i not in zero_forecasts_indices]
            
            # Manual calculation of metrics
            res_valid = calculate_metrics_manually(valid_forecasts, valid_test_entries, season_length)
            print(f"  ✅ MSE={res_valid['MSE[mean]']:.4f}, MAE={res_valid['MAE[0.5]']:.4f}, MASE={res_valid['MASE[0.5]']:.4f}")
        except Exception as e:
            print(f"  ❌ Error in manual calculation: {e}")
            import traceback
            traceback.print_exc()
            res_valid = res_all
    else:
        res_valid = res_all
    
    return {
        'total': total_predictions,
        'valid': num_valid,
        'zeros': num_zeros,
        'rate': success_rate,
        'all': res_all,
        'valid_only': res_valid,
    }


def main(model_name="gpt-5"):
    """Main function."""
    
    print(f"\n{'='*80}")
    print(f"RECOMPUTE METRICS: {model_name}")
    print(f"{'='*80}\n")
    
    results_dir = Path(f"results/{model_name}")
    
    if not results_dir.exists():
        print(f"❌ Not found: {results_dir}")
        return
    
    json_files = sorted(results_dir.glob("intermediate_*.json"))
    
    if not json_files:
        print(f"❌ No intermediate files")
        return
    
    print(f"Found {len(json_files)} files:\n")
    for f in json_files:
        print(f"  • {f.name}")
    
    # Output CSV files - TWO separate files
    csv_file_all = results_dir / "all_results_gluonts.csv"  # GluonTS on all predictions
    csv_file_valid = results_dir / "valid_results_manual.csv"  # Manual calculation on non-zero predictions
    
    print(f"\n📝 Output files:")
    print(f"  1. {csv_file_all} (GluonTS on ALL predictions)")
    print(f"  2. {csv_file_valid} (Manual on valid predictions only)\n")
    
    # Write headers for both files
    header_all = [
        "dataset", "model",
        "total_predictions", "valid_predictions", "zero_predictions", "success_rate",
        "MSE[mean]", "MSE[0.5]", "MAE[0.5]", "MASE[0.5]", "MAPE[0.5]", "sMAPE[0.5]", 
        "RMSE[mean]", "NRMSE[mean]", "ND[0.5]", "mean_wql",
        "domain", "num_variates",
    ]
    
    header_valid = [
        "dataset", "model",
        "total_predictions", "valid_predictions", "zero_predictions", "success_rate",
        "MSE[mean]", "MSE[0.5]", "MAE[0.5]", "MASE[0.5]", "MAPE[0.5]", "sMAPE[0.5]", 
        "RMSE[mean]", "NRMSE[mean]", "ND[0.5]", "mean_wql",
        "domain", "num_variates",
    ]
    
    with open(csv_file_all, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header_all)
    
    with open(csv_file_valid, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header_valid)
    
    # Metrics
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
    
    # Process files
    for json_file in json_files:
        try:
            # Read dataset config from JSON file (more reliable than parsing filename)
            with open(json_file) as f:
                data = json.load(f)
            
            ds_config = data.get('dataset_config')
            if not ds_config:
                print(f"\n⚠️  No dataset_config in {json_file.name}, skipping...")
                continue
            
            # Parse: "kdd_cup_2018/D/short" -> name=kdd_cup_2018/D, term=short
            parts = ds_config.split('/')
            if len(parts) < 2:
                print(f"\n⚠️  Invalid dataset_config format: {ds_config}, skipping...")
                continue
            
            # Last part is term, everything before is the dataset name (which may include frequency)
            term = parts[-1]
            ds_name_with_freq = "/".join(parts[:-1])  # e.g., "kdd_cup_2018/D"
            
            # For display: extract just the base name and freq
            ds_parts = ds_name_with_freq.split('/')
            ds_base = ds_parts[0]  # e.g., "kdd_cup_2018"
            ds_freq = ds_parts[1] if len(ds_parts) > 1 else "?"
            
            print(f"\n{'-'*80}")
            print(f"Loading: {ds_base}, freq={ds_freq}, term={term}")
            print(f"  From: {json_file.name}")
            print(f"  Config: {ds_config}")
            
            # Load dataset
            try:
                # Apply name mapping if needed (map just the base name)
                actual_ds_base = DATASET_NAME_MAP.get(ds_base, ds_base)
                if actual_ds_base != ds_base:
                    print(f"  Name mapping: {ds_base} -> {actual_ds_base}")
                
                # Check if this dataset has no frequency subfolder
                if actual_ds_base in DATASETS_WITHOUT_FREQ_SUBFOLDER:
                    # Dataset has no frequency subfolder, use base name only
                    actual_ds_name = actual_ds_base
                    print(f"  Path rewrite: {ds_name_with_freq} -> {actual_ds_name} (no freq subfolder)")
                else:
                    # Dataset has frequency subfolder, reconstruct full path
                    if len(ds_parts) > 1:
                        actual_ds_name = f"{actual_ds_base}/{ds_freq}"
                    else:
                        actual_ds_name = actual_ds_base
                
                print(f"  Attempting to load dataset: {actual_ds_name}")
                dataset = Dataset(name=actual_ds_name, term=term, to_univariate=False)
                
                # Print dataset info for debugging
                test_list = list(dataset.test_data.input)
                print(f"  ✅ Dataset loaded successfully")
                print(f"     Test entries: {len(test_list)}")
                print(f"     Windows: {dataset.windows}")
                print(f"     Freq: {dataset.freq}")
                
            except FileNotFoundError as e:
                print(f"\n⚠️  Dataset not found: {ds_name}")
                print(f"   Error: {str(e)}")
                print(f"   Skipping this dataset...")
                continue
            except Exception as e:
                print(f"\n❌ Error loading dataset: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Get info (use base name for lookup in properties map)
            ds_key = ds_base.lower()
            domain = dataset_properties_map.get(ds_key, {}).get("domain", "Unknown")
            num_variates = dataset_properties_map.get(ds_key, {}).get("num_variates", 1)
            
            # Evaluate
            result = evaluate_dataset(json_file, dataset, metrics)
            
            if result is None:
                print(f"\n❌ Failed: {ds_config}")
                continue
            
            # Save to BOTH files
            # File 1: All predictions (GluonTS)
            with open(csv_file_all, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    ds_config, model_name,
                    result['total'], result['valid'], result['zeros'], f"{result['rate']:.4f}",
                    result['all'].get("MSE[mean]", np.nan), 
                    result['all'].get("MSE[0.5]", np.nan), 
                    result['all'].get("MAE[0.5]", np.nan), 
                    result['all'].get("MASE[0.5]", np.nan), 
                    result['all'].get("MAPE[0.5]", np.nan), 
                    result['all'].get("sMAPE[0.5]", np.nan),
                    result['all'].get("RMSE[mean]", np.nan), 
                    result['all'].get("NRMSE[mean]", np.nan), 
                    result['all'].get("ND[0.5]", np.nan),
                    result['all'].get("mean_weighted_sum_quantile_loss", np.nan),
                    domain, num_variates,
                ])
            
            # File 2: Valid only predictions (Manual calculation)
            with open(csv_file_valid, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    ds_config, model_name,
                    result['total'], result['valid'], result['zeros'], f"{result['rate']:.4f}",
                    result['valid_only'].get("MSE[mean]", np.nan), 
                    result['valid_only'].get("MSE[0.5]", np.nan), 
                    result['valid_only'].get("MAE[0.5]", np.nan),
                    result['valid_only'].get("MASE[0.5]", np.nan), 
                    result['valid_only'].get("MAPE[0.5]", np.nan), 
                    result['valid_only'].get("sMAPE[0.5]", np.nan),
                    result['valid_only'].get("RMSE[mean]", np.nan), 
                    result['valid_only'].get("NRMSE[mean]", np.nan), 
                    result['valid_only'].get("ND[0.5]", np.nan),
                    result['valid_only'].get("mean_weighted_sum_quantile_loss", np.nan),
                    domain, num_variates,
                ])
            
            print(f"\n✅ Saved: {ds_config}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"✅ Done! Results saved to:")
    print(f"  1. {csv_file_all} (GluonTS on ALL predictions)")
    print(f"  2. {csv_file_valid} (Manual on valid predictions only)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gpt-5"
    main(model_name)
