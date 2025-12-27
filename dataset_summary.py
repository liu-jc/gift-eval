#!/usr/bin/env python3
"""
Script to generate a summary of all datasets in the GIFT-EVAL benchmark.
Provides average context length and prediction length for each dataset.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from gift_eval.data import Dataset
import numpy as np

# Load environment variables
load_dotenv()


def get_all_dataset_names(gift_eval_path):
    """Get all dataset names from the GIFT_EVAL directory."""
    dataset_names = []
    gift_eval_path = Path(gift_eval_path)
    
    for dataset_dir in gift_eval_path.iterdir():
        if dataset_dir.name.startswith("."):
            continue
        if dataset_dir.is_dir():
            freq_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
            if freq_dirs:
                for freq_dir in freq_dirs:
                    dataset_names.append(f"{dataset_dir.name}/{freq_dir.name}")
            else:
                dataset_names.append(dataset_dir.name)
    
    return sorted(dataset_names)


def process_dataset(dataset_name, term="short"):
    """Process a single dataset and return summary statistics."""
    try:
        dataset = Dataset(name=dataset_name, term=term, to_univariate=False)
        
        # Get prediction length
        prediction_length = dataset.prediction_length
        
        # Get frequency
        freq = dataset.freq
        
        # Calculate average context length and count series in one pass
        context_lengths = []
        num_series = 0
        try:
            for data_entry in dataset.training_dataset:
                target = data_entry["target"]
                # Handle both univariate and multivariate cases
                if isinstance(target, np.ndarray):
                    if target.ndim > 1:
                        context_lengths.append(target.shape[-1])
                    else:
                        context_lengths.append(len(target))
                elif hasattr(target, '__len__'):
                    context_lengths.append(len(target))
                
                num_series += 1
                if num_series >= 1000:  # Limit to avoid very long processing
                    break
        except Exception as e:
            print(f"  Warning: Error processing training dataset: {e}", file=sys.stderr)
        
        avg_context_length = float(np.mean(context_lengths)) if context_lengths else None
        
        return {
            "dataset_name": dataset_name,
            "term": term,
            "frequency": freq,
            "prediction_length": prediction_length,
            "avg_context_length": avg_context_length,
            "num_series": num_series,
            "status": "success"
        }
    except Exception as e:
        return {
            "dataset_name": dataset_name,
            "term": term,
            "frequency": None,
            "prediction_length": None,
            "avg_context_length": None,
            "num_series": None,
            "status": f"error: {str(e)}"
        }


def main():
    """Main function to generate dataset summary."""
    # Get GIFT_EVAL path
    gift_eval_path = os.getenv("GIFT_EVAL")
    
    if not gift_eval_path:
        print("Error: GIFT_EVAL environment variable not set.", file=sys.stderr)
        print("Please set GIFT_EVAL in your .env file or environment.", file=sys.stderr)
        sys.exit(1)
    
    # Get all dataset names
    print("Discovering datasets...")
    dataset_names = get_all_dataset_names(gift_eval_path)
    print(f"Found {len(dataset_names)} datasets\n")
    # short datasets and med_long datasets used in the GIFT-EVAL benchmark
    short_datasets = "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
    med_long_datasets = "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
    # med_long_datasets = "bizitobs_l2c/H"
    # Process each dataset with all terms
    results = []
    terms = ["short", "medium", "long"]
    # total_tasks = len(dataset_names) * len(terms)
    task_count = 0
    
    print("Processing datasets with all terms (this may take a while)...")
    print("-" * 100)
    
    for i, dataset_name in enumerate(dataset_names, 1):
        print(f"[Dataset {i}/{len(dataset_names)}] Processing: {dataset_name}")
        
        for term in terms:
            if (
                term == "medium" or term == "long"
            ) and dataset_name not in med_long_datasets.split():
                continue
            task_count += 1
            # print(f"  [{task_count}/{total_tasks}] Term: {term}")
            print(f"  Term: {term}")
            result = process_dataset(dataset_name, term=term)
            results.append(result)
            
            if result["status"] == "success":
                print(f"    ✓ Frequency: {result['frequency']}")
                print(f"    ✓ Prediction length: {result['prediction_length']}")
                if result["avg_context_length"] is not None:
                    print(f"    ✓ Avg context length: {result['avg_context_length']:.2f}")
                print(f"    ✓ Number of series (sampled): {result['num_series']}")
            else:
                # Only print error if it's not a "term not supported" type error
                error_msg = result['status']
                if "error" in error_msg.lower():
                    print(f"    ✗ {error_msg}")
        print()
    
    # Generate summary
    print("=" * 100)
    print("DATASET SUMMARY")
    print("=" * 100)
    print()
    
    # Filter successful results
    successful_results = [r for r in results if r["status"] == "success"]
    failed_results = [r for r in results if r["status"] != "success"]
    
    if successful_results:
        unique_datasets = len(set(r["dataset_name"] for r in successful_results))
        print(f"Successfully processed: {len(successful_results)} dataset-term combinations "
              f"({unique_datasets} unique datasets)")
        print()
        
        # Calculate overall statistics
        all_context_lengths = [r["avg_context_length"] for r in successful_results 
                              if r["avg_context_length"] is not None]
        all_prediction_lengths = [r["prediction_length"] for r in successful_results 
                                 if r["prediction_length"] is not None]
        all_num_series = [r["num_series"] for r in successful_results 
                         if r["num_series"] is not None and r["num_series"] > 0]
        
        print(f"Overall Statistics:")
        if all_context_lengths:
            print(f"  Average context length (across datasets): {np.mean(all_context_lengths):.2f}")
            print(f"  Median context length: {np.median(all_context_lengths):.2f}")
            print(f"  Min context length: {np.min(all_context_lengths):.2f}")
            print(f"  Max context length: {np.max(all_context_lengths):.2f}")
            print()
        
        if all_prediction_lengths:
            print(f"  Average prediction length: {np.mean(all_prediction_lengths):.2f}")
            print(f"  Median prediction length: {np.median(all_prediction_lengths):.2f}")
            print(f"  Min prediction length: {np.min(all_prediction_lengths)}")
            print(f"  Max prediction length: {np.max(all_prediction_lengths)}")
            print()
        
        if all_num_series:
            total_series = sum(all_num_series)
            avg_series = np.mean(all_num_series)
            print(f"  Average number of series (per dataset-term): {avg_series:.2f}")
            print(f"  Total number of series (across all dataset-terms): {total_series:,}")
            print()
        
        # Print detailed table
        print("Detailed Dataset Information:")
        print("-" * 120)
        print(f"{'Dataset Name':<40} {'Term':<8} {'Freq':<8} {'Pred Len':<10} {'Avg Ctx Len':<15} {'Num Series':<12}")
        print("-" * 120)
        
        # Sort by dataset name, then by term
        sorted_results = sorted(successful_results, key=lambda x: (x["dataset_name"], x["term"]))
        
        for result in sorted_results:
            dataset_name = result["dataset_name"]
            term = result["term"]
            freq = result["frequency"] or "N/A"
            pred_len = result["prediction_length"] or "N/A"
            ctx_len = f"{result['avg_context_length']:.2f}" if result["avg_context_length"] is not None else "N/A"
            num_series = result["num_series"] or "N/A"
            
            print(f"{dataset_name:<40} {term:<8} {str(freq):<8} {str(pred_len):<10} {ctx_len:<15} {str(num_series):<12}")
        
        print()
        
        # Print summary by term
        print("Summary by Term:")
        print("-" * 120)
        for term in terms:
            term_results = [r for r in successful_results if r["term"] == term]
            if term_results:
                term_context_lengths = [r["avg_context_length"] for r in term_results 
                                        if r["avg_context_length"] is not None]
                term_prediction_lengths = [r["prediction_length"] for r in term_results 
                                          if r["prediction_length"] is not None]
                
                term_num_series = [r["num_series"] for r in term_results 
                                  if r["num_series"] is not None and r["num_series"] > 0]
                
                print(f"\n{term.upper()} term ({len(term_results)} datasets):")
                if term_context_lengths:
                    print(f"  Avg context length: {np.mean(term_context_lengths):.2f} "
                          f"(min: {np.min(term_context_lengths):.2f}, "
                          f"max: {np.max(term_context_lengths):.2f})")
                if term_prediction_lengths:
                    print(f"  Avg prediction length: {np.mean(term_prediction_lengths):.2f} "
                          f"(min: {np.min(term_prediction_lengths)}, "
                          f"max: {np.max(term_prediction_lengths)})")
                if term_num_series:
                    term_total_series = sum(term_num_series)
                    term_avg_series = np.mean(term_num_series)
                    print(f"  Avg number of series: {term_avg_series:.2f}")
                    print(f"  Total number of series: {term_total_series:,}")
        print()
    
    if failed_results:
        print(f"Failed to process: {len(failed_results)} dataset-term combinations")
        print("-" * 100)
        for result in failed_results:
            print(f"  {result['dataset_name']} ({result['term']}): {result['status']}")
        print()
    
    # Save to CSV
    import csv
    csv_filename = "dataset_summary.csv"
    print(f"Saving detailed results to {csv_filename}...")
    
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['dataset_name', 'term', 'frequency', 'prediction_length', 
                     'avg_context_length', 'num_series', 'status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Summary saved to {csv_filename}")


if __name__ == "__main__":
    main()

