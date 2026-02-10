#!/usr/bin/env python3
"""
Master Model Comparison Runner
Runs all 4 models and generates a comparison report
Run: python run_all_models.py
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

OUTPUT_DIR = Path("outputs/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_all_models():
    """Run all individual model scripts and aggregate results"""
    
    print("\n" + "="*70)
    print("🎯 FIBRE FORECAST - COMPLETE MODEL COMPARISON")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_start = time.time()
    
    models = [
        ('run_prophet_model.py', 'Prophet'),
        ('run_sarima_model.py', 'SARIMA'),
        ('run_xgboost_model.py', 'XGBoost'),
        ('run_expsmoothing_model.py', 'Exponential Smoothing'),
    ]
    
    results = {}
    failed = []
    
    for script, name in models:
        print(f"\n{'─'*70}")
        print(f"Running {name}...")
        try:
            result = subprocess.run(
                [sys.executable, script],
                capture_output=False,
                timeout=300
            )
            if result.returncode == 0:
                print(f"✅ {name} completed successfully")
            else:
                print(f"❌ {name} failed with return code {result.returncode}")
                failed.append(name)
        except subprocess.TimeoutExpired:
            print(f"⏱️  {name} timed out")
            failed.append(name)
        except Exception as e:
            print(f"❌ {name} error: {e}")
            failed.append(name)
    
    # Aggregate results from JSON files
    print("\n" + "="*70)
    print("📊 AGGREGATING RESULTS")
    print("="*70)
    
    aggregated = {}
    
    for script, name in models:
        # Find the latest result file for this model
        pattern = name.lower().replace(' ', '')
        result_files = list(OUTPUT_DIR.glob(f"*{pattern.lower()}*results*.json"))
        
        if result_files:
            latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
            try:
                with open(latest_file, 'r') as f:
                    data = json.load(f)
                    aggregated[name] = data['metrics']
                    print(f"✅ Loaded results for {name}")
            except Exception as e:
                print(f"❌ Failed to load {name}: {e}")
        else:
            print(f"⚠️  No results file found for {name}")
    
    # Create comparison table
    if aggregated:
        print("\n" + "="*70)
        print("🏆 FINAL RANKINGS")
        print("="*70)
        
        df_results = pd.DataFrame(aggregated).T[['MAE', 'RMSE', 'MAPE', 'sMAPE', 'training_time', 'total_time']]
        df_results = df_results.sort_values('MAPE')
        
        print("\nRanked by MAPE (lower is better):")
        print("─" * 100)
        print(f"{'Rank':<6} {'Model':<25} {'MAPE %':<12} {'MAE':<12} {'RMSE':<12} {'Time (s)':<12}")
        print("─" * 100)
        
        for rank, (model_name, metrics) in enumerate(df_results.iterrows(), 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            print(f"{medal} {rank:<4} {model_name:<25} {metrics['MAPE']:>10.2f}% {metrics['MAE']:>10.2f} {metrics['RMSE']:>10.2f} {metrics['total_time']:>10.2f}s")
        
        # Save comparison
        comparison_file = OUTPUT_DIR / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_results.to_csv(comparison_file)
        print(f"\n📊 Detailed comparison saved to: {comparison_file}")
        
        # Save summary JSON
        summary_file = OUTPUT_DIR / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_runtime': time.time() - all_start,
                'models_tested': len(aggregated),
                'models_failed': len(failed),
                'failed_models': failed,
                'ranking': df_results.to_dict(),
                'best_model': df_results.index[0],
                'best_mape': float(df_results.iloc[0]['MAPE'])
            }, f, indent=2)
        
        print(f"✅ Summary saved to: {summary_file}")
    
    print("\n" + "="*70)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Runtime: {time.time() - all_start:.2f}s")
    print("="*70)
    
    if failed:
        print(f"\n⚠️  Failed models: {', '.join(failed)}")
    else:
        print("\n✅ All models completed successfully!")

if __name__ == "__main__":
    run_all_models()
