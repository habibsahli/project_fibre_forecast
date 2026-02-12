#!/usr/bin/env python3
"""
LSTM vs XGBoost Model Comparison
Direct head-to-head comparison of the two best performing models
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def load_latest_results():
    """Load the most recent results for LSTM and XGBoost"""
    results_dir = Path("outputs/results")

    # Find latest LSTM results
    lstm_files = list(results_dir.glob("lstm_results_*.json"))
    lstm_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    lstm_file = lstm_files[0] if lstm_files else None

    # Find latest XGBoost results
    xgb_files = list(results_dir.glob("xgboost_results_*.json"))
    xgb_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    xgb_file = xgb_files[0] if xgb_files else None

    results = {}

    if lstm_file:
        with open(lstm_file, 'r') as f:
            results['LSTM'] = json.load(f)

    if xgb_file:
        with open(xgb_file, 'r') as f:
            results['XGBoost'] = json.load(f)

    return results

def create_comparison_table(results):
    """Create a comparison table of model metrics"""

    comparison_data = []

    for model_name, result in results.items():
        metrics = result['metrics']
        row = {
            'Model': model_name,
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'MAPE': metrics['MAPE'],
            'sMAPE': metrics['sMAPE'],
            'Training_Time': metrics['training_time']
        }
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)

    # Find best values for each metric (lower is better for all metrics)
    best_mae = df['MAE'].min()
    best_rmse = df['RMSE'].min()
    best_mape = df['MAPE'].min()
    best_smape = df['sMAPE'].min()
    best_time = df['Training_Time'].min()

    # Add ranking column
    df['Overall_Rank'] = df.apply(lambda row:
        (row['MAE'] == best_mae) * 4 +
        (row['RMSE'] == best_rmse) * 3 +
        (row['sMAPE'] == best_smape) * 2 +
        (row['Training_Time'] == best_time) * 1, axis=1)

    df = df.sort_values('Overall_Rank', ascending=False)

    return df

def print_comparison(results, df):
    """Print detailed comparison"""

    print("\n" + "="*80)
    print("🤖 LSTM vs XGBoost - HEAD-TO-HEAD COMPARISON")
    print("="*80)
    print(f"Comparison Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data Points: {results['LSTM']['data_points']['train_size'] + results['LSTM']['data_points']['test_size']} total days")
    print(f"Test Set: {results['LSTM']['data_points']['test_size']} days")

    print("\n" + "="*60)
    print("📊 PERFORMANCE METRICS COMPARISON")
    print("="*60)

    # Print table
    print(df.to_string(index=False, float_format='%.3f'))

    print("\n" + "="*60)
    print("🏆 WINNER DETERMINATION")
    print("="*60)

    winner = df.iloc[0]['Model']
    print(f"🥇 **WINNER: {winner}**")

    # Detailed analysis
    lstm_metrics = results['LSTM']['metrics']
    xgb_metrics = results['XGBoost']['metrics']

    print("\n📈 Detailed Analysis:")
    print(f"   MAE:  LSTM ({lstm_metrics['MAE']:.3f}) vs XGBoost ({xgb_metrics['MAE']:.3f})")
    print(f"   RMSE: LSTM ({lstm_metrics['RMSE']:.3f}) vs XGBoost ({xgb_metrics['RMSE']:.3f})")
    print(f"   sMAPE: LSTM ({lstm_metrics['sMAPE']:.2f}%) vs XGBoost ({xgb_metrics['sMAPE']:.2f}%)")
    print(f"   Time: LSTM ({lstm_metrics['training_time']:.2f}s) vs XGBoost ({xgb_metrics['training_time']:.2f}s)")

    # Calculate improvements
    mae_improvement = ((lstm_metrics['MAE'] - xgb_metrics['MAE']) / lstm_metrics['MAE']) * 100
    rmse_improvement = ((lstm_metrics['RMSE'] - xgb_metrics['RMSE']) / lstm_metrics['RMSE']) * 100
    smape_improvement = ((lstm_metrics['sMAPE'] - xgb_metrics['sMAPE']) / lstm_metrics['sMAPE']) * 100
    time_ratio = lstm_metrics['training_time'] / xgb_metrics['training_time']

    if winner == 'XGBoost':
        print("\n💪 XGBoost Advantages:")
        print(f"   • {abs(mae_improvement):.1f}% better MAE accuracy")
        print(f"   • {abs(rmse_improvement):.1f}% better RMSE accuracy")
        print(f"   • {abs(smape_improvement):.1f}% better sMAPE accuracy")
        print(f"   • {time_ratio:.1f}x faster training")
    else:
        print("\n🧠 LSTM Advantages:")
        print(f"   • {abs(mae_improvement):.1f}% better MAE accuracy")
        print(f"   • {abs(rmse_improvement):.1f}% better RMSE accuracy")
        print(f"   • {abs(smape_improvement):.1f}% better sMAPE accuracy")
        print(f"   • {time_ratio:.1f}x faster training")
    print("\n" + "="*60)
    print("💡 RECOMMENDATIONS")
    print("="*60)

    if winner == 'XGBoost':
        print("✅ Use XGBoost for production forecasting")
        print("   • Superior accuracy with faster training")
        print("   • Excellent feature importance insights")
        print("   • Robust to different data patterns")
        print("   • Consider LSTM for specialized time-series needs")
    else:
        print("✅ Use LSTM for production forecasting")
        print("   • Better time-series pattern recognition")
        print("   • Superior long-term dependency modeling")
        print("   • Consider XGBoost for faster inference needs")

    print("\n🔄 Next Steps:")
    print("   • Integrate winner into forecasting pipeline")
    print("   • Use generative models for uncertainty quantification")
    print("   • Consider ensemble of both models for best results")

def main():
    """Main comparison function"""
    print("Loading latest model results...")

    results = load_latest_results()

    if not results:
        print("❌ No model results found!")
        return

    if len(results) < 2:
        print(f"⚠️  Only {len(results)} model(s) found. Need both LSTM and XGBoost for comparison.")
        return

    df = create_comparison_table(results)
    print_comparison(results, df)

    # Save comparison
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path("outputs/results") / f"lstm_xgboost_comparison_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 Comparison saved to: {output_file}")

if __name__ == "__main__":
    main()