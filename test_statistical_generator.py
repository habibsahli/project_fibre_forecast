#!/usr/bin/env python3
"""
Test script for Statistical Generative Model on fibre subscription data.

This script demonstrates:
1. Loading fibre subscription data from database
2. Training statistical generative model
3. Generating synthetic scenarios
4. Evaluating synthetic data quality
5. Visualizing results
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.generative_models import StatisticalTimeSeriesGenerator
from src.forecasting.data_loader import load_daily_data, get_engine

def test_statistical_generator_basic():
    """Basic test with synthetic fibre data."""
    print("🧪 Testing Statistical Generator with synthetic data...")

    # Create synthetic fibre data for testing
    dates = pd.date_range('2024-01-01', periods=365, freq='D')

    # Simulate realistic fibre subscription patterns
    # Weekly pattern (higher on weekdays)
    weekly_pattern = np.sin(2 * np.pi * np.arange(365) / 7) * 300 + 2800

    # Monthly pattern (end of month peaks)
    monthly_pattern = np.sin(2 * np.pi * np.arange(365) / 30) * 200

    # Overall trend (growing subscriptions)
    trend = np.linspace(2500, 3200, 365)

    # Random noise + occasional spikes
    noise = np.random.normal(0, 150, 365)
    spikes = np.random.choice([0, 500], 365, p=[0.95, 0.05])  # 5% chance of spike

    subscriptions = weekly_pattern + monthly_pattern + trend + noise + spikes

    df_test = pd.DataFrame({
        'date': dates,
        'nb_abonnements': subscriptions.astype(int)
    })

    # Train statistical generator
    model = StatisticalTimeSeriesGenerator(seasonal_periods=[7, 30], noise_factor=0.1)
    model.fit(df_test)

    # Generate synthetic scenarios
    scenarios = model.generate_scenarios(n_scenarios=5, forecast_horizon=30)
    print(f"✅ Generated {len(scenarios)} synthetic scenarios")

    # Plot results
    plt.figure(figsize=(14, 8))

    # Plot historical data
    plt.subplot(2, 1, 1)
    plt.plot(df_test['date'], df_test['nb_abonnements'],
             label='Real Historical Data', color='blue', alpha=0.7)
    plt.title('Statistical Generator: Historical Fibre Subscription Data')
    plt.xlabel('Date')
    plt.ylabel('Daily Subscriptions')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot scenarios
    plt.subplot(2, 1, 2)
    last_date = df_test['date'].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=30)

    plt.plot(df_test['date'], df_test['nb_abonnements'],
             label='Historical', color='blue', alpha=0.5)

    colors = plt.cm.viridis(np.linspace(0, 1, len(scenarios)))
    for i, (idx, row) in enumerate(scenarios.iterrows()):
        plt.plot(future_dates, row['forecast'],
                color=colors[i], linewidth=2, alpha=0.8,
                label=f'Scenario {i+1}')

    plt.title('Statistical Generator: Fibre Subscription Forecast Scenarios')
    plt.xlabel('Date')
    plt.ylabel('Daily Subscriptions')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/statistical_generator_test.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("📊 Plot saved to outputs/statistical_generator_test.png")
    return model, df_test

def test_statistical_generator_real_data():
    """Test with real fibre data from database."""
    print("🔗 Testing Statistical Generator with real fibre data...")

    try:
        # Load real data
        engine = get_engine()
        df_daily = load_daily_data(engine)

        if len(df_daily) < 60:
            print("⚠️  Not enough data for statistical modeling (need at least 60 days)")
            return None, None

        print(f"📊 Loaded {len(df_daily)} days of real fibre subscription data")
        print(f"   Date range: {df_daily['date'].min()} to {df_daily['date'].max()}")
        print(f"   Mean daily subscriptions: {df_daily['nb_abonnements'].mean():.0f}")
        print(f"   Total subscriptions: {df_daily['nb_abonnements'].sum():,}")

        # Train statistical generator
        model = StatisticalTimeSeriesGenerator(seasonal_periods=[7, 30], noise_factor=0.15)
        model.fit(df_daily)

        # Evaluate synthetic data quality
        quality_metrics = model.evaluate_synthetic_quality(df_daily, n_samples=50)
        print("\n📊 Synthetic Data Quality Assessment:")
        print(f"   - Mean difference: {quality_metrics['mean_rel_diff']:.3f}")
        print(f"   - Std difference: {quality_metrics['std_rel_diff']:.3f}")
        print(f"   - Min difference: {quality_metrics['min_rel_diff']:.3f}")
        print(f"   - Max difference: {quality_metrics['max_rel_diff']:.3f}")
        print(f"   - Trend difference: {quality_metrics['trend_rel_diff']:.3f}")
        # Generate scenarios
        scenarios = model.generate_scenarios(n_scenarios=10, forecast_horizon=30)
        print(f"✅ Generated {len(scenarios)} forecast scenarios")

        # Plot real data + scenarios
        plt.figure(figsize=(16, 10))

        # Historical data plot
        plt.subplot(2, 2, 1)
        plt.plot(df_daily['date'], df_daily['nb_abonnements'],
                color='blue', linewidth=1.5)
        plt.title('Historical Fibre Subscription Data')
        plt.xlabel('Date')
        plt.ylabel('Daily Subscriptions')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # Scenario forecast plot
        plt.subplot(2, 2, 2)
        last_date = df_daily['date'].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=30)

        # Plot last 90 days of history for context
        recent_data = df_daily.tail(90)
        plt.plot(recent_data['date'], recent_data['nb_abonnements'],
                label='Recent History', color='blue', linewidth=2)

        # Plot scenarios
        colors = plt.cm.plasma(np.linspace(0, 1, len(scenarios)))
        for i, (idx, row) in enumerate(scenarios.iterrows()):
            plt.plot(future_dates, row['forecast'],
                    color=colors[i], alpha=0.7, linewidth=1.5,
                    label=f'Scenario {i+1}')

        plt.title('Statistical Generator: 30-Day Forecast Scenarios')
        plt.xlabel('Date')
        plt.ylabel('Daily Subscriptions')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # Scenario distribution plot
        plt.subplot(2, 2, 3)
        scenario_means = [row['forecast'].mean() for idx, row in scenarios.iterrows()]
        plt.hist(scenario_means, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(df_daily['nb_abonnements'].tail(30).mean(),
                   color='red', linestyle='--', linewidth=2,
                   label='Recent Average')
        plt.title('Distribution of Scenario Means')
        plt.xlabel('Average Daily Subscriptions')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Scenario range plot
        plt.subplot(2, 2, 4)
        scenario_ranges = [(row['forecast'].max() - row['forecast'].min())
                          for idx, row in scenarios.iterrows()]
        plt.hist(scenario_ranges, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.title('Distribution of Scenario Ranges')
        plt.xlabel('Min-Max Range')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        output_dir = Path('outputs')
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / 'statistical_generator_real_data.png',
                   dpi=150, bbox_inches='tight')
        plt.show()

        print("📊 Plot saved to outputs/statistical_generator_real_data.png")

        # Save scenarios to CSV
        scenarios_df = pd.DataFrame()
        for i, (idx, row) in enumerate(scenarios.iterrows()):
            temp_df = pd.DataFrame({
                'date': future_dates,
                'scenario': f'scenario_{i+1}',
                'forecast': row['forecast']
            })
            scenarios_df = pd.concat([scenarios_df, temp_df], ignore_index=True)

        scenarios_df.to_csv(output_dir / 'statistical_scenarios.csv', index=False)
        print("💾 Scenarios saved to outputs/statistical_scenarios.csv")

        # Save quality metrics
        quality_df = pd.DataFrame([quality_metrics])
        quality_df.to_csv(output_dir / 'synthetic_quality_metrics.csv', index=False)
        print("📈 Quality metrics saved to outputs/synthetic_quality_metrics.csv")

        return model, df_daily

    except Exception as e:
        print(f"❌ Error testing with real data: {e}")
        print("💡 Make sure PostgreSQL is running and database is populated")
        return None, None

def analyze_scenarios(scenarios_df):
    """Analyze generated scenarios for business insights."""
    print("\n📈 Scenario Analysis for Business Planning:")

    # Calculate statistics for each scenario
    scenario_stats = []
    for scenario in scenarios_df['scenario'].unique():
        scenario_data = scenarios_df[scenarios_df['scenario'] == scenario]['forecast']
        stats = {
            'scenario': scenario,
            'mean_daily': scenario_data.mean(),
            'total_30_days': scenario_data.sum(),
            'min_daily': scenario_data.min(),
            'max_daily': scenario_data.max(),
            'volatility': scenario_data.std(),
            'trend': scenario_data.iloc[-1] - scenario_data.iloc[0]
        }
        scenario_stats.append(stats)

    stats_df = pd.DataFrame(scenario_stats)
    print("\nScenario Statistics (30-day forecast):")
    print(stats_df.round(2).to_string(index=False))

    # Business insights
    print("\n💼 Business Insights:")
    print(f"  📊 Average scenario: {stats_df['mean_daily'].mean():.0f} subscriptions/day")
    print(f"  📈 Total range: {stats_df['total_30_days'].min():,.0f} - {stats_df['total_30_days'].max():,.0f} subscriptions")
    print(f"  🎯 Best case: {stats_df.loc[stats_df['total_30_days'].idxmax(), 'scenario']} (+{stats_df['total_30_days'].max() - stats_df['total_30_days'].mean():,.0f})")
    print(f"  ⚠️  Worst case: {stats_df.loc[stats_df['total_30_days'].idxmin(), 'scenario']} ({stats_df['total_30_days'].min() - stats_df['total_30_days'].mean():,.0f})")
    print(f"  📊 Volatility range: {stats_df['volatility'].min():.1f} - {stats_df['volatility'].max():.1f}")
    return stats_df

def main():
    """Main test function."""
    print("📊 Statistical Generative Model Test for Fibre Subscriptions")
    print("=" * 70)

    # Test 1: Basic functionality with synthetic data
    print("\n1️⃣ Testing with synthetic data...")
    model_synthetic, df_synthetic = test_statistical_generator_basic()

    # Test 2: Real data (if available)
    print("\n2️⃣ Testing with real fibre data...")
    model_real, df_real = test_statistical_generator_real_data()

    if model_real is not None and df_real is not None:
        # Load scenarios for analysis
        scenarios_df = pd.read_csv('outputs/statistical_scenarios.csv')
        analyze_scenarios(scenarios_df)

        # Load quality metrics
        quality_df = pd.read_csv('outputs/synthetic_quality_metrics.csv')
        print("\n🔍 Synthetic Data Quality:")
        for col in ['mean_rel_diff', 'std_rel_diff', 'trend_rel_diff']:
            if col in quality_df.columns:
                value = quality_df[col].iloc[0]
                print(f"  {col}: {value:.1%}")

    print("\n✅ Statistical Generator testing completed!")
    print("\n🚀 Next steps:")
    print("   - Use scenarios for demand planning and inventory management")
    print("   - Integrate synthetic data into forecasting model training")
    print("   - Implement uncertainty-aware business decisions")
    print("   - Consider upgrading to TimeGAN for more advanced generation")

if __name__ == "__main__":
    main()