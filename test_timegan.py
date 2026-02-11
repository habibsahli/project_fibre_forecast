#!/usr/bin/env python3
"""
Test script for TimeGAN generative model on fibre subscription data.

This script demonstrates:
1. Loading fibre subscription data from database
2. Training TimeGAN model
3. Generating synthetic scenarios
4. Visualizing results
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from generative_models import TimeGANModel
from forecasting.data_loader import load_daily_data, get_engine

def test_timegan_basic():
    """Basic TimeGAN test with synthetic data."""
    print("🧪 Testing TimeGAN with synthetic data...")

    # Create synthetic fibre data for testing
    dates = pd.date_range('2024-01-01', periods=365, freq='D')
    # Simulate weekly pattern + trend + noise
    weekly_pattern = np.sin(2 * np.pi * np.arange(365) / 7) * 500
    trend = np.linspace(2000, 3000, 365)
    noise = np.random.normal(0, 200, 365)
    subscriptions = weekly_pattern + trend + noise + 2500

    df_test = pd.DataFrame({
        'date': dates,
        'nb_abonnements': subscriptions.astype(int)
    })

    # Train TimeGAN
    model = TimeGANModel(seq_len=30, epochs=20)  # Short training for demo
    model.fit(df_test)

    # Generate synthetic scenarios
    scenarios = model.generate_scenarios(n_scenarios=5, forecast_horizon=30)
    print(f"✅ Generated {len(scenarios)} synthetic scenarios")

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(df_test['date'], df_test['nb_abonnements'], label='Real Data', alpha=0.7)

    for i, row in scenarios.iterrows():
        plt.plot(pd.date_range(df_test['date'].max() + pd.Timedelta(days=1), periods=30),
                row['forecast'], alpha=0.6, label=f'Scenario {i+1}')

    plt.title('TimeGAN: Real vs Synthetic Fibre Subscription Scenarios')
    plt.xlabel('Date')
    plt.ylabel('Daily Subscriptions')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/timegan_test_scenarios.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("📊 Plot saved to outputs/timegan_test_scenarios.png")
    return model

def test_timegan_real_data():
    """Test TimeGAN with real fibre data from database."""
    print("🔗 Testing TimeGAN with real fibre data...")

    try:
        # Load real data
        engine = get_engine()
        df_daily = load_daily_data(engine)

        if len(df_daily) < 60:
            print("⚠️  Not enough data for TimeGAN (need at least 60 days)")
            return None

        print(f"📊 Loaded {len(df_daily)} days of real fibre subscription data")

        # Train TimeGAN
        model = TimeGANModel(seq_len=30, epochs=50)
        model.fit(df_daily)

        # Generate scenarios
        scenarios = model.generate_scenarios(n_scenarios=10, forecast_horizon=30)
        print(f"✅ Generated {len(scenarios)} scenarios from real data")

        # Plot real data + scenarios
        plt.figure(figsize=(14, 7))

        # Plot historical data
        plt.plot(df_daily['date'], df_daily['nb_abonnements'],
                label='Historical Data', color='blue', linewidth=2)

        # Plot scenarios
        last_date = df_daily['date'].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=30)

        colors = plt.cm.viridis(np.linspace(0, 1, len(scenarios)))
        for i, (idx, row) in enumerate(scenarios.iterrows()):
            plt.plot(future_dates, row['forecast'],
                    color=colors[i], alpha=0.7,
                    label=f'Scenario {i+1}')

        plt.title('TimeGAN: Fibre Subscription Forecast Scenarios', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Daily Subscriptions', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        output_dir = Path('outputs')
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / 'timegan_real_data_scenarios.png',
                   dpi=150, bbox_inches='tight')
        plt.show()

        print("📊 Plot saved to outputs/timegan_real_data_scenarios.png")

        # Save scenarios to CSV
        scenarios_df = pd.DataFrame()
        for i, (idx, row) in enumerate(scenarios.iterrows()):
            temp_df = pd.DataFrame({
                'date': future_dates,
                'scenario': f'scenario_{i+1}',
                'forecast': row['forecast']
            })
            scenarios_df = pd.concat([scenarios_df, temp_df], ignore_index=True)

        scenarios_df.to_csv(output_dir / 'timegan_scenarios.csv', index=False)
        print("💾 Scenarios saved to outputs/timegan_scenarios.csv")

        return model

    except Exception as e:
        print(f"❌ Error testing with real data: {e}")
        print("💡 Make sure PostgreSQL is running and database is populated")
        return None

def analyze_scenarios(scenarios_df):
    """Analyze generated scenarios for insights."""
    print("\n📈 Scenario Analysis:")

    # Calculate statistics for each scenario
    scenario_stats = []
    for scenario in scenarios_df['scenario'].unique():
        scenario_data = scenarios_df[scenarios_df['scenario'] == scenario]['forecast']
        stats = {
            'scenario': scenario,
            'mean': scenario_data.mean(),
            'std': scenario_data.std(),
            'min': scenario_data.min(),
            'max': scenario_data.max(),
            'trend': scenario_data.iloc[-1] - scenario_data.iloc[0]  # End - start
        }
        scenario_stats.append(stats)

    stats_df = pd.DataFrame(scenario_stats)
    print(stats_df.round(2))

    # Overall statistics
    print("
📊 Overall Statistics:"    print(f"  Average daily subscriptions: {stats_df['mean'].mean():.0f}")
    print(f"  Scenario variability (std): {stats_df['mean'].std():.0f}")
    print(f"  Most optimistic scenario: {stats_df.loc[stats_df['mean'].idxmax(), 'scenario']}")
    print(f"  Most pessimistic scenario: {stats_df.loc[stats_df['mean'].idxmin(), 'scenario']}")

    return stats_df

def main():
    """Main test function."""
    print("🎭 TimeGAN Generative Model Test for Fibre Subscriptions")
    print("=" * 60)

    # Test 1: Basic functionality with synthetic data
    print("\n1️⃣ Testing with synthetic data...")
    model_synthetic = test_timegan_basic()

    # Test 2: Real data (if available)
    print("\n2️⃣ Testing with real fibre data...")
    model_real = test_timegan_real_data()

    if model_real is not None:
        # Load scenarios for analysis
        scenarios_df = pd.read_csv('outputs/timegan_scenarios.csv')
        analyze_scenarios(scenarios_df)

    print("\n✅ TimeGAN testing completed!")
    print("\n💡 Next steps:")
    print("   - Integrate synthetic data into forecasting pipeline")
    print("   - Use scenarios for uncertainty quantification")
    print("   - Tune hyperparameters for better performance")

if __name__ == "__main__":
    main()