"""
Q3: Forecasting using Single Exponential Smoothing (SES)
Forecast the loss for 2016 and calculate expected loss (mean), standard deviation, and variance.

Formula: Ft+1 = Ft + alpha * (At - Ft)
Initial forecast F(2011) = 75 million
Smoothing constant alpha = 0.5 (commonly used default)
"""

import numpy as np

# Given data
years = [2011, 2012, 2013, 2014, 2015]
actual_losses = [76, 75, 74, 79, 78]  # USD Millions
alpha = 0.5  # Smoothing constant
initial_forecast = 75  # Initial loss forecast in millions

print("=" * 80)
print("Q3: FORECASTING AND RISK MANAGEMENT")
print("=" * 80)

# Part 1: Descriptive Answer
print("""
FORECASTING IN RISK MANAGEMENT
-------------------------------
Forecasting is the process of making predictions about future events based on
historical data and analysis. It uses statistical techniques, mathematical models,
and informed judgment to estimate future trends, values, or outcomes.

IMPORTANCE IN RISK MANAGEMENT:
1. Early Warning: Forecasting helps identify potential risks before they materialize,
   allowing organizations to take proactive measures.
2. Capital Allocation: Accurate loss forecasting helps banks and financial institutions
   allocate adequate capital reserves (as required by Basel accords).
3. Strategic Planning: Enables organizations to plan budgets, resources, and
   contingency measures based on expected future losses.
4. Regulatory Compliance: Financial regulators require institutions to forecast
   potential losses for stress testing and capital adequacy assessments.
5. Performance Monitoring: Comparing actual vs. forecasted values helps evaluate
   the effectiveness of risk mitigation strategies.

TIME-SERIES TECHNIQUES IN QUANTITATIVE FORECASTING:
1. Simple/Single Exponential Smoothing (SES): Assigns exponentially decreasing
   weights to past observations. Best for data without trend or seasonality.
2. Double Exponential Smoothing (Holt's): Extends SES to capture linear trends
   in the data using two smoothing equations.
3. Triple Exponential Smoothing (Holt-Winters): Extends Holt's method to capture
   both trend and seasonality using three smoothing equations.
4. Moving Average: Uses the average of a fixed number of past observations.
5. ARIMA (AutoRegressive Integrated Moving Average): Combines autoregression,
   differencing, and moving average components.
6. GARCH: Used for forecasting volatility in financial time series.
""")

# Part 2: SES Calculation
print("=" * 80)
print("SINGLE EXPONENTIAL SMOOTHING (SES) CALCULATION")
print("=" * 80)
print(f"\nFormula: Ft+1 = Ft + alpha * (At - Ft)")
print(f"Smoothing constant (alpha): {alpha}")
print(f"Initial forecast F(2011): {initial_forecast} million USD")

print(f"\n{'Year':<8} {'Actual (At)':<15} {'Forecast (Ft)':<15} {'Error (At-Ft)':<15} {'alpha*(At-Ft)'}")
print("-" * 68)

forecasts = [initial_forecast]  # F(2011) = 75
errors = []

for i, (year, actual) in enumerate(zip(years, actual_losses)):
    ft = forecasts[i]
    error = actual - ft
    adjustment = alpha * error
    ft_next = ft + adjustment
    errors.append(error)

    print(f"{year:<8} {actual:<15} {ft:<15.2f} {error:<15.2f} {adjustment:.2f}")
    forecasts.append(ft_next)

# Forecast for 2016
forecast_2016 = forecasts[-1]
print(f"\n{'2016':<8} {'?':<15} {forecast_2016:<15.2f}")

print(f"\n{'=' * 50}")
print(f"FORECASTED LOSS FOR 2016: {forecast_2016:.2f} million USD")
print(f"{'=' * 50}")

# Step-by-step calculation
print(f"\nStep-by-step SES Calculation:")
print(f"  F(2011) = {initial_forecast} (given initial forecast)")
forecasts_step = [initial_forecast]
for i, (year, actual) in enumerate(zip(years, actual_losses)):
    ft = forecasts_step[i]
    ft_next = ft + alpha * (actual - ft)
    forecasts_step.append(ft_next)
    print(f"  F({year+1}) = F({year}) + {alpha} * (A({year}) - F({year}))")
    print(f"         = {ft:.2f} + {alpha} * ({actual} - {ft:.2f})")
    print(f"         = {ft:.2f} + {alpha} * {actual - ft:.2f}")
    print(f"         = {ft:.2f} + {alpha * (actual - ft):.2f}")
    print(f"         = {ft_next:.2f}")

# Statistical measures
print(f"\n{'=' * 50}")
print(f"STATISTICAL MEASURES OF ACTUAL LOSSES")
print(f"{'=' * 50}")

mean_loss = np.mean(actual_losses)
std_dev = np.std(actual_losses, ddof=1)  # Sample std dev
variance = np.var(actual_losses, ddof=1)  # Sample variance

print(f"  Actual Losses: {actual_losses}")
print(f"  Expected Loss (Mean):    {mean_loss:.2f} million USD")
print(f"  Standard Deviation:      {std_dev:.4f} million USD")
print(f"  Variance:                {variance:.4f} million USD^2")

print(f"\n  Calculation Details:")
print(f"  Mean = Sum / N = {sum(actual_losses)} / {len(actual_losses)} = {mean_loss:.2f}")
deviations = [(x - mean_loss) ** 2 for x in actual_losses]
print(f"  Deviations squared: {[f'{d:.2f}' for d in deviations]}")
print(f"  Sum of squared deviations: {sum(deviations):.2f}")
print(f"  Variance = {sum(deviations):.2f} / {len(actual_losses)-1} = {variance:.4f}")
print(f"  Std Dev = sqrt({variance:.4f}) = {std_dev:.4f}")
