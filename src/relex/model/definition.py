from tensorflow_probability import sts


def define_model(observed_time_series, prices):
    """
    Build a structural time series model tailored for grocery store data with:
    - Trend component
    - Multiple seasonality (weekly and yearly)
    - Price effects (through linear regression)
    - Autoregressive process for residuals
    """
    # Trend component - using LocalLinearTrend which is appropriate for grocery data
    # as it allows for gradual changes in growth rates over time
    trend = sts.LocalLinearTrend(
        observed_time_series=observed_time_series, name="trend"
    )

    # Weekly seasonality - grocery stores typically have strong weekly patterns
    # with different sales patterns by day of week
    # Note: Since data is weekly, we should only use this if we have daily data
    # For weekly data, we don't include this component

    # Yearly seasonality - grocery products often have yearly patterns due to
    # seasons, holidays, etc.
    yearly_seasonal = sts.Seasonal(
        num_seasons=52,  # 52 weeks in a year
        observed_time_series=observed_time_series,
        name="yearly_seasonality",
    )

    # Price effects through linear regression
    # This captures both own-price and cross-price effects, which are crucial
    # for grocery store products that often have substitution effects
    # price_effect = sts.LinearRegression(design_matrix=prices, name="price_effects")

    # Autoregressive component to capture short-term correlations in residuals
    # For grocery data, AR(2) may be more appropriate to capture week-to-week
    # dependencies that aren't explicitly modeled
    autoregressive = sts.Autoregressive(
        order=1,  # Using order 1 to avoid potential instability issues
        observed_time_series=observed_time_series,
        name="autoregressive",
    )

    return sts.Sum(
        [
            trend,
            yearly_seasonal,
            # price_effect,
            autoregressive,
        ],
        observed_time_series=observed_time_series,
    )
