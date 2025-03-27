import numpy as np
import tensorflow as tf
from tensorflow_probability import sts


def define_model(historical_data, forecast_data, item_num):
    """
    Build a structural time series model with consistent dtypes:
    - Trend component
    - Yearly seasonality
    - Monthly seasonality
    - Autoregressive process for residuals
    - Own-price elasticity effects
    - Cross-price effects (cross-price elasticity), the effect of a price decrease on one SKU on another SKU

    Parameters:
      training_data: DataFrame with historical sales and price data columns.
      forecast_data: DataFrame with forecast-period price data (for regressors).
      item_num: The item number to model.

    Returns:
      An STS model that includes all components.
    """
    # Assert that forecast_data has the same columns as training_data
    expected_forecast_columns = [
        col for col in historical_data.columns if not col.startswith("sales_")
    ]
    forecast_columns = forecast_data.columns.tolist()
    assert set(expected_forecast_columns).issubset(set(forecast_columns)), (
        f"Forecast data is missing some columns that are in training data: {set(expected_forecast_columns) - set(forecast_columns)}"
    )

    # Extract the relevant time series from the training dataframe.
    sales_col = f"sales_item_{item_num}"
    observed_time_series = tf.convert_to_tensor(
        historical_data[sales_col].values, dtype=tf.float32
    )

    # Define the trend component.
    trend = sts.LocalLinearTrend(
        observed_time_series=observed_time_series, name="trend"
    )

    # Define the yearly seasonality component.
    yearly_seasonal = sts.Seasonal(
        num_seasons=52,  # 52 weeks in a year
        observed_time_series=observed_time_series,
        name="yearly_seasonality",
    )

    # Define the monthly seasonality component.
    monthly_seasonal = sts.Seasonal(
        num_seasons=4,  # ~4 weeks in a month
        observed_time_series=observed_time_series,
        name="monthly_seasonality",
    )

    # Define the autoregressive component.
    autoregressive = sts.Autoregressive(
        order=2,
        observed_time_series=observed_time_series,
        name="autoregressive",
    )

    # Create the components list.
    components = [trend, yearly_seasonal, monthly_seasonal, autoregressive]

    # Add own-price effect
    own_price_col = f"price_item_{item_num}_centered"
    # Get historical and forecast regressor values.
    own_price_hist = historical_data[own_price_col].values
    own_price_forecast = forecast_data[own_price_col].values

    # Concatenate to form the full design matrix.
    full_design = np.concatenate([own_price_hist, own_price_forecast], axis=0)
    full_design_tensor = tf.convert_to_tensor(full_design, dtype=tf.float32)
    full_design_matrix = tf.reshape(full_design_tensor, [-1, 1])

    # Create the regression component using the full design matrix.
    own_price_effect = sts.LinearRegression(
        design_matrix=full_design_matrix,
        name="own_price_effect",
    )

    # Add the own-price component to your model.
    components.append(own_price_effect)

    # Add cross-price effects for all other items
    all_items = [1, 2, 3, 4]
    other_items = [i for i in all_items if i != item_num]

    for other_item in other_items:
        cross_price_col = f"price_item_{other_item}_centered"
        # Get historical and forecast regressor values for the other item
        cross_price_hist = historical_data[cross_price_col].values
        cross_price_forecast = forecast_data[cross_price_col].values

        # Concatenate to form the full design matrix
        cross_full_design = np.concatenate(
            [cross_price_hist, cross_price_forecast], axis=0
        )
        cross_full_design_tensor = tf.convert_to_tensor(
            cross_full_design, dtype=tf.float32
        )
        cross_full_design_matrix = tf.reshape(cross_full_design_tensor, [-1, 1])

        # Create the regression component for cross-price effect
        cross_price_effect = sts.LinearRegression(
            design_matrix=cross_full_design_matrix,
            name=f"cross_price_effect_{other_item}",
        )

        # Add the cross-price component to your model
        components.append(cross_price_effect)

    # Combine all components into a single sum model.
    model = sts.Sum(
        components,
        observed_time_series=observed_time_series,
    )

    return model
