# Custom function to decompose forecasts into components
from logging import getLogger

import tensorflow as tf
import tensorflow_probability as tfp

logger = getLogger(__name__)


def decompose_historical_time_series(
    model, historical_data, parameter_samples, sales_col
):
    # Convert to tensor directly (not passing DataFrame to STS)
    observed_time_series = tf.convert_to_tensor(
        historical_data[sales_col].values, dtype=tf.float32
    )

    # Decompose the time series into components - only for training data
    logger.info("Decomposing historical time series into components...")
    component_dists = tfp.sts.decompose_by_component(
        model,
        observed_time_series=observed_time_series,
        parameter_samples=parameter_samples,
    )
    component_means = {k.name: c.mean().numpy() for k, c in component_dists.items()}
    component_stddevs = {k.name: c.stddev().numpy() for k, c in component_dists.items()}

    return component_means, component_stddevs


def decompose_forecast_time_series(model, forecast_dist, parameter_samples):
    # Now decompose the forecast into components
    logger.info("Decomposing forecast into components...")
    forecast_component_dists = tfp.sts.decompose_forecast_by_component(
        model=model,
        forecast_dist=forecast_dist,
        parameter_samples=parameter_samples,
    )
    forecast_component_means = {}
    forecast_component_stddevs = {}

    logger.info("\nComponent dimensions:")
    for k, c in forecast_component_dists.items():
        mean_val = c.mean().numpy()
        std_val = c.stddev().numpy()
        logger.info(
            f"Component {k.name}: mean shape {mean_val.shape}, stddev shape {std_val.shape}"
        )

        # The forecast components might have an extra dimension at the end
        # Extract the appropriate slice based on shape
        if mean_val.ndim > 1:
            mean_val = mean_val[..., 0]  # Take the first slice of the last dimension
        if std_val.ndim > 1:
            std_val = std_val[..., 0]

        forecast_component_means[k.name] = mean_val  # Use consistent variable name
        forecast_component_stddevs[k.name] = std_val  # Use consistent variable name

    return forecast_component_means, forecast_component_stddevs
