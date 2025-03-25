from logging import getLogger

import numpy as np

logger = getLogger(__name__)


def evaluate(data, forecast_mean, component_means, forecast_component_means):
    # Calculate forecast accuracy metrics
    forecast_error = data["test_sales"] - forecast_mean
    rmse = np.sqrt(np.mean(forecast_error**2))
    mape = np.mean(np.abs(forecast_error / data["test_sales"])) * 100

    logger.info("\nForecast Accuracy Metrics:")
    logger.info(f"RMSE: {rmse:.2f}")
    logger.info(f"MAPE: {mape:.2f}%")

    # Calculate component contributions (variance explained)
    total_variance = np.var(data["train_sales"])
    component_variances = {name: np.var(mean) for name, mean in component_means.items()}
    component_percentages = {
        name: var / total_variance * 100 for name, var in component_variances.items()
    }

    logger.info("\nComponent Contribution (% variance explained):")
    for name, percentage in component_percentages.items():
        logger.info(f"{name}: {percentage:.2f}%")

    # Calculate component contributions to forecast
    forecast_variance = np.var(forecast_mean)
    forecast_component_variances = {
        name: np.var(mean)
        for name, mean in forecast_component_means.items()  # Use consistent variable name
    }
    forecast_component_percentages = {
        name: var / forecast_variance * 100
        for name, var in forecast_component_variances.items()
    }

    logger.info("\nComponent Contribution to Forecast (% variance explained):")
    for name, percentage in forecast_component_percentages.items():
        logger.info(f"{name}: {percentage:.2f}%")

    return rmse, mape
