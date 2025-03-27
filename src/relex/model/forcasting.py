import tensorflow as tf
import tensorflow_probability as tfp


def forecast_model(
    item_model, historical_data, parameter_samples, num_steps_forecast, sales_col
):
    # Convert to tensor directly (not passing DataFrame to STS)
    observed_time_series = tf.convert_to_tensor(
        historical_data[sales_col].values, dtype=tf.float32
    )

    forecast_dist = tfp.sts.forecast(
        model=item_model,
        observed_time_series=observed_time_series,
        parameter_samples=parameter_samples,
        num_steps_forecast=num_steps_forecast,
    )

    # Get forecast statistics
    forecast_mean = forecast_dist.mean().numpy()[..., 0]
    forecast_scale = forecast_dist.stddev().numpy()[..., 0]
    forecast_samples = forecast_dist.sample(10).numpy()[..., 0]

    return forecast_dist, forecast_mean, forecast_scale, forecast_samples
