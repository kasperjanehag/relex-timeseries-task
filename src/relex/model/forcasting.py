import tensorflow_probability as tfp


def forecast_model(item_model, data, parameter_samples):
    forecast_dist = tfp.sts.forecast(
        model=item_model,
        observed_time_series=data["train_sales"],
        parameter_samples=parameter_samples,
        num_steps_forecast=data["num_forecast_steps"],
    )

    # Get forecast statistics
    forecast_mean = forecast_dist.mean().numpy()[..., 0]
    forecast_scale = forecast_dist.stddev().numpy()[..., 0]
    forecast_samples = forecast_dist.sample(10).numpy()[..., 0]

    return forecast_dist, forecast_mean, forecast_scale, forecast_samples
