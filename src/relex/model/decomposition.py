# Custom function to decompose forecasts into components
import tensorflow_probability as tfp
from tensorflow_probability import sts


def decompose_forecast_components(
    model, observed_time_series, parameter_samples, forecast_dist, num_steps_forecast
):
    """
    Decompose the forecast into its component parts.
    This shows how each component contributes to the forecast.
    """
    # Dictionary to store component forecasts
    component_forecasts_dists = {}

    # For each component in the model
    for component in model.components:
        # Create a new model with just this component
        single_component_model = sts.Sum(
            [component], observed_time_series=observed_time_series
        )

        # Extract the relevant parameters for this component
        component_params = {}
        for param_name, param_value in parameter_samples.items():
            if component.name in param_name:
                component_params[param_name] = param_value

        # We need to add observation noise scale for the single component model
        component_params["observation_noise_scale"] = parameter_samples[
            "observation_noise_scale"
        ]

        # Forecast just this component
        component_forecast_dist = tfp.sts.forecast(
            model=single_component_model,
            observed_time_series=observed_time_series,
            parameter_samples=component_params,
            num_steps_forecast=num_steps_forecast,
        )

        # Store the component forecast
        component_forecasts_dists[component.name] = component_forecast_dist

    return component_forecasts_dists
