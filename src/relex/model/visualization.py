import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_forecast(
    dates_pd,
    y,
    forecast_mean,
    forecast_scale,
    forecast_samples,
    title,
    x_locator=None,
    x_formatter=None,
):
    """Plot a forecast distribution against the 'true' time series."""
    colors = sns.color_palette()
    c1, c2 = colors[0], colors[1]
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)

    num_steps = len(y)
    num_steps_forecast = forecast_mean.shape[-1]
    num_steps_train = num_steps - num_steps_forecast

    ax.plot(dates_pd, y, lw=2, color=c1, label="ground truth")

    # Create forecast dates correctly - weekly frequency
    last_train_date = dates_pd[num_steps_train - 1]

    # Create weekly forecast dates
    forecast_dates = pd.date_range(
        start=pd.to_datetime(last_train_date) + pd.Timedelta(days=7),
        periods=num_steps_forecast,
        freq="W",
    )

    # Convert to numpy datetime64 array for consistency
    forecast_steps = np.array(forecast_dates, dtype="datetime64[ns]")

    ax.plot(forecast_steps, forecast_samples.T, lw=1, color=c2, alpha=0.1)

    ax.plot(forecast_steps, forecast_mean, lw=2, ls="--", color=c2, label="forecast")
    ax.fill_between(
        forecast_steps,
        forecast_mean - 2 * forecast_scale,
        forecast_mean + 2 * forecast_scale,
        color=c2,
        alpha=0.2,
    )

    ymin, ymax = (
        min(np.min(forecast_samples), np.min(y)),
        max(np.max(forecast_samples), np.max(y)),
    )
    yrange = ymax - ymin
    ax.set_ylim([ymin - yrange * 0.1, ymax + yrange * 0.1])
    ax.set_title(f"{title}")
    ax.legend()

    if x_locator is not None:
        ax.xaxis.set_major_locator(x_locator)
        ax.xaxis.set_major_formatter(x_formatter)
        fig.autofmt_xdate()

    # ax.axvline(
    #     dates_pd[len(data["train_sales"]) - 1],
    #     linestyle="--",
    #     color="red",
    #     label="Forecast Start",
    # )
    # ax.legend()
    # plt.tight_layout()
    # plt.show()

    return fig, ax


def plot_components_with_forecast(
    dates,
    train_dates,
    forecast_dates,
    component_means_dict,
    component_stddevs_dict,
    component_forecast_means_dict=None,
    component_forecast_stddevs_dict=None,
    x_locator=None,
    x_formatter=None,
):
    """
    Plot the contributions of posterior components including their forecasts.
    Handles different dimension shapes for historical and forecast components.
    """
    import collections

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    colors = sns.color_palette()
    c1, c2 = colors[0], colors[1]
    c3 = colors[2]  # For forecasts

    axes_dict = collections.OrderedDict()
    num_components = len(component_means_dict)
    fig = plt.figure(figsize=(12, 2.5 * num_components))

    for i, component_name in enumerate(component_means_dict.keys()):
        component_mean = component_means_dict[component_name]
        component_stddev = component_stddevs_dict[component_name]

        # Ensure component_mean and component_stddev are 1D arrays
        if component_mean.ndim == 0:
            component_mean = np.array([component_mean])
        if component_stddev.ndim == 0:
            component_stddev = np.array([component_stddev])

        ax = fig.add_subplot(num_components, 1, 1 + i)

        # Plot historical component
        ax.plot(
            train_dates, component_mean, lw=2, color=c1, label="Historical Component"
        )
        ax.fill_between(
            train_dates,
            component_mean - 2 * component_stddev,
            component_mean + 2 * component_stddev,
            color=c1,
            alpha=0.3,
        )

        # If we have forecast components, plot those too
        if (
            component_forecast_means_dict is not None
            and component_forecast_stddevs_dict is not None
            and component_name in component_forecast_means_dict
        ):
            forecast_mean = component_forecast_means_dict[component_name]
            forecast_stddev = component_forecast_stddevs_dict[component_name]

            # Ensure forecast_mean and forecast_stddev are 1D arrays
            if forecast_mean.ndim == 0:
                forecast_mean = np.array([forecast_mean])
            if forecast_stddev.ndim == 0:
                forecast_stddev = np.array([forecast_stddev])

            # Print debug info if shapes still don't match forecast_dates
            if len(forecast_mean) != len(forecast_dates):
                print(
                    f"Warning: {component_name} forecast_mean length {len(forecast_mean)} != forecast_dates length {len(forecast_dates)}"
                )
                # Handle potential mismatch by taking the appropriate number of elements
                if len(forecast_mean) > len(forecast_dates):
                    forecast_mean = forecast_mean[: len(forecast_dates)]
                    forecast_stddev = forecast_stddev[: len(forecast_dates)]
                else:
                    # If there are fewer forecast values than dates, extend with the last value
                    extension = [forecast_mean[-1]] * (
                        len(forecast_dates) - len(forecast_mean)
                    )
                    forecast_mean = np.concatenate([forecast_mean, extension])
                    extension_std = [forecast_stddev[-1]] * (
                        len(forecast_dates) - len(forecast_stddev)
                    )
                    forecast_stddev = np.concatenate([forecast_stddev, extension_std])

            ax.plot(
                forecast_dates,
                forecast_mean,
                lw=2,
                ls="--",
                color=c3,
                label="Forecast Component",
            )
            ax.fill_between(
                forecast_dates,
                forecast_mean - 2 * forecast_stddev,
                forecast_mean + 2 * forecast_stddev,
                color=c3,
                alpha=0.3,
            )

        ax.set_title(component_name)
        if x_locator is not None:
            ax.xaxis.set_major_locator(x_locator)
            ax.xaxis.set_major_formatter(x_formatter)

        # Add legend if we have both historical and forecast data
        if component_forecast_means_dict is not None:
            ax.legend(loc="best")

        # Add vertical line at the forecast start point
        if len(train_dates) > 0:
            ax.axvline(train_dates[-1], linestyle="--", color="red", alpha=0.5)

        axes_dict[component_name] = ax

    fig.autofmt_xdate()
    fig.tight_layout()
    return fig, axes_dict


def plot_elbo_loss_curve(elbo_loss_curve, item_num):
    # Plot the loss curve
    plt.figure(figsize=(10, 4))
    plt.plot(elbo_loss_curve)
    plt.title(f"ELBO Loss Curve for Item {item_num}")
    plt.xlabel("Iteration")
    plt.ylabel("ELBO Loss")
    plt.show()


def plot_training_data(sales_df: pd.DataFrame, price_df: pd.DataFrame):
    # Visualize the data before modeling
    plt.figure(figsize=(14, 10))

    # Plot sales
    plt.subplot(2, 1, 1)
    for col in [col for col in sales_df.columns]:
        plt.plot(sales_df.index, sales_df[col], label=col)
    plt.title("Sales by Item")
    plt.legend()
    plt.grid(True)

    # Plot prices
    plt.subplot(2, 1, 2)
    for col in [col for col in price_df.columns]:
        plt.plot(price_df.index, price_df[col], label=col)
    plt.title("Prices by Item")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
