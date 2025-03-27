import collections

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_forecast(
    historical_data,
    forecast_data,
    forecast_mean,
    forecast_scale,
    forecast_samples,
    sales_col,
    title,
    x_locator=None,
    x_formatter=None,
):
    """
    Plot a forecast distribution against the observed time series.

    Parameters:
    train_df: DataFrame containing training data with datetime index
    test_df: DataFrame containing test data with datetime index
    forecast_mean: Mean values of the forecast
    forecast_scale: Standard deviation of the forecast
    forecast_samples: Sample paths from the forecast distribution
    sales_col: Column name for the sales data
    title: Title for the plot
    x_locator: Optional matplotlib locator for x-axis
    x_formatter: Optional matplotlib formatter for x-axis

    Returns:
    fig, ax: The matplotlib figure and axis objects
    """
    colors = sns.color_palette()
    c1, c2 = colors[0], colors[1]
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)

    # Extract datetime index and sales data
    train_dates = historical_data.index
    test_dates = forecast_data.index
    all_dates = pd.concat([historical_data, forecast_data]).index

    # Get actual sales data
    train_sales = historical_data[sales_col].values
    test_sales = (
        forecast_data[sales_col].values if sales_col in forecast_data.columns else None
    )

    # Plot the training data
    ax.plot(train_dates, train_sales, lw=2, color=c1, label="observed")

    # Plot test data if available
    if test_sales is not None:
        ax.plot(test_dates, test_sales, lw=2, color=c1, linestyle="-")

    # Plot forecast samples
    for i in range(forecast_samples.shape[0]):
        ax.plot(test_dates, forecast_samples[i], lw=1, color=c2, alpha=0.1)

    # Plot forecast mean and confidence interval
    ax.plot(test_dates, forecast_mean, lw=2, ls="--", color=c2, label="forecast")
    ax.fill_between(
        test_dates,
        forecast_mean - 2 * forecast_scale,
        forecast_mean + 2 * forecast_scale,
        color=c2,
        alpha=0.2,
        label="95% confidence",
    )

    # Set y-axis limits
    all_values = np.concatenate(
        [
            train_sales,
            forecast_samples.flatten(),
            forecast_mean - 2 * forecast_scale,
            forecast_mean + 2 * forecast_scale,
        ]
    )

    if test_sales is not None:
        all_values = np.concatenate([all_values, test_sales])

    ymin, ymax = np.min(all_values), np.max(all_values)
    yrange = ymax - ymin
    ax.set_ylim([ymin - yrange * 0.1, ymax + yrange * 0.1])

    # Set the title and customize the x-axis if needed
    ax.set_title(title)
    ax.legend()

    if x_locator is not None:
        ax.xaxis.set_major_locator(x_locator)
        ax.xaxis.set_major_formatter(x_formatter)
        fig.autofmt_xdate()

    # Add vertical line to separate training and forecast periods
    ax.axvline(
        train_dates[-1], linestyle="--", color="k", alpha=0.5, label="Forecast Start"
    )

    ax.legend()
    plt.tight_layout()

    return fig, ax


def plot_components_with_forecast(
    historical_dates,
    forecast_dates,
    historical_component_means,
    historical_component_stddevs,
    forecast_component_means=None,
    forecast_component_stddevs=None,
    x_locator=None,
    x_formatter=None,
    item_num=None,
):
    """
    Plot the contributions of posterior components including their forecasts.
    Handles different dimension shapes for historical and forecast components.
    """

    colors = sns.color_palette()
    c1, c2 = colors[0], colors[1]
    c3 = colors[2]  # For forecasts

    axes_dict = collections.OrderedDict()
    num_components = len(historical_component_means)
    fig = plt.figure(figsize=(12, 2.5 * num_components))

    for i, component_name in enumerate(historical_component_means.keys()):
        component_mean = historical_component_means[component_name]
        component_stddev = historical_component_stddevs[component_name]

        # Ensure component_mean and component_stddev are 1D arrays
        if component_mean.ndim == 0:
            component_mean = np.array([component_mean])
        if component_stddev.ndim == 0:
            component_stddev = np.array([component_stddev])

        ax = fig.add_subplot(num_components, 1, 1 + i)

        # Plot historical component
        ax.plot(
            historical_dates,
            component_mean,
            lw=2,
            color=c1,
            label="Historical Component",
        )
        ax.fill_between(
            historical_dates,
            component_mean - 2 * component_stddev,
            component_mean + 2 * component_stddev,
            color=c1,
            alpha=0.3,
        )

        # If we have forecast components, plot those too
        if (
            forecast_component_means is not None
            and forecast_component_stddevs is not None
            and component_name in forecast_component_means
        ):
            forecast_mean = forecast_component_means[component_name]
            forecast_stddev = forecast_component_stddevs[component_name]

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
        if forecast_component_means is not None:
            ax.legend(loc="best")

        # Add vertical line at the forecast start point
        if len(historical_dates) > 0:
            ax.axvline(historical_dates[-1], linestyle="--", color="red", alpha=0.5)

        axes_dict[component_name] = ax

    fig.autofmt_xdate()
    fig.tight_layout()
    plt.suptitle(f"Component Decomposition for Item {item_num}")
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


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
