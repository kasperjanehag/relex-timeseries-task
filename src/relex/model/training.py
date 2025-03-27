import logging

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from relex.model.visualization import plot_elbo_loss_curve

logger = logging.getLogger(__name__)


def train_model(model, historical_data, item_num, sales_col):
    """
    Train a structural time series model using variational inference.

    Parameters:
    model: The STS model to train
    historical_data: DataFrame containing training data
    item_num: The item number being modeled
    sales_col: The column name for sales data

    Returns:
    variational_posteriors: Fitted variational distribution
    """

    # Convert to tensor directly (not passing DataFrame to STS)
    observed_time_series = tf.convert_to_tensor(
        historical_data[sales_col].values, dtype=tf.float32
    )

    # Build the variational surrogate posterior
    variational_posteriors = tfp.sts.build_factored_surrogate_posterior(model=model)

    # Create a loss tracker to store the loss at each step
    losses = []

    # Simply return the losses as the trace
    def trace_fn(state):
        # Get the loss from the state
        loss = state.loss
        # Store the loss in our list
        losses.append(float(loss))
        return loss

    # Minimize the variational loss to fit the model
    logger.info(f"Training model for item {item_num}...")

    _ = tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn=model.joint_distribution(
            observed_time_series=observed_time_series
        ).log_prob,
        surrogate_posterior=variational_posteriors,
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        num_steps=200,
        jit_compile=False,  # Disable JIT compilation
        trace_fn=trace_fn,
    )

    # Convert list of losses to numpy array
    elbo_loss_curve = np.array(losses)

    # Plot the ELBO loss curve
    plot_elbo_loss_curve(elbo_loss_curve, item_num)

    logger.info(f"Model training complete for item {item_num}")

    return variational_posteriors
