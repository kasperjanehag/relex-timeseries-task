import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from relex.model.visualization import plot_elbo_loss_curve


def train_model(item_model, data, item_num):
    # Build and fit the variational distribution
    variational_posteriors = tfp.sts.build_factored_surrogate_posterior(
        model=item_model
    )

    # Minimize the variational loss to fit the model
    elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn=item_model.joint_distribution(
            observed_time_series=data["train_sales"]
        ).log_prob,
        surrogate_posterior=variational_posteriors,
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        num_steps=200,
        jit_compile=True,
    )

    plot_elbo_loss_curve(elbo_loss_curve, item_num)

    return variational_posteriors
