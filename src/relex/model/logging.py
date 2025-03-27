from logging import getLogger

import numpy as np

logger = getLogger(__name__)


def log_inferred_parameters(model, parameter_samples):
    logger.info("Inferred parameters:")
    for param in model.parameters:
        logger.info(
            "{}: {} +- {}".format(
                param.name,
                np.mean(parameter_samples[param.name], axis=0),
                np.std(parameter_samples[param.name], axis=0),
            )
        )
