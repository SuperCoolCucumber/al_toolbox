from typing import Union

import numpy as np
from datasets.arrow_dataset import Dataset

from ..utils.transformers_dataset import TransformersDataset


def lc(
    model,
    X_pool: Union[np.ndarray, Dataset, TransformersDataset],
    n_instances: int,
    **kwargs,
):
    """
    Selects instances with the least prediction confidence (regarding the most likely class)
    https://arxiv.org/abs/cmp-lg/9407020.
    """
    probas = model.predict_proba(X_pool)
    uncertainty_estimates = 1 - np.max(probas, axis=1)
    argsort = np.argsort(-uncertainty_estimates)
    query_idx = argsort[:n_instances]
    query = X_pool.select(query_idx)

    return query_idx, query, uncertainty_estimates
