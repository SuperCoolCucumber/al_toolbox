import numpy as np
from typing import Union
from datasets.arrow_dataset import Dataset

from ..al_strategy_utils import take_idx

from ...utils.transformers_dataset import TransformersDataset


def modal_sampling(
    model,
    X_pool: Union[Dataset, TransformersDataset],
    n_instances: int,
    modal_strategy,
    **kwargs,
):
    from modAL.uncertainty import (
        entropy_sampling,
        uncertainty_sampling,
        margin_sampling,
    )

    modal_strategy_dict = {
        "lc": uncertainty_sampling,
        "ent": entropy_sampling,
        "margin": margin_sampling,
    }

    query_idx = modal_strategy_dict[modal_strategy](
        model, X_pool, n_instances=n_instances, **kwargs
    )

    uncertainty_estimates = np.zeros(len(X_pool))
    query = take_idx(X_pool, query_idx)

    return query_idx, query, uncertainty_estimates
