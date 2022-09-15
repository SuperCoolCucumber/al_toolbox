from typing import Union

import numpy as np
from datasets.arrow_dataset import Dataset

from ..utils.transformers_dataset import TransformersDataset


def akim(
    model,
    X_pool: Union[np.ndarray, Dataset, TransformersDataset],
    n_instances: int,
    **kwargs,
):
    dev_data: Dataset = model.dev_data
    subset_size = kwargs.get("subset_size", 100)
    label_name = kwargs.get("label_name", model.data_config["label_name"])
    dev_subset = dev_data.train_test_split(
        train_size=subset_size, shuffle=True, seed=model.seed
    )["train"]

    probas = model.predict_proba(X_pool)
    pseudo_labels = probas.argmax(1)

    for i, (inst, p_label) in enumerate(zip(X_pool, pseudo_labels)):
        model.model(**inst["input_ids"].unsqueeze(0))
        loss = None

    return

    # argsort = np.argsort(-uncertainty_estimates)
    # query_idx = argsort[:n_instances]
    # query = X_pool.select(query_idx)

    # return query_idx, query, uncertainty_estimates
