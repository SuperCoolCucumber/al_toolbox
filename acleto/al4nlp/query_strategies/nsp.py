from typing import Union
from datasets import Dataset
import numpy as np

from .al_strategy_utils import filter_by_metric, filter_by_uncertainty
from ..utils.transformers_dataset import TransformersDataset


def nsp(
    model,
    X_pool: Union[np.ndarray, Dataset, TransformersDataset],
    n_instances: int,
    **kwargs,
):
    # Filtering part begin
    filtering_mode = kwargs.get("filtering_mode", None)
    uncertainty_threshold = kwargs.get("uncertainty_threshold", None)
    uncertainty_mode = kwargs.get(
        "uncertainty_mode", "absolute"
    )  # "relative" or "absolute"
    # Filtering part end
    generate_output = model.generate(X_pool, to_numpy=True)
    scores = generate_output["sequences_scores"]
    sequences_ids = generate_output["sequences"]
    # The larger the score, the more confident the model is
    uncertainty_estimates = -scores
    # Filtering part begin
    if filtering_mode == "uncertainty":
        query_idx, uncertainty_estimates = filter_by_uncertainty(
            uncertainty_estimates=uncertainty_estimates,
            uncertainty_threshold=uncertainty_threshold,
            uncertainty_mode=uncertainty_mode,
            n_instances=n_instances,
        )
    elif filtering_mode in ["rouge1", "rouge2", "rougeL", "sacrebleu"]:
        query_idx, uncertainty_estimates = filter_by_metric(
            uncertainty_estimates=uncertainty_estimates,
            uncertainty_threshold=uncertainty_threshold,
            uncertainty_mode=uncertainty_mode,
            n_instances=n_instances,
            texts=X_pool[model.data_config["text_name"]],
            generated_sequences_ids=sequences_ids,
            tokenizer=model.tokenizer,
            metric_cache_dir=model.cache_dir / "metrics",
            metric_name=filtering_mode,
            agg=kwargs.get("filtering_aggregation", "precision"),
        )
    else:
        argsort = np.argsort(-uncertainty_estimates)
        query_idx = argsort[:n_instances]
    # Filtering part end
    query = X_pool.select(query_idx)

    return query_idx, query, uncertainty_estimates
