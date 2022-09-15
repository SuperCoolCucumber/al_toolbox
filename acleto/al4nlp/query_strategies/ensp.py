from typing import Union
from datasets import Dataset
import numpy as np
from transformers import set_seed

from .al_strategy_utils import assign_ue_scores_for_unlabeled_data, get_X_pool_subsample
from ..utils.transformers_dataset import TransformersDataset


def ensp(
    model,
    X_pool: Union[np.ndarray, Dataset, TransformersDataset],
    n_instances: int,
    **kwargs,
):
    aggregation = kwargs.get("aggregation", "var")
    func_for_agg = getattr(np, aggregation)
    larger_is_less_uncertain = False
    if aggregation in ["mean", "median", "max", "min"]:
        larger_is_less_uncertain = True
    use_log = kwargs.get("use_log", False)
    mc_iterations = kwargs.get("mc_iterations", 1)
    seed = kwargs.get("seed", 42)

    generate_kwargs = dict(to_numpy=True)
    if kwargs.get("enable_dropout", False):
        if seed is not None:
            set_seed(seed)
        # Since BART exploits F.dropout instead of nn.Dropout, we can only turn it on via .train()
        model.model.train()
        update_kwargs = {"do_sample": False, "to_eval_mode": True}
    else:
        model.model.eval()
        update_kwargs = {
            "do_sample": True,
            "to_eval_mode": False,
            "top_p": kwargs.get("generate_top_p", 0.95),
        }
    generate_kwargs.update(update_kwargs)

    if mc_iterations <= 1:
        num_beams = kwargs.get("num_beams", 10)
        mc_iterations = num_beams / model._trainer_kwargs.generation_num_beams
        num_return_sequences = kwargs.get("num_return_sequences", 10)
        assert (
            num_return_sequences <= num_beams
        ), f"`num_beams` must be >= `num_return_sequences`, got values {num_beams}, {num_return_sequences}"

    if kwargs.get("subsample_size_mc_dropout", True):
        X_pool_subsample, subsample_indices = get_X_pool_subsample(
            X_pool, mc_iterations, model.seed
        )
    else:
        X_pool_subsample, subsample_indices = X_pool, np.arange(len(X_pool))

    if isinstance(mc_iterations, int):
        summaries, log_scores = [], []
        for _ in range(mc_iterations):
            generate_output = model.generate(X_pool_subsample, **generate_kwargs)
            log_scores.append(generate_output["sequences_scores"])
            summaries.append(generate_output["predictions"])
    else:
        generate_output = model.generate(
            X_pool_subsample,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            **generate_kwargs,
        )
        log_scores = generate_output["sequences_scores"]
        # summaries = generate_output["predictions"]

    scores = np.r_[log_scores]
    if not use_log:
        scores = np.exp(scores)
    subsample_uncertainty_estimates = func_for_agg(scores, axis=0)
    if larger_is_less_uncertain:
        subsample_uncertainty_estimates = -subsample_uncertainty_estimates

    argsort = np.argsort(-subsample_uncertainty_estimates)
    subsample_query_idx = argsort[:n_instances]
    query = X_pool_subsample.select(subsample_query_idx)
    query_idx = subsample_indices[subsample_query_idx]

    uncertainty_estimates = assign_ue_scores_for_unlabeled_data(
        len(X_pool), subsample_indices, subsample_uncertainty_estimates
    )

    return query_idx, query, uncertainty_estimates
