import logging
from typing import Union

import numpy as np
import torch
from datasets.arrow_dataset import Dataset

from .al_strategy_utils import (
    assign_ue_scores_for_unlabeled_data,
    calculate_bald_score_cls,
    calculate_bald_score_ner,
    get_query_idx_for_selecting_by_number_of_tokens,
    get_X_pool_subsample
)
from .strategy_utils.batchbald.consistent_dropout import make_dropouts_consistent
from ..utils.transformers_dataset import TransformersDataset

log = logging.getLogger()


def entropy(probas):
    return - (probas * probas.log()).sum(-1)


def kl_div(p, q):
    return (p * (p / q).log()).sum()


def akim_batchbald(
    model,
    X_pool: Union[np.ndarray, Dataset, TransformersDataset],
    n_instances: int,
    **bald_kwargs,
):
    mc_iterations = bald_kwargs.get("mc_iterations", 10)
    use_stable_dropout = bald_kwargs.get("use_stable_dropout", False)
    use_subsample = bald_kwargs.get("use_subsample", False)

    lamb = bald_kwargs.get("lamb", None)
    if lamb is not None:
        label_name = model.data_config["label_name"]
        labels = bald_kwargs.get("X_train")[label_name]
        true_labels_probas = torch.Tensor(np.bincount(labels) / len(labels))

    if use_subsample:
        X_pool_subsample, subsample_indices = get_X_pool_subsample(
            X_pool, mc_iterations, model.seed
        )
    else:
        X_pool_subsample = X_pool

    # Make dropout consistent inside huggingface model
    if use_stable_dropout:
        make_dropouts_consistent(model.model)
    else:
        model.enable_dropout()

    if bald_kwargs.get("only_head_dropout", False):
        raise NotImplementedError
    else:
        # Stable dropout
        probas = []
        for _ in range(mc_iterations):
            if use_stable_dropout:
                # Reset masks
                model.enable_dropout()
                model.disable_dropout()
            probas_iter = model.predict_proba(
                X_pool_subsample, use_predict_loop=True, to_eval_mode=False
            )
            probas.append(probas_iter)

    query_idx = []
    unselected_idx = list(range(len(X_pool_subsample)))
    query_scores = []
    probas = torch.Tensor(probas).transpose(1, 0)
    av_probas = probas.mean(1)
    data_unc = entropy(probas).mean(-1)
    for i in range(n_instances):
        unc_iter = torch.empty(len(unselected_idx), dtype=torch.float32, device=av_probas.device)
        for j, idx in enumerate(unselected_idx):
            avg_query_probas = av_probas[query_idx + [idx]].mean(0)
            unc = entropy(avg_query_probas)
            if lamb is not None:
                unc -= lamb * kl_div(
                    true_labels_probas, avg_query_probas
                )
            unc_iter[j].copy_(unc)

        data_unc_iter = (data_unc[query_idx].sum(0) + data_unc[unselected_idx]) / (len(query_idx) + 1)
        scores_iter = unc_iter - data_unc_iter
        argmax = scores_iter.argmax().item()
        iter_query_idx = unselected_idx[argmax]
        query_idx.append(iter_query_idx)
        unselected_idx.remove(iter_query_idx)
        query_scores.append(scores_iter[argmax].item())

    uncertainty_estimates = np.empty(len(X_pool_subsample), dtype=np.float32)
    uncertainty_estimates[query_idx] = query_scores
    uncertainty_estimates[unselected_idx] = torch.cat((scores_iter[:argmax], scores_iter[argmax + 1:]))

    query = X_pool_subsample.select(query_idx)
    if use_subsample:
        # Update query_idx since now it corresponds to the sampled set
        query_idx = subsample_indices[query_idx]
        uncertainty_estimates = assign_ue_scores_for_unlabeled_data(
            len(X_pool), subsample_indices, uncertainty_estimates
        )

    return query_idx, query, uncertainty_estimates
