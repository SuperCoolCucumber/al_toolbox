import os.path
import sys
from pathlib import Path
from copy import deepcopy
from functools import partial
from importlib import import_module

from ..active_learner import ActiveLearner
from ..pool_subsampling_strategies import (
    ups_subsampling,
    random_subsampling,
    naive_subsampling,
)
from ..query_strategies.actune import actune
from ..query_strategies.al_strategy import (
    mahalanobis_triplet_sampling,
    triplet_sampling,
    mahalanobis_filtering_sampling,
    ddu_sampling,
    logits_lc_sampling,
    margin_sampling,
    oracle_sampling,
    hybrid_sampling,
    ssal_sampling,
    ddu_sampling_cv,
)
from ..query_strategies.alps import alps
from ..query_strategies.badge import badge
from ..query_strategies.bait import bait
from ..query_strategies.bald import bald
from ..query_strategies.batchbald import batchbald
from ..query_strategies.breaking_ties import breaking_ties
from ..query_strategies.cal import cal
from ..query_strategies.cluster_margin import cluster_margin
from ..query_strategies.coreset import coreset
from ..query_strategies.egl import egl
from ..query_strategies.embeddings_km import embeddings_km
from ..query_strategies.entropy import entropy
from ..query_strategies.lc import lc
from ..query_strategies.mahalanobis_sampling import mahalanobis_sampling
from ..query_strategies.mnlp import mnlp
from ..query_strategies.random_sampling import random_sampling
from ..query_strategies.strategy_wrappers.small_text_sampling import small_text_sampling
from ..query_strategies.strategy_wrappers.modal_sampling import modal_sampling

QUERY_STRATEGIES = {
    # Classification strategies
    "random": partial(random_sampling, select_by_number_of_tokens=False),
    "entropy": entropy,
    "lc": lc,
    "ssal": ssal_sampling,
    "logits_lc": logits_lc_sampling,
    "br_ties": breaking_ties,
    "mahalanobis": mahalanobis_sampling,
    "mahalanobis_triplet": mahalanobis_triplet_sampling,
    "mahalanobis_filtering": mahalanobis_filtering_sampling,
    "triplet": triplet_sampling,
    "ddu": ddu_sampling,
    "margin": margin_sampling,
    "cal": cal,
    "oracle": oracle_sampling,
    "cluster_margin": cluster_margin,
    "hybrid": hybrid_sampling,
    "alps": alps,
    "coreset": coreset,
    "bait": bait,
    "badge": badge,
    "emb_km": embeddings_km,
    "egl": egl,
    "actune_lc": partial(actune, query_strategy=lc),
    "actune_ent": partial(actune, query_strategy=entropy),
    "actune_cal": partial(actune, query_strategy=cal),
    "actune_mahalanobis": partial(actune, query_strategy=mahalanobis_sampling),
    "small-text_lc": partial(small_text_sampling, small_text_strategy="lc"),
    "small-text_ent": partial(small_text_sampling, small_text_strategy="ent"),
    "modal_lc": partial(modal_sampling, modal_strategy="lc"),
    "modal_ent": partial(modal_sampling, modal_strategy="ent"),
    # NER strategies
    "mnlp_tokens": partial(mnlp, select_by_number_of_tokens=True),
    "mnlp_samples": partial(mnlp, select_by_number_of_tokens=False),
    "random_tokens": partial(random_sampling, select_by_number_of_tokens=True),
    "random_samples": partial(random_sampling, select_by_number_of_tokens=False),
    # BALD
    "bald": partial(bald, select_by_number_of_tokens=False, only_head_dropout=False),
    "bald_head": partial(
        bald, select_by_number_of_tokens=False, only_head_dropout=True
    ),
    "bald_tokens": partial(
        bald, select_by_number_of_tokens=True, only_head_dropout=False
    ),
    "bald_samples": partial(
        bald, select_by_number_of_tokens=False, only_head_dropout=False
    ),
    "bald_tokens_head": partial(
        bald, select_by_number_of_tokens=True, only_head_dropout=True
    ),
    "bald_samples_head": partial(
        bald, select_by_number_of_tokens=False, only_head_dropout=True
    ),
    # BatchBald
    "batchbald": partial(
        batchbald, select_by_number_of_tokens=False, only_head_dropout=False
    ),
    "batchbald_head": partial(
        batchbald, select_by_number_of_tokens=False, only_head_dropout=True
    ),
    "batchbald_tokens": partial(
        batchbald, select_by_number_of_tokens=True, only_head_dropout=False
    ),
    "batchbald_samples": partial(
        batchbald, select_by_number_of_tokens=False, only_head_dropout=False
    ),
    "batchbald_tokens_head": partial(
        batchbald, select_by_number_of_tokens=True, only_head_dropout=True
    ),
    "batchbald_samples_head": partial(
        batchbald, select_by_number_of_tokens=False, only_head_dropout=True
    ),
    # CV strategies
    "ddu_cv": ddu_sampling_cv,
}

SAMPLING_STRATEGIES = {
    "ups": ups_subsampling,
    "random": random_subsampling,
    "naive": naive_subsampling,
}


def construct_active_learner(
    model, config, initial_data, log_dir: str or Path, original_pool_size=None
):

    # TODO: rewrite using `split_by_tokens` as `strategy_kwargs`
    initial_data_copy = deepcopy(initial_data)
    use_subsampling = config.sampling_type is not None
    postfix = ""
    if ("split_by_tokens" in config) and (config.split_by_tokens):
        postfix += "_tokens"
    elif "split_by_tokens" in config:  # avoid adding "_samples" for classification
        postfix += "_samples"

    if config.strategy == "actune":
        postfix += getattr(config, "base_strategy", "_lc")

    if config.strategy == "small-text":
        postfix += getattr(config, "small_text_strategy", "_lc")

    if config.strategy == "modal":
        postfix += getattr(config, "modal_strategy", "_lc")

    query_strategy = QUERY_STRATEGIES.get(f"{config.strategy}{postfix}")
    # In this case, we assume that `config.strategy` refers to the path of the file with the strategy
    if query_strategy is None:
        query_strategy = _get_strategy_from_path(config.strategy)
    sampling_strategy = SAMPLING_STRATEGIES.get(config.sampling_type)
    if use_subsampling and sampling_strategy is None:
        sampling_strategy = _get_strategy_from_path(config.sampling_type)
    sampling_kwargs = {
        "gamma_or_k_confident_to_save": config.gamma_or_k_confident_to_save,
        "T": config.T,
    }
    strategy_kwargs = config.strategy_kwargs

    learner = ActiveLearner(
        estimator=model,
        query_strategy=query_strategy,
        train_data=initial_data_copy,
        strategy_kwargs=strategy_kwargs,
        sampling_strategy=sampling_strategy,
        sampling_kwargs=sampling_kwargs,
        log_dir=log_dir,
        original_pool_size=original_pool_size,
    )

    return learner


def _get_strategy_from_path(strategy_path):
    if (not strategy_path.endswith("/")) and ("/" in strategy_path):
        path = strategy_path[: strategy_path.rindex("/")]
    elif "/" in strategy_path:
        path = strategy_path[: strategy_path[:-2].rindex("/")]
    else:
        path = (
            Path("/".join(os.path.abspath(__file__).split("/")[:-2]))
            / "query_strategies"
        )
    strategy_name = strategy_path.split("/")[-1] or strategy_path.split("/")[-2]
    if strategy_name.endswith(".py"):
        strategy_name = strategy_name[:-3]
    sys.path.append(path)

    # For the custom function
    if "/" in strategy_path:
        strategy_func = getattr(import_module(strategy_name), strategy_name, None)
        if strategy_func is None:
            strategy_func = getattr(
                import_module(strategy_name), strategy_name + "_sampling"
            )
    else:
        package = "acleto.al4nlp.query_strategies"
        # From the library
        try:
            strategy_func = getattr(
                import_module("." + strategy_name, package=package), strategy_name, None
            )
        except:
            strategy_func = getattr(
                import_module("." + strategy_name + "_sampling", package=package),
                strategy_name + "_sampling",
                None,
            )
    return strategy_func
