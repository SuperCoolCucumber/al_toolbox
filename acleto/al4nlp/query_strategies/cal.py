from datasets.arrow_dataset import Dataset
import logging
from typing import Union
import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from .al_strategy_utils import _get_labels

from ..utils.get_embeddings import get_embeddings
from ..utils.transformers_dataset import TransformersDataset

log = logging.getLogger()


def cal(
    model,
    X_pool: Union[Dataset, TransformersDataset],
    n_instances: int,
    X_train: Union[Dataset, TransformersDataset],
    **cal_kwargs,
):
    """
    Selects unlabeled data points from the pool,
    whose predictive likelihoods diverge the most from their neighbors in the training set.
    https://arxiv.org/pdf/2109.03764.pdf
    """
    num_neighbors = cal_kwargs.get("n_neighbors", 10)
    if num_neighbors > len(X_train):
        num_neighbors = len(X_train)
    logits = model.predict_logits(X_pool)
    train_logits = model.predict_logits(X_train)
    kwargs = dict(
        # Necessary
        model_wrapper=model,
        data_train=X_train,
        data_test=X_pool,
        # General
        data_is_tokenized=False,
        data_config=None,
        batch_size=model._batch_size_kwargs.eval_batch_size,
        to_numpy=True,
        probas=logits,
        train_probas=train_logits,
        num_nei=num_neighbors,
    )

    uncertainty_estimates = calculate_cal_scores(**kwargs)
    query_idx = np.argpartition(uncertainty_estimates, -n_instances)[-n_instances:]

    query = X_pool.select(query_idx)

    return query_idx, query, uncertainty_estimates

# Contrastive Active Learning (CAL) https://aclanthology.org/2021.emnlp-main.51.pdf
def calculate_cal_scores(
    model_wrapper,
    data_train,
    data_test,
    probas,
    train_probas,
    use_activation: bool = False,
    use_spectralnorm: bool = False,
    data_is_tokenized=False,
    data_config=None,
    batch_size=100,
    to_numpy=True,
    num_nei=10,
    operator="mean",
):
    data_config = data_config if data_config is not None else model_wrapper.data_config
    kwargs = dict(
        # General
        model=model_wrapper.model,
        prepare_model=True,
        batch_size=batch_size,
        to_numpy=False,
        # DDU
        use_activation=use_activation,
        use_spectralnorm=use_spectralnorm,
        # Tokenization
        data_is_tokenized=data_is_tokenized,
        tokenizer=model_wrapper.tokenizer,
        task=model_wrapper.task,
        text_name=data_config["text_name"],
        label_name=data_config["label_name"],
    )

    train_embeddings = (
        get_embeddings(dataloader_or_data=data_train, **kwargs).detach().cpu()
    )
    test_embeddings = (
        get_embeddings(dataloader_or_data=data_test, **kwargs).detach().cpu()
    )

    if not isinstance(data_train, TransformersDataset):
        data_train = TransformersDataset(data_train)
    train_labels = _get_labels(data_train, data_config)

    distances = None
    num_adv = None

    neigh = KNeighborsClassifier(n_neighbors=num_nei)
    neigh.fit(X=train_embeddings, y=np.array(train_labels))

    criterion = torch.nn.KLDivLoss(reduction="none")

    kl_scores = []
    num_adv = 0
    distances = []
    pairs = []
    for unlab_i, candidate in enumerate(
        tqdm(
            zip(test_embeddings, probas),
            desc="Finding neighbours for every unlabeled data point",
        )
    ):
        # find indices of closesest "neighbours" in train set
        unlab_representation, unlab_logit = candidate
        distances_, neighbours = neigh.kneighbors(
            X=[candidate[0].numpy()], return_distance=True
        )
        distances.append(distances_[0])

        # remove outliers?
        # cur_mean_dist = np.max(distances_[0])

        labeled_neighbours_labels = train_labels[neighbours[0]]
        preds_neigh = [np.argmax(train_probas[n], axis=1) for n in neighbours]
        neigh_prob = torch.nn.functional.softmax(
            torch.Tensor(train_probas[neighbours]).to("cpu"), dim=-1
        )
        pred_candidate = [np.argmax(candidate[1])]

        uda_softmax_temp = 1
        candidate_log_prob = torch.nn.functional.log_softmax(
            torch.Tensor(candidate[1] / uda_softmax_temp).to("cpu"), dim=-1
        )
        kl = np.array(
            [
                torch.sum(criterion(candidate_log_prob, n), dim=-1).numpy()
                for n in neigh_prob
            ]
        )

        if operator == "mean":
            kl_scores.append(kl.mean())
        elif operator == "max":
            kl_scores.append(kl.max())
        elif operator == "median":
            kl_scores.append(np.median(kl))

    return np.array(kl_scores)
