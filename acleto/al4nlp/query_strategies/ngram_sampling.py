from typing import Union, Tuple
from datasets import Dataset
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from ..utils.transformers_dataset import TransformersDataset


def ngram_sampling(
    model,
    X_pool: Union[Dataset, TransformersDataset],
    n_instances: int,
    X_train: Union[Dataset, TransformersDataset],
    **strategy_kwargs,
) -> Tuple[np.ndarray, Union[Dataset, TransformersDataset], np.ndarray]:
    text_name = model.data_config["text_name"]
    ngram_range = strategy_kwargs.get("ngram_range", (1, 1))
    if isinstance(ngram_range, int):
        ngram_range = (ngram_range, ngram_range)
    truncate_texts = strategy_kwargs.get("truncate_texts", True)
    tokenizer = model.tokenizer
    tokenization_kwargs = {"truncation": False}
    if truncate_texts:
        tokenization_kwargs["truncation"] = True
    X_train_texts = tokenizer.batch_decode(
        tokenizer(X_train[text_name], **tokenization_kwargs)["input_ids"]
    )
    X_pool_texts = tokenizer.batch_decode(
        tokenizer(X_pool[text_name], **tokenization_kwargs)["input_ids"]
    )

    counter = CountVectorizer(ngram_range=ngram_range)
    unlabeled_ngramms = counter.fit_transform(X_pool_texts)
    labeled_ngramms = counter.transform(X_train_texts)

    ngrams_unl_freq = unlabeled_ngramms.sum(axis=0).A.ravel()
    ngrams_lab_freq = labeled_ngramms.sum(axis=0).A.ravel() + 1
    ngrams_scores = np.log(ngrams_unl_freq / ngrams_lab_freq)

    uncertainty_estimates = (
        unlabeled_ngramms.dot(ngrams_scores) / unlabeled_ngramms.sum(axis=1).A.ravel()
    )
    query_idx = np.argsort(-uncertainty_estimates)[:n_instances]
    query = X_pool.select(query_idx)
    return query_idx, query, uncertainty_estimates
