import logging
from typing import Union

import numpy as np
from datasets.arrow_dataset import Dataset
from scipy import stats
from sklearn.metrics import pairwise_distances

import torch
from torch import nn

from ..utils.get_embeddings import get_grad_embeddings
from ..utils.transformers_dataset import TransformersDataset

log = logging.getLogger()


def badge(
    model,
    X_pool: Union[Dataset, TransformersDataset],
    n_instances: int,
    **badge_kwargs,
):
    """
    Measures uncertainty as the gradient magnitude with respect to parameters in the final (output) layer,
    which is computed using the most likely label according to the model. To capture diversity, collect a
    batch of examples where these gradients span a diverse set of directions. https://arxiv.org/abs/1906.03671
    """
    logits = model.predict_logits(X_pool)

    kwargs = dict(
        # Necessary
        model_wrapper=model,
        data_test=X_pool,
        # General
        data_is_tokenized=False,
        data_config=None,
        batch_size=model._batch_size_kwargs.eval_batch_size,
        to_numpy=False,
        logits=logits,
    )

    vectors = calculate_badge_scores(**kwargs).cpu().detach().numpy()

    query_idx = np.array(init_centers(vectors, k=n_instances))

    query = X_pool.select(query_idx)

    # Uncertainty estimates are not defined for BADGE
    uncertainty_estimates = np.zeros(len(X_pool))

    return query_idx, query, uncertainty_estimates


def init_centers(x, k):
    ind = np.argmax([np.linalg.norm(s, 2) for s in x])
    mu = [x[ind]]
    inds_all = [ind]
    cent_inds = [0.0] * len(x)
    cent = 0
    while len(mu) < k:
        if len(mu) == 1:
            d2 = pairwise_distances(x, mu).ravel().astype(float)
        else:
            new_d = pairwise_distances(x, [mu[-1]]).ravel().astype(float)
            for i in range(len(x)):
                if d2[i] > new_d[i]:
                    cent_inds[i] = cent
                    d2[i] = new_d[i]
        d2 = d2.ravel().astype(float)
        d_dist = (d2 ** 2) / sum(d2 ** 2)
        custom_dist = stats.rv_discrete(
            name="custm", values=(np.arange(len(d2)), d_dist)
        )
        ind = custom_dist.rvs(size=1)[0]
        while ind in inds_all:
            ind = custom_dist.rvs(size=1)[0]
        mu.append(x[ind])
        inds_all.append(ind)
        cent += 1
    return inds_all

def calculate_badge_scores(
    model_wrapper,
    data_test,
    logits,
    use_activation: bool = False,
    use_spectralnorm: bool = False,
    data_is_tokenized=False,
    data_config=None,
    batch_size=100,
    to_numpy=True,
):
    data_config = data_config if data_config is not None else model_wrapper.data_config
    kwargs = dict(
        # General
        model=model_wrapper.model,
        prepare_model=True,
        batch_size=batch_size,
        to_numpy=False,
        data_is_tokenized=data_is_tokenized,
        tokenizer=model_wrapper.tokenizer,
        task=model_wrapper.task,
        text_name=data_config["text_name"],
        label_name=data_config["label_name"],
    )
    """Return the loss gradient with respect to the penultimate layer for BADGE"""
    pooled_output = get_grad_embeddings(dataloader_or_data=data_test, **kwargs).to(
        model_wrapper.model.device
    )
    batch_size, num_classes = logits.shape
    probs = nn.functional.softmax(
        torch.Tensor(logits).to(model_wrapper.model.device), dim=-1
    )
    preds = probs.argmax(dim=1)
    preds_oh = nn.functional.one_hot(preds, num_classes=num_classes)
    preds_oh = preds_oh.type(torch.cuda.FloatTensor)
    scales = probs - preds_oh
    grads_3d = torch.einsum("bi,bj->bij", scales, pooled_output)
    grads = grads_3d.view(batch_size, -1)
    return grads
