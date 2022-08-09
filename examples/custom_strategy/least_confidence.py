import numpy as np


def least_confidence(model, X_pool, n_instances, **kwargs):
    probas = model.predict_proba(X_pool)
    uncertainty_estimates = 1 - probas.max(axis=1)
    query_idx = np.argsort(-uncertainty_estimates)[:n_instances]
    query = X_pool.select(query_idx)
    return query_idx, query, uncertainty_estimates
