.. _new_stategies_addition:

=======================
New strategies addition
=======================

An AL query strategy should be designed as a function that:
   1) Receives 3 positional arguments and additional strategy kwargs:
    - ``model`` of inherited class ``TransformersBaseWrapper`` or ``PytorchEncoderWrapper`` or ``FlairModelWrapper``: model wrapper;
    - ``X_pool`` of class ``Dataset`` or ``TransformersDataset``: dataset with the unlabeled instances;
    - ``n_instances`` of class ``int``: number of instances to query;
    - ``kwargs``: additional strategy-specific arguments.
   2) Outputs 3 objects in the following order:
    - ``query_idx`` of class ``array-like``: array with the indices of the queried instances;
    - ``query`` of class ``Dataset`` or ``TransformersDataset``: dataset with the queried instances;
    - ``uncertainty_estimates`` of class ``np.ndarray``: uncertainty estimates of the instances from ``X_pool``. The higher the value - the more uncertain the model is in the instance.

The function with the strategy should be named the same as the file where it is placed (e.g. function ``def my_strategy`` inside a file ``path_to_strategy/my_strategy.py``).
Use your strategy, setting ``al.strategy=PATH_TO_FILE_YOUR_STRATEGY`` in the experiment config.

The example is presented in ``examples/benchmark_custom_strategy.ipynb``
