.. _new_subsampling_addition:

========================================
New pool subsampling strategies addition
========================================

The addition of a new pool subsampling query strategy is similar to the addition of an AL query strategy. A subsampling strategy should be designed as a function that:
   1) It must receive 2 positional arguments and additional subsampling strategy kwargs:
     - ``uncertainty_estimates`` of class ``np.ndarray``: uncertainty estimates of the instances in the order they are stored in the unlabeled data;
     - ``gamma_or_k_confident_to_save`` of class ``float`` or ``int``: either a share / number of instances to save (as in random / naive subsampling) or an internal parameter (as in UPS);
     - ``kwargs``: additional subsampling strategy specific arguments.
   2) It must output the indices of the instances to use (sampled indices) of class ``np.ndarray``.

The function with the strategy should be named the same as the file where it is placed (e.g. function ``def my_subsampling_strategy`` inside a file ``path_to_strategy/my_subsampling_strategy.py``).
Use your subsampling strategy, setting ``al.sampling_type=PATH_TO_FILE_YOUR_SUBSAMPLING_STRATEGY`` in the experiment config.

The example is presented in ``examples/benchmark_custom_strategy.ipynb``
