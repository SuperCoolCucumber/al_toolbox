.. _post_processing_usage:

===============
Post-processing
===============

Our framework provides tools for effective data post-processing for its re-usability and a possibility to build powerful models on it.
PLASM, which aims to alleviate the acquisition-successor mismatch problem and allow to build a model of an
arbitrary type using the labeled data without performance degradation, is implemented in `post_processing/pipeline_plasm`.
It uses the config `cls_plasm` / `ner_plasm` (from `jupyterlab_demo/configs). A brief explanation of the config structure:
    - pseudo-labeling model parameters are contained in the key `labeling_model`;
    - successor model parameters are contained in the key `successor_model`;
    - post-processing options are contained in the key `post_processing`:

        - `label_smoothing`: str / float / None, a parameter for label smoothing (LS) for pseudo-labeled instances. Accepts several options:
            - "adaptive": LS value equals the quality of the labeling model on the validation data.
            - float, 0 < value < 1: absolute value of label smoothing
            - None (default): no label smoothing is used
        - `labeled_weight`: int / float, weight for the labeled-by-human data. 1 < value < +inf
        - `use_subsample_for_pl`: int / float / None, the size of the subsample used for pseudo-labeling (float means taking the share of the unlabeled data). None means that no subsampling is used.
        - `uncertainty_threshold`: float / None, the value of the threshold for filtering by uncertainty. If None, no filtering by uncertainty is used.
        - `filter_by_quantile`: bool, only used for classification, ignored if `uncertainty_threshold` is None. If True, `uncertainty_threshold` most uncertain instances are filtered. Otherwise, all instances whose (1 - max_prob) < `uncertainty_threshold` are filtered.
        - `tracin`:
            - `use`: bool, whether to use TracIn for filtering
            - `max_num_processes`: int, value > 0, maximum number of processes per one GPU
            - `quantile`: str / float (0 < value < 1), share of unlabeled data instances to filter using the TracIn score.
            - `num_model_checkpoints`: int, value > 0, how many model checkpoints to save and use for TracIn.
            - `nu`: float / int, value for TracIn algorithm.