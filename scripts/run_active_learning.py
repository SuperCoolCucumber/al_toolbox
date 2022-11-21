import copy
import json
import logging
import os
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from acleto.al4nlp.utils.embeddings import load_embeddings_if_necessary
from acleto.al4nlp.utils.general import log_config
from acleto.al4nlp.utils.main_decorator import main_decorator
from acleto.al4nlp.utils.restore_queries import restore_queries

log = logging.getLogger()


OmegaConf.register_new_resolver(
    "to_string", lambda x: x.replace("/", "_").replace("-", "_"), replace=True
)
OmegaConf.register_new_resolver(
    "get_patience_value", lambda dev_size: 1000 if dev_size == 0 else 5, replace=True
)


@main_decorator
def run_active_learning(config, work_dir):
    # Imports inside function to set environment variables before imports
    from acleto.al_benchmark.simulated_active_learning import (
        al_loop,
        initial_split,
    )
    from acleto.al4nlp.utils.data.load_data import load_data
    from acleto.al4nlp.utils.transformers_dataset import TransformersDataset

    # Log config so that it is visible from the console
    log_config(log, config)
    log.info("Loading data...")
    cache_dir = config.cache_dir if config.cache_model_and_dataset else None
    train_instances, dev_instances, test_instances, labels_or_id2label = load_data(
        config.data, config.acquisition_model.type, cache_dir,
    )
    embeddings, word2idx = load_embeddings_if_necessary(
        train_instances, dev_instances, test_instances, config=config
    )
    # Make the initial split of the data onto labeled and unlabeled
    initial_data, unlabeled_data = initial_split(
        train_instances,
        config.al,
        config.acquisition_model.type,
        work_dir,
        tokens_column_name=config.data.text_name,
        labels_column_name=config.data.label_name,
        seed=config.seed,
    )
    log.info("Done.")

    # TODO: update all the code & configs for it
    # whether deep ensemble will be used
    use_de = "deep_ensemble" in config.al and config.al.deep_ensemble is not None
    id_first_iteration = 0

    try:
        from_checkpoint = config.get("from_checkpoint", None)
        if (
            from_checkpoint is not None
            and from_checkpoint["path"] is not None
        ):
            work_dir = Path(from_checkpoint["path"])
            initial_data, unlabeled_data = restore_queries(
                from_checkpoint,
                train_instances,
                initial_data,
                unlabeled_data,
                config.data.text_name
            )
        else:
            log.info("Starting active learning...")

        models = al_loop(
            config,
            work_dir,
            initial_data=initial_data,
            dev_data=dev_instances,
            test_data=test_instances,
            unlabeled_data=unlabeled_data,
            labels_or_id2label=labels_or_id2label,
            id_first_iteration=id_first_iteration,
            embeddings=embeddings,
            word2idx=word2idx,
        )
        log.info("Done with active learning...")
        return models
    except Exception as e:
        log.error(e, exc_info=True)


@hydra.main(
    config_path=os.environ.get("HYDRA_CONFIG_PATH", "al_benchmark/configs"),
    config_name=os.environ.get("HYDRA_CONFIG_NAME", "al_cls"),
)
def main(config):
    run_active_learning(config)


if __name__ == "__main__":
    main()
