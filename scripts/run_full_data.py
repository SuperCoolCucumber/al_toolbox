import copy
import json
import logging
import os
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from acleto.al4nlp.utils.embeddings import load_embeddings_if_necessary
from acleto.al4nlp.utils.general import get_time_dict_path_full_data, log_config
from acleto.al4nlp.utils.main_decorator import main_decorator

log = logging.getLogger()


OmegaConf.register_new_resolver(
    "to_string", lambda x: x.replace("/", "_").replace("-", "_"), replace=True
)
OmegaConf.register_new_resolver(
    "get_patience_value", lambda dev_size: 1000 if dev_size == 0 else 5, replace=True
)


@main_decorator
def run_full_data(config, work_dir: Path or str):
    # Imports inside function to set environment variables before imports
    from acleto.al4nlp.constructors import construct_wrapper
    from acleto.al4nlp.utils.data.load_data import load_data
    from datasets import concatenate_datasets

    # Log config so that it is visible from the console
    log_config(log, config)
    log.info("Loading data...")
    cache_dir = config.cache_dir if config.cache_model_and_dataset else None
    train_instances, dev_instances, test_instances, labels_or_id2label = load_data(
        config.data, config.model.type, cache_dir,
    )
    if (
        dev_instances == test_instances
        and config.model.training.dev_size == 0
        and not config.get("force_use_dev_sample")
    ):
        config.model.training.dev_size = 0.1

    embeddings, word2idx = load_embeddings_if_necessary(
        train_instances, dev_instances, test_instances, config=config
    )

    # Initialize time dict
    time_dict_path = get_time_dict_path_full_data(config)

    log.info("Fitting the model...")
    model = construct_wrapper(
        config,
        config.model,
        dev_instances,
        config.framework,
        labels_or_id2label,
        "model",
        time_dict_path,
        embeddings=embeddings,
        word2idx=word2idx,
    )

    model.fit(train_instances)

    dev_metrics = model.evaluate(dev_instances)
    log.info(f"Dev metrics: {dev_metrics}")

    test_metrics = model.evaluate(test_instances)
    log.info(f"Test metrics: {test_metrics}")

    with open(work_dir / "dev_metrics.json", "w") as f:
        json.dump(dev_metrics, f)

    with open(work_dir / "metrics.json", "w") as f:
        json.dump(test_metrics, f)

    if config.dump_model:
        model.model.save_pretrained(work_dir / "model_checkpoint")
    log.info("Done with evaluation.")

    if getattr(config, "push_to_hub", False):
        dataset_name = (
            config.data.dataset_name
            if isinstance(config.data.dataset_name, str)
            else config.data.dataset_name[-1]
        )
        hub_name = f"{config.model.checkpoint}_{dataset_name}_{config.seed}"
        log.info("Hub name:", {hub_name})
        model.model.push_to_hub(hub_name, use_temp_dir=True)
        model.tokenizer.push_to_hub(hub_name, use_temp_dir=True)


@hydra.main(
    config_path=os.environ["HYDRA_CONFIG_PATH"],
    config_name=os.environ["HYDRA_CONFIG_NAME"],
)
def main(config):
    run_full_data(config)


if __name__ == "__main__":
    main()
