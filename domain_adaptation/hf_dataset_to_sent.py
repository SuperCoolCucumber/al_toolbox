import logging
import os
from pathlib import Path

import hydra
from datasets import load_dataset, Dataset
from tqdm import tqdm
from acleto.al4nlp.utils.data.load_data import load_data

log = logging.getLogger()


def instance_to_sentence(string, instance):
    try:
        string += instance + "\n"
    except:
        # ner case
        string += " ".join(instance).strip() + "\n"
    return string


@hydra.main(
    config_path=os.environ["HYDRA_CONFIG_PATH"],
    config_name=os.environ["HYDRA_CONFIG_NAME"],
)
def main(config):
    task = config.data.source_task
    train_data, val_data, test_data, _ = load_data(config.data, task, config.cache_dir)
    log.info(f"Train size: {len(train_data)}")
    log.info(f"Val size: {len(val_data)}")
    if test_data is not None:
        log.info(f"Test size: {len(test_data)}")
    else:
        log.info(f"Test size: {len(dataset['test'])}")

    os.makedirs(os.path.dirname(config.train_data_file), exist_ok=True)
    os.makedirs(os.path.dirname(config.eval_data_file), exist_ok=True)
    string = ""
    for inst in tqdm(train_data):
        string = instance_to_sentence(string, inst[config.data.text_name])

    with open(config.train_data_file, "w") as f:
        f.write(string)

    string = ""
    for inst in tqdm(val_data):
        string = instance_to_sentence(string, inst[config.data.text_name])

    with open(config.eval_data_file, "w") as f:
        f.write(string)


if __name__ == "__main__":
    main()
