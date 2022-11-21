from datasets import concatenate_datasets, Dataset
import numpy as np
import logging
from pathlib import Path
import os
import json
from omegaconf import DictConfig
from typing import Union

from acleto.al4nlp.utils.transformers_dataset import TransformersDataset


log = logging.getLogger()


def restore_queries(
        from_checkpoint: Union[DictConfig, str, Path],
        train_instances: Union[Dataset, TransformersDataset],
        initial_data: Union[Dataset, TransformersDataset] = None,
        unlabeled_data: Union[Dataset, TransformersDataset] = None,
        text_column: str = "text"
):
    if isinstance(from_checkpoint, (Path, str)):
        work_dir = Path(from_checkpoint)
        last_iter = None
    else:
        work_dir = Path(from_checkpoint["path"])
        last_iter = from_checkpoint.get("last_iteration", None)

    ids_paths = sorted(
        [x for x in os.listdir(work_dir) if x.startswith("ids_data_query_")]
    )
    if last_iter is not None:
        ids_paths = ids_paths[: last_iter + 1]
    id_first_iteration = len(ids_paths) - 1
    log.info(
        f"Resuming active learning from iteration {id_first_iteration + 1}..."
    )

    for i in range(len(ids_paths)):
        ids_data_query_path = f"ids_data_query_{i}.json"
        with open(work_dir / ids_data_query_path) as f:
            query_ids = json.load(f)
        if ids_data_query_path == "ids_data_query_0.json":
            query = train_instances.select(query_ids)
            if initial_data is not None:
                assert (
                    query[text_column] == initial_data[text_column]
                ), "Initial iteration results differ!"
            else:
                initial_data = query
                unlabeled_data = train_instances.select(
                    np.setdiff1d(range(len(train_instances)), query_ids)
                )
        else:
            query = unlabeled_data.select(query_ids)
            if isinstance(initial_data, TransformersDataset):
                initial_data.add(query)
            elif isinstance(initial_data, Dataset):
                initial_data = concatenate_datasets([initial_data, query])
            else:
                initial_data = query
            unlabeled_data = unlabeled_data.select(
                np.setdiff1d(range(len(unlabeled_data)), query_ids)
            )
    log.info(
        f"Query ids up to iteration {id_first_iteration} inclusive successfully loaded."
    )
    return initial_data, unlabeled_data
