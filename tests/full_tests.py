import os
import warnings
from pathlib import Path
from shutil import rmtree
from time import time
from unittest.mock import patch
import argparse
import sys
sys.path.append("../../distillation")

from hydra import initialize, compose

warnings.filterwarnings("ignore")


os.environ["HYDRA_CONFIG_PATH"] = ""
os.environ["HYDRA_CONFIG_NAME"] = ""
if (
    Path("/proc/driver/nvidia/version").exists()
    and os.environ.get("USE_CUDA_FOR_TESTS", True)
    and len(os.environ.get("CUDA_VISIBLE_DEVICES", "0")) > 0
):
    USE_CUDA = True
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_ID_TO_USE", "0")
else:
    USE_CUDA = False
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

start_dir = os.getcwd()
Path("test_output").mkdir(exist_ok=True)
os.chdir("test_output")

from scripts.run_active_learning import run_active_learning
from scripts.run_full_data import run_full_data
from domain_adaptation.run_lm import main as run_da
from domain_adaptation.hf_dataset_to_sent import main as prepare_da_data
from distillation.scripts.binarized_data import main as dist_binarize
from distillation.scripts.token_counts import main as dist_counts
from distillation.train import main as dist_train
from test_parameters import (
    MODEL_OVERRIDES,
    AL_OVERRIDES,
    FULL_DATA_OVERRIDES,
    DA_OVERRIDES,
    DA_NER_OVERRIDES,
    DA_CLS_OVERRIDES,
    NER_TRANSFORMERS,
    CLS_TRANSFORMERS,
    ATS_TRANSFORMERS,
    AL_NER_TRANSFORMERS,
    AL_CLS_TRANSFORMERS,
    AL_ATS_TRANSFORMERS,
    FULL_NER_TRANSFORMERS,
    FULL_CLS_TRANSFORMERS,
    FULL_ATS_TRANSFORMERS,
    SUCCESSOR_OVERRIDES,
    TARGET_OVERRIDES,
    TARGET_OVERRIDES,
    DIST_BINARIZE_NAMESPACE,
    DIST_COUNTS_NAMESPACE,
    DIST_TRAIN_NAMESPACE,
)


# define some utils functions
def run_al_test_with_params(
    config_path, config_name, overrides, test_name, run_type="al"
):
    start_time = time()
    init_dir = os.getcwd()
    # To be able to launch from both `tests` dir and root dir
    data_path = get_data_path()
    if run_type != "distillation":
        key = "data.path"
        overrides = overrides + [
            f"{key}={data_path}",
        ]
        with initialize(config_path=config_path):
            config = compose(config_name=config_name, overrides=overrides,)
        work_dir = config.output_dir
    else:
        work_dir = config_path
        os.makedirs(work_dir, exist_ok=True)
        os.chdir(work_dir)
    Path(work_dir).mkdir(exist_ok=True)
    if run_type == "full":
        run_full_data(config)
    elif run_type == "al":
        run_active_learning(config)
    elif run_type == "da":
        prepare_da_data(config)
        run_da(config)
    elif run_type == "distillation":
        # run binarized_data
        with patch('argparse.ArgumentParser.parse_args',
                   return_value=DIST_BINARIZE_NAMESPACE) as args:
            os.makedirs("./data/", exist_ok=True)
            dist_binarize(**args)
        # run token counts
        with patch('argparse.ArgumentParser.parse_args',
                   return_value=DIST_COUNTS_NAMESPACE) as args:
            dist_counts(**args)
        # run distillation
        with patch('argparse.ArgumentParser.parse_args',
                   return_value=DIST_TRAIN_NAMESPACE) as args:
            dist_train(**args)
    else:
        raise NotImplementedError(f"{run_type} is not supported!")
    print(f"Test {test_name} passed within {time() - start_time:.1f} seconds")
    os.chdir(init_dir)
    rmtree(work_dir, ignore_errors=True)


def get_data_path():
    data_path = "../acleto/al_benchmark/data"
    if "al_benchmark" not in os.listdir():
        data_path = f"../{data_path}"
    return data_path


# block 1: test al and full data training for diff models and frameworks
# in this test we didn't check results but just check if experiments working
# without exceptions


def test_al_ner_transformers():
    config_path = "../acleto/al_benchmark/configs"
    config_name = "al_ner"
    test_name = "al ner transformers"
    run_al_test_with_params(config_path, config_name, AL_NER_TRANSFORMERS, test_name)


def test_full_ner_transformers():
    config_path = "../acleto/al_benchmark/configs"
    config_name = "full_data_ner"
    test_name = "full ner transformers"
    run_al_test_with_params(
        config_path, config_name, FULL_NER_TRANSFORMERS, test_name, "full"
    )


def test_al_cls_transformers():
    config_path = "../acleto/al_benchmark/configs"
    config_name = "al_cls"
    test_name = "al cls transformers"
    run_al_test_with_params(config_path, config_name, AL_CLS_TRANSFORMERS, test_name)


def test_full_cls_transformers():
    config_path = "../acleto/al_benchmark/configs"
    config_name = "full_data_cls"
    test_name = "full cls transformers"
    run_al_test_with_params(
        config_path, config_name, FULL_CLS_TRANSFORMERS, test_name, "full"
    )


def test_al_ner_plasm_transformers():
    config_path = "../acleto/al_benchmark/configs"
    config_name = "al_ner_plasm"
    test_name = "al ner plasm tracin transformers"
    plasm_tracin_overrides = (
        AL_NER_TRANSFORMERS + SUCCESSOR_OVERRIDES + TARGET_OVERRIDES
    )
    run_al_test_with_params(config_path, config_name, plasm_tracin_overrides, test_name)


def test_al_cls_plasm_transformers():
    config_path = "../acleto/al_benchmark/configs"
    config_name = "al_cls_plasm"
    test_name = "al cls plasm tracin transformers"
    plasm_tracin_overrides = (
        AL_CLS_TRANSFORMERS + SUCCESSOR_OVERRIDES + TARGET_OVERRIDES
    )
    run_al_test_with_params(config_path, config_name, plasm_tracin_overrides, test_name)


def test_al_ner_flair():
    config_path = "../acleto/al_benchmark/configs"
    config_name = "al_ner_bilstm_crf_flair"
    test_name = "al ner flair"
    flair_overrides = (
        AL_OVERRIDES + MODEL_OVERRIDES("acquisition_model")[1:] + NER_TRANSFORMERS
    )
    run_al_test_with_params(config_path, config_name, flair_overrides, test_name)


# abstractive summarization tests
def test_al_ats():
    config_path = "../acleto/al_benchmark/configs"
    config_name = "al_ats"
    test_name = "al ats"
    run_al_test_with_params(config_path, config_name, AL_ATS_TRANSFORMERS, test_name)


def test_full_ats():
    config_path = "../acleto/al_benchmark/configs"
    config_name = "full_data_ats"
    test_name = "full data ats"
    run_al_test_with_params(config_path, config_name, FULL_ATS_TRANSFORMERS, test_name, "full")


# domain adaptation tests
def test_domain_adaptation_ner():
    config_path = "../domain_adaptation/configs"
    config_name = "domain_adaptation"
    test_name = "domain adaptation ner"
    run_al_test_with_params(config_path, config_name, DA_NER_OVERRIDES, test_name, "da")


def test_domain_adaptation_cls():
    config_path = "../domain_adaptation/configs"
    config_name = "domain_adaptation"
    test_name = "domain adaptation cls"
    run_al_test_with_params(config_path, config_name, DA_CLS_OVERRIDES, test_name, "da")


def test_distillation():
    work_dir = "./test_distillation/"
    config_name = None
    test_name = "distillation"
    run_al_test_with_params(work_dir, config_name, None, test_name, "distillation")

    
def test_final():
    # just a final test that clears all cached files from previous tests
    os.chdir(start_dir)
    rmtree("test_output", ignore_errors=True)


"""
def test_al_ner_pytorch():
    config_path = "../acleto/al_benchmark/configs"
    config_name = "al_ner_bilstm"
    test_name = "al ner pytorch"
    pytorch_ner_overrides = AL_NER_TRANSFORMERS + [
        "+data.n_vectors=30",
        f"acquisition_model.embeddings_cache_dir=test_ner_use_{'cuda' if USE_CUDA else 'cpu'}/embeddings",
    ]
    run_al_test_with_params(config_path, config_name, pytorch_ner_overrides, test_name)
"""
