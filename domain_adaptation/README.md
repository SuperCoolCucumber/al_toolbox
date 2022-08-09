# Domain adaptation
Main code adapted from https://github.com/allenai/dont-stop-pretraining. This code includes the main script [run_lm.py](run_lm.py) that runs LM training and the [hf_dataset_to_sent.py](hf_dataset_to_sent.py) script that transforms Huggingface datasets into the appropriate format for model training.
## How to run
Firstly one has to transform a dataset to the appropriate format for training which is just a .txt file with each train sentence placed on a separate line. In the case of using datasets from the Huggingface library we have a simple python script for this purpose: [hf_dataset_to_sent.py](hf_dataset_to_sent.py).
After preparing the dataset for LM training one have to set training parameters into the config file (look for examples at [configs](configs)) and run the [run_lm.py](run_lm.py) script.
The full pipeline of domain adaptation could be run with the following commands:
```
export HYDRA_CONFIG_PATH=./configs
export HYDRA_CONFIG_NAME=domain_adaptation.yaml
export output_dir="bert-conll"
export dataset_name="'conll2003'"

python ./hf_dataset_to_sent.py \
 output_dir=$output_dir \
 data.dataset_name=$dataset_name \
 data.text_column_name=tokens \
 data.has_validation=True

python ./run_lm.py \
 output_dir=$output_dir \
 data.dataset_name=$dataset_name \
 weight_decay=1e-3 \
 learning_rate=1e-6 \
 gradient_accumulation_steps=4 \
 gpus=[0] \
 n_gpu=1 \
```
One should also look for train script examples in train.sh.
After domain adaptation is finished, just replace model.checkpoint entry in a config file for active learning with the path to the adapted model.