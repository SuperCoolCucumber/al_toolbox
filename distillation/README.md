# Acquisition model distillation
Main code adapted from https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation.
## How to run
Firstly one has to transform a dataset to the appropriate format - this can be done with [binarized_data.py](./scripts/binarized_data.py) and [token_counts.py](./scripts/token_counts.py) scripts. 
After preparing the dataset one have to prepare distillation model config file (look for examples at [training_configs](training_configs)) and run the [train.py](train.py) script.
The full pipeline of model distillation could be run with the following commands:
```
python scripts/binarized_data.py \
    --dataset_name ag_news \
    --tokenizer_type electra \
    --tokenizer_name google/electra-base-discriminator \
    --dump_file ./data/binarized_text &

python scripts/token_counts.py \
    --data_file data/binarized_text.electra.pickle \
    --token_counts_dump data/token_counts.electra.pickle \
    --vocab_size 30522 &
    
python train.py \
    --force \
    --student_type distilelectra \
    --student_config training_configs/distilelectra.json \
    --teacher_type electra \
    --teacher_name google/electra-base-discriminator \
    --alpha_ce 5.0 --alpha_mlm 2.0 --alpha_cos 1.0 --alpha_act 1.0 --alpha_clm 0.0 --mlm \
    --freeze_pos_embs \
    --data_file data/binarized_text.electra.pickle \
    --token_counts data/token_counts.electra.pickle \
    --dump_path ./serialization_dir/distilelectra \
    --force
```
One should also look for train script examples in [prepare_data.sh](prepare_data.sh), [token_count.sh](token_count.sh) and [distil.sh](distil.sh). In case of using multiple GPU instances for training one could use the [distil_distributed.sh](distil_distributed.sh) script.
After distillation is finished, just replace model.checkpoint entry in a config file for active learning with the path to the student model.