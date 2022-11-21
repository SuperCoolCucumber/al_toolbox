export HYDRA_CONFIG_PATH=./configs
export HYDRA_CONFIG_NAME=domain_adaptation.yaml
export output_dir="bert-conll"
export dataset_name="'conll2003'"

python ./hf_dataset_to_sent.py \
 output_dir=$output_dir \
 data.dataset_name=$dataset_name \
 data.text_name=tokens \
 data.label_name=ner_tags \
 data.source_task=ner \

python ./run_lm.py \
 output_dir=$output_dir \
 data.dataset_name=$dataset_name \
 weight_decay=1e-3 \
 learning_rate=1e-6 \
 gradient_accumulation_steps=4 \
 gpus=[0] \
 n_gpu=1 \
 max_steps=200