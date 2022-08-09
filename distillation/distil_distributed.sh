export NODE_RANK=0
export N_NODES=1

export N_GPU_NODE=4
export WORLD_SIZE=4

pkill -f 'python -u train.py'

python -m torch.distributed.launch \
    --nproc_per_node=$N_GPU_NODE \
    --nnodes=$N_NODES \
    --node_rank $NODE_RANK \
     train.py \
        --force \
        --n_gpu $WORLD_SIZE \
        --student_type distilelectra \
        --student_config training_configs/distilelectra.json \
        --teacher_type electra \
        --teacher_name google/electra-base-discriminator \
        --alpha_ce 5.0 --alpha_mlm 2.0 --alpha_cos 1.0 --alpha_act 1.0 --alpha_clm 0.0 --mlm \
        --freeze_pos_embs \
        --data_file data/binarized_text.electra.pickle \
        --token_counts data/token_counts.electra.pickle \
        --dump_path ./serialization_dir/distilelectra \
        --force # overwrites the `dump_path` if it already exists.