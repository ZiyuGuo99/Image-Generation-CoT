#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONWARNINGS="ignore"

TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun \
--nnodes=1 --nproc_per_node=8 --node_rank=0 \
--master_port=12475 \
main.py \
--prompts_file geneval/prompts/generation_prompts.txt \
--metadata_file geneval/prompts/evaluation_metadata.jsonl \
--config config.yaml \
--reward_model parm \
--dpo \
"$@"

