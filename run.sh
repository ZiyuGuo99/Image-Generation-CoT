#!/bin/bash

# Set reward model and dpo (modify as needed)
REWARD_MODEL="parm"  # Choose from "orm_zs", "orm_ft", "parm", or leave empty to run without a reward model
DPO_MODEL="dpo_iter_parm_guide"  # Choose from "dpo", "dpo_iter", "dpo_iter_parm_guide", or leave empty to run vanilla show-o


torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=12475 main.py \
--prompts_file geneval/prompts/generation_prompts.txt \
--metadata_file geneval/prompts/evaluation_metadata.jsonl \
--config config.yaml \
--reward_model $REWARD_MODEL \
--dpo_model $DPO_MODEL 