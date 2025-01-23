#!/bin/bash

IMAGE_FOLDER="show-o_parm_dpo_20"  
RESULTS_FOLDER="../results"  

python evaluation/evaluate_images.py \
    "$RESULTS_FOLDER/$IMAGE_FOLDER" \
    --outfile "$RESULTS_FOLDER/$IMAGE_FOLDER.jsonl" \
    --model-path "object"

python evaluation/summary_scores.py "$RESULTS_FOLDER/$IMAGE_FOLDER.jsonl"