import argparse
import json
import sys
import os

import pandas as pd
from matplotlib import pyplot as plt

os.environ["TOKENIZERS_PARALLELISM"] = "true"
import torch.distributed as dist

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from torch_geometric import seed_everything
from tqdm import tqdm
import math
import pytorch_lightning as pl

pl.seed_everything(1234)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, grandparent_dir)
from models import MAGVITv2, get_mask_chedule

from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next

from transformers import AutoTokenizer, AutoConfig

from models.modeling_utils import ConfigMixin, ModelMixin, register_to_config
from models.sampling import cosine_schedule, mask_by_random_topk
from models.phi import PhiForCausalLM

from selector import ImageSelector
torch.set_grad_enabled(False)

from parm import run_parm
from orm import run_orm
from baseline import run_showo

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts_file",
        type=str,
        default='prompts.txt',
        help="Text file containing prompts, one per line"
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default='metadata.jsonl',
        help="Metadata for geneval"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="show-o",
        help="Huggingface model name"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="dir to write results to",
        default="geneval/outputs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="how many samples can be produced simultaneously",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="showo_prm.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--validation_prompts_file",
        type=str,
        default=None,
        help="Path to the validation prompts file",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.75,
        help="Guidance scale for generation",
    )
    parser.add_argument(
        "--generation_timesteps",
        type=int,
        default=18,
        help="Number of timesteps for generation",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default='t2i',
        help="Mode of operation",
    )
    parser.add_argument(
        "--eval_num",
        type=int,
        default=4,
        help="for geneval benchmark"
    )
    parser.add_argument(
        "--search_num",
        type=int,
        default=20,
        help="search number"
    )
    parser.add_argument(
        "--reward_model",
        type=str,
        default=None,
        help="Mode of reward model",
    )
    parser.add_argument(
        '--dpo', 
        action='store_true',  
        help="Enable DPO"
    )
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_args()

    if opt.dpo:     
        print("Running with DPO...")

    if opt.reward_model == None:
        print("Running without reward model...")
        opt.outdir = "results/show-o_dpo" if opt.dpo else "results/show-o"
        run_showo(opt)
    elif opt.reward_model == 'orm':
        print("Running with reward model: ORM")
        opt.outdir = "results/show-o_orm_dpo" if opt.dpo else "results/show-o_orm"
        run_orm(opt)
    elif opt.reward_model == 'parm':
        print("Running with reward model: PARM")
        opt.outdir = "results/show-o_parm_dpo_20" if opt.dpo else "results/show-o_parm"
        run_parm(opt)
    else:
        raise ValueError('mode is not supported')