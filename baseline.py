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
import traceback

def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")
    
class Showo(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            w_clip_vit,
            vocab_size,
            llm_vocab_size,
            llm_model_path='',
            codebook_size=8192,
            num_vq_tokens=256,
            load_from_showo=True,
            **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.register_to_config(mask_token_id=vocab_size - 1)
        if load_from_showo:
            config = AutoConfig.from_pretrained(llm_model_path)
            self.showo = PhiForCausalLM(config)
        else:
            self.showo = PhiForCausalLM.from_pretrained(llm_model_path, attn_implementation='sdpa')
        self.showo.resize_token_embeddings(self.vocab_size)
        self.output_size = self.vocab_size

        if self.w_clip_vit:
            self.mm_projector = torch.nn.Sequential(
                torch.nn.Linear(1024, 2048),
                torch.nn.GELU(),
                torch.nn.Linear(2048, 2048)
            )

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = True

    def forward(
            self,
            input_ids,
            input_embeddings=None,
            attention_mask=None,
            labels=None,
            label_smoothing=0.0,
            batch_size_t2i=0,
            batch_size_lm=0,
            batch_size_mmu=0,
            max_seq_length=128,
            labels_mask_text=None,
            labels_mask_image=None,
            **kwargs,
    ):

        if input_embeddings is None:
            logits = self.showo(input_ids=input_ids, attention_mask=attention_mask)['logits']
        else:
            logits = self.showo(inputs_embeds=input_embeddings, attention_mask=attention_mask)['logits']

        if labels is not None:
            # 1. Mask token prediction (discrete diffusion) for image generation
            # Note that, max_seq_length indicates the maximum number of text tokens, maybe a bit confused.
            loss_t2i = F.cross_entropy(
                logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.output_size),
                labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
            )

            # 2. Next token prediction for language modeling
            loss_lm = F.cross_entropy(
                logits[batch_size_t2i:batch_size_t2i + batch_size_lm, :-1].contiguous().view(-1, self.output_size),
                labels[batch_size_t2i:batch_size_t2i + batch_size_lm, 1:].contiguous().view(-1), ignore_index=-100,
            )

            # 3. Next token prediction for captioning/multimodal understanding
            loss_mmu = F.cross_entropy(
                logits[-batch_size_mmu:, :-1].contiguous().view(-1, self.output_size),
                labels[-batch_size_mmu:, 1:].contiguous().view(-1), ignore_index=-100,
            )

            return logits, loss_t2i, loss_lm, loss_mmu

        return logits

    def t2i_generate(
            self,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            prompts_batch=None,
            device=None,
            outpath=None,
            process_path=None,
            sample_path=None,
            vq_model=None,
            selector=None,
            temperature=1.0,
            timesteps=18,  # ideal number of steps is 18 in maskgit paper
            guidance_scale=0,
            noise_schedule=cosine_schedule,
            generator: torch.Generator = None,
            config=None,
            **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """
        input_ids_copy = input_ids.clone().detach()
        uncond_input_ids_copy = uncond_input_ids.clone().detach()
        attention_mask_copy = attention_mask.clone().detach()
        
        for _ in range(config.eval_num):
            input_ids = input_ids_copy.clone().detach()
            uncond_input_ids = uncond_input_ids_copy.clone().detach()
            attention_mask = attention_mask_copy.clone().detach()
            # begin with all image token ids masked
            mask_token_id = self.config.mask_token_id
            num_vq_tokens = config.model.showo.num_vq_tokens
            num_new_special_tokens = config.model.showo.num_new_special_tokens

            input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
            input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id,
                                                        mask_token_id,
                                                        input_ids_minus_lm_vocab_size - config.model.showo.llm_vocab_size - num_new_special_tokens)

            # for classifier-free guidance
            if uncond_input_ids is not None:
                uncond_prefix = uncond_input_ids[:, :config.dataset.preprocessing.max_seq_length + 1]

            for step in range(timesteps):
                if uncond_input_ids is not None and guidance_scale > 0:
                    uncond_input_ids = torch.cat(
                        [uncond_prefix, input_ids[:, config.dataset.preprocessing.max_seq_length + 1:]], dim=1)
                    model_input = torch.cat([input_ids, uncond_input_ids])
                    cond_logits, uncond_logits = self(model_input, attention_mask=attention_mask).chunk(2)
                    # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                    # it seems that muse has a different cfg setting
                    logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                    logits = logits[:, -(num_vq_tokens + 1):-1, config.model.showo.llm_vocab_size + num_new_special_tokens:-1]
                else:
                    logits = self(input_ids, attention_mask=attention_mask)
                    logits = logits[:, -(num_vq_tokens + 1):-1, config.model.showo.llm_vocab_size + num_new_special_tokens:-1]

                probs = logits.softmax(dim=-1)
                sampled = probs.reshape(-1, logits.size(-1))
                sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])

                unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
                sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
                # Defines the mask ratio for the next round. The number to mask out is
                # determined by mask_ratio * unknown_number_in_the_beginning.
                ratio = 1.0 * (step + 1) / timesteps
                mask_ratio = noise_schedule(torch.tensor(ratio))
                # Computes the probabilities of each selected tokens.
                selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
                selected_probs = selected_probs.squeeze(-1)

                # Ignores the tokens given in the input by overwriting their confidence.
                selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
                # Gets mask lens for each sample in the batch according to the mask ratio.
                mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
                # Keeps at least one of prediction in this round and also masks out at least
                # one and for the next iteration
                mask_len = torch.max(
                    torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
                )
                # Adds noise for randomness
                temperature = temperature * (1.0 - ratio)
                masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
                # Masks tokens with lower confidence.
                input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id,
                                                              sampled_ids + config.model.showo.llm_vocab_size
                                                              + num_new_special_tokens)
                input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)

                sampled_ids_copy = sampled_ids.clone()
                sampled_ids_copy = torch.clamp(sampled_ids_copy, max=config.model.showo.codebook_size - 1, min=0)
                images = vq_model.decode_code(sampled_ids_copy)

                images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
                images *= 255.0
                images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

                image_filename = f"step{step}.png"
                image_path = os.path.join(process_path, image_filename)
                
                samples = [Image.fromarray(image) for image in images]

                for sample in samples:
                    sample.save(image_path)

                if step == timesteps - 1:
                    os.system("cp " + image_path + " " + os.path.join(sample_path, f"{_:04d}.png")) 
        return None
        
def main(opt):
    # Load prompts
    with open(opt.prompts_file) as fp:
        prompts = [line.strip() for line in fp if line.strip()]

    # Load metadata
    metadata_list = []

    try:
        with open(opt.metadata_file, 'r', encoding='utf-8') as metadata_file:
            metadata_list = json.load(metadata_file)
    except json.JSONDecodeError:
        with open(opt.metadata_file, 'r', encoding='utf-8') as metadata_file:
            for line in metadata_file:
                metadata_list.append(json.loads(line))

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()

    torch.cuda.set_device(device)
    # print(f"Starting rank={rank}, world_size={dist.get_world_size()}.")

    # Load model
    if opt.model == 'show-o':
        cli_conf = OmegaConf.create({
            "batch_size": opt.batch_size,
            "validation_prompts_file": opt.validation_prompts_file,
            "generation_timesteps": opt.generation_timesteps,
            "guidance_scale": opt.guidance_scale,
            "mode": "t2i",
            "eval_num": opt.eval_num,
            "config": opt.config,
        })
        yaml_conf = OmegaConf.load(cli_conf.config)
        config = OmegaConf.merge(yaml_conf, cli_conf)

        tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")

        uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                           special_tokens=(
                                               "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>",
                                               "<|t2v|>",
                                               "<|v2v|>", "<|lvg|>"),
                                           ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)
        
        model = Showo.from_pretrained(opt.dpo_model_path).to(device)
        model.eval()

        mask_token_id = model.config.mask_token_id

        # load from users passed arguments
        if config.get("validation_prompts_file", None) is not None:
            config.dataset.params.validation_prompts_file = config.validation_prompts_file
        config.training.batch_size = config.batch_size
        config.training.guidance_scale = config.guidance_scale
    else:
        raise ValueError('model is not supported')
    
    # Load vq_model
    vq_model = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    # Load selector
    selector = None

    dist.barrier()
    global_n_samples = dist.get_world_size()
    total_prompts = int(math.ceil(len(prompts) / global_n_samples) * global_n_samples)
    new_prompts = prompts + [prompts[0]] * (total_prompts - len(prompts))
    per_gpu_prompts = new_prompts[rank:total_prompts:global_n_samples]
    
    os.makedirs(opt.outdir, exist_ok=True)

    for index, prompt in tqdm(enumerate(per_gpu_prompts), total=len(per_gpu_prompts), desc=f"Rank {rank}"):
        
        global_index = index * global_n_samples + rank

        if global_index > len(prompts) - 1:
            break

        outpath = os.path.join(opt.outdir, f"{global_index:0>5}")
        os.makedirs(outpath, exist_ok=True)
        
        process_path = os.path.join(outpath, "process")
        os.makedirs(process_path, exist_ok=True)

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)

        if global_index < len(metadata_list):
            metadata = metadata_list[global_index]
            if metadata['prompt'] != prompt:
                raise ValueError(f"Mismatch detected at index {global_index}: Metadata prompt '{metadata['prompt']}' "
                              f"does not match the current prompt '{prompt}'. Aborting process.")
            with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
                json.dump(metadata, fp, indent=4)

        # Generate images
        prompts_batch = [prompt]
        image_tokens = torch.ones((len(prompts_batch), config.model.showo.num_vq_tokens),
                                  dtype=torch.long, device=device) * mask_token_id

        input_ids, _ = uni_prompting((prompts_batch, image_tokens), 't2i_gen')
        
        if config.training.guidance_scale > 0:
            uncond_input_ids, _ = uni_prompting(([''], image_tokens), 't2i_gen')
            attention_mask = create_attention_mask_predict_next(torch.cat([input_ids, uncond_input_ids], dim=0),
                                                                pad_id=int(
                                                                    uni_prompting.sptids_dict['<|pad|>']),
                                                                soi_id=int(
                                                                    uni_prompting.sptids_dict['<|soi|>']),
                                                                eoi_id=int(
                                                                    uni_prompting.sptids_dict['<|eoi|>']),
                                                                rm_pad_in_image=True)
        else:
            attention_mask = create_attention_mask_predict_next(input_ids,
                                                                pad_id=int(
                                                                    uni_prompting.sptids_dict['<|pad|>']),
                                                                soi_id=int(
                                                                    uni_prompting.sptids_dict['<|soi|>']),
                                                                eoi_id=int(
                                                                    uni_prompting.sptids_dict['<|eoi|>']),
                                                                rm_pad_in_image=True)
            uncond_input_ids = None

        if config.get("mask_schedule", None) is not None:
            schedule = config.mask_schedule.schedule
            args = config.mask_schedule.get("params", {})
            mask_schedule = get_mask_chedule(schedule, **args)
        else:
            mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))

        gen_token_ids = model.t2i_generate(
            input_ids=input_ids.clone(),
            uncond_input_ids=uncond_input_ids.clone(),
            attention_mask=attention_mask.clone(),
            prompts_batch=prompts_batch,
            device=device,
            outpath=outpath,
            process_path=process_path,
            sample_path=sample_path,
            vq_model=vq_model,
            selector=selector,
            guidance_scale=config.training.guidance_scale,
            temperature=config.training.get("generation_temperature", 1.0),
            timesteps=config.generation_timesteps,
            noise_schedule=mask_schedule,
            noise_type=config.training.get("noise_type", "mask"),
            seq_len=config.model.showo.num_vq_tokens,
            uni_prompting=uni_prompting,
            config=config,
        )

    dist.barrier()
    if rank == 0:
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()

def run_showo(opt):
    try:
        main(opt)
    except Exception as e:
        rank = int(os.environ.get("RANK", -1))
        print(f"Error in rank {rank}: {e}")
        traceback.print_exc()
