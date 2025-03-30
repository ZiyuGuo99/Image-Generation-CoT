import sys, os, copy, random, yaml
from accelerate import Accelerator
from accelerate.utils import DistributedType, set_seed
from torch import nn
import torch.nn.functional as F
sys.path.append("./")
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from tqdm.auto import tqdm
os.environ['NCCL_P2P_DISABLE']='1'
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from dataclasses import field, dataclass
from safetensors.torch import load_file
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
import logging
import math
from pathlib import Path
from typing import Union, Dict, Optional, List, Any
import numpy as np
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW
from PIL import Image
from transformers import AutoTokenizer
from models import Showo, MAGVITv2, get_mask_chedule
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next
from models.lr_schedulers import get_scheduler
from accelerate.logging import get_logger
from models.logging import set_verbosity_info, set_verbosity_error
from llava.utils import rank0_print
from training.utils import flatten_omega_conf, mask_or_random_replace_tokens, AverageMeter

logger = get_logger(__name__, log_level="INFO")


@dataclass
class OtherArguments:
    config: Optional[str] = field(default="dpo.yaml")


def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        max_seq_length=128,
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
        label_pad_token_id: The label pad token id.
        is_encoder_decoder: Whether the model is an encoder-decoder model.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

    logits = logits[:, max_seq_length + 1:]
    labels = labels[:, max_seq_length + 1:]
    labels = labels.clone()
    loss_mask = labels != label_pad_token_id

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == label_pad_token_id] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def get_vq_model_class(model_type):
    assert model_type == "magvitv2"
    return MAGVITv2


def load_data(data_path):
    data_list = []
    if "jsonl" in data_path:
        with open(data_path, "r") as json_file:
            for line in json_file:
                data_list.append(json.loads(line.strip()))
    else:
        with open(data_path, "r") as json_file:
            data_list = json.load(json_file)
    return data_list


class DPODataset(Dataset):
    def __init__(self, config: dict, device=None):
        super(DPODataset, self).__init__()
        # Handle multiple JSON files specified in the data_path
        self.list_data_dict = []

        data_path = config.dataset.params.data_path
        assert data_path.endswith(".yaml")
        with open(data_path, "r") as file:
            yaml_data = yaml.safe_load(file)
            datasets = yaml_data.get("datasets")
            for dataset in datasets:
                json_path = dataset.get("json_path")
                sampling_strategy = dataset.get("sampling_strategy", "all")
                sampling_number = None

                rank0_print(f"Loading {json_path} with {sampling_strategy} sampling strategy")
                cur_data_dict = load_data(json_path)

                if ":" in sampling_strategy:
                    sampling_strategy, sampling_number = sampling_strategy.split(":")
                    if "%" in sampling_number:
                        sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                    else:
                        sampling_number = int(sampling_number)

                # Apply the sampling strategy
                if sampling_strategy == "first" and sampling_number is not None:
                    cur_data_dict = cur_data_dict[:sampling_number]
                elif sampling_strategy == "end" and sampling_number is not None:
                    cur_data_dict = cur_data_dict[-sampling_number:]
                elif sampling_strategy == "random" and sampling_number is not None:
                    random.shuffle(cur_data_dict)
                    cur_data_dict = cur_data_dict[:sampling_number]

                rank0_print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                self.list_data_dict.extend(cur_data_dict)

        self.config = config
        self.device = device

    def __len__(self):
        return len(self.list_data_dict)

    def transform_image(self, image: Image):
        resolution = self.config.dataset.preprocessing.resolution
        image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
        image = transforms.CenterCrop((resolution, resolution))(image)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
        return image

    def __getitem__(self, i):
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"
        data_dict = copy.deepcopy(self.list_data_dict[i])
        data_dict["chosen"] = self.transform_image(Image.open(data_dict["chosen"]))
        data_dict['rejected'] = self.transform_image(Image.open(data_dict["rejected"]))
        return data_dict


@dataclass
class DPODataCollator(object):

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_prompt_list = [feature['prompt'] for feature in features]
        batch_chosen_image = torch.stack([feature['chosen'] for feature in features], dim=0)
        batch_rejected_image = torch.stack([feature['rejected'] for feature in features], dim=0)
        final_batch = {}
        final_batch['batch_prompt_list'] = batch_prompt_list
        final_batch['batch_chosen_image'] = batch_chosen_image
        final_batch['batch_rejected_image'] = batch_rejected_image
        return final_batch


def main():
    other_args = OtherArguments()
    config = OmegaConf.load(other_args.config)
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            config.training.batch_size_t2i
        )

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.name + "_beta_" + str(config.training.beta) + "_lr_rate_" + str(
                config.optimizer.params.learning_rate) + "_weight_decay_" + str(config.optimizer.params.weight_decay),
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        # wandb_config.pop("experiment.resume_from_checkpoint")

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    device = accelerator.device

    pretrained = config.model.rw_model
    model_name = "llava_qwen"
    device_map = {"": device}
    llava_model_args = {
        "multimodal": True,
    }

    overwrite_config = {}
    overwrite_config["image_aspect_ratio"] = "pad"
    llava_model_args["overwrite_config"] = overwrite_config
    if config.training.reward_coef != 0:
        reward_tokenizer, reward_model, reward_image_processor, _ = load_pretrained_model(pretrained, None, model_name,
                                                                                          device_map=device_map,
                                                                                          **llava_model_args)

        reward_model.eval()

    if config.training.seed is not None:
        set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-1_5', padding_side="left")

    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=(
                                           "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
                                           "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
                                       ),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)

    rank0_print('special tokens : \n', uni_prompting.sptids_dict)

    def find_checkpoint_file(ckt_dir):
        for file_name in os.listdir(ckt_dir):
            if file_name.endswith('.bin') or file_name.endswith('.safetensors'):
                return os.path.join(ckt_dir, file_name)
        raise FileNotFoundError("No .bin or .safetensors file found in the directory.")

    def load_model_weights(model, ckt_dir, device):
        ckt_path = find_checkpoint_file(ckt_dir)

        if ckt_path.endswith('.safetensors'):
            state_dict = load_file(ckt_path)
        else:
            state_dict = torch.load(ckt_path, map_location=device)

        state_dict = {k: v.to(device) for k, v in state_dict.items()}

        model.load_state_dict(state_dict)

    # VQ model for processing image into discrete tokens
    vq_model = get_vq_model_class(config.model.vq_model.type)

    vq_model = vq_model.from_pretrained("showlab/magvitv2").to(device)
    vq_model.eval()
    vq_model.requires_grad_(False)

    # Initialize Show-o model
    if config.model.showo.load_from_showo:
        model = Showo.from_pretrained(config.model.showo.pretrained_model_path).to(device)

        if config.model.showo.vocab_size != model.vocab_size:
            model.showo.resize_token_embeddings(config.model.showo.vocab_size)
            model.config.codebook_size = config.model.showo.codebook_size
            model.config.vocab_size = config.model.showo.vocab_size
            model.vocab_size = config.model.showo.vocab_size
            model.output_size = config.model.showo.vocab_size
            model.config.mask_token_id = config.model.showo.vocab_size - 1
            model.mask_token_id = config.model.showo.vocab_size - 1
    else:
        model = Showo(**config.model.showo).to(device)
        load_model_weights(model, config.model.showo.pretrained_model_path, str(device))
        ref_model = Showo(**config.model.showo).to(device)
        load_model_weights(ref_model, config.model.showo.pretrained_model_path, str(device)) ###


    mask_id = model.mask_token_id

    parameter_names = [n for n, _ in ref_model.named_parameters()]
    for param_name in parameter_names:
        param = ref_model.get_parameter(param_name)
        param.requires_grad = False
    ref_model.eval()

    # Create mask scheduler
    if config.get("mask_schedule", None) is not None:
        schedule = config.mask_schedule.schedule
        args = config.mask_schedule.get("params", {})
        mask_schedule = get_mask_chedule(schedule, **args)
    else:
        mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))

    if hasattr(model, 'module'):
        mask_dtype = model.module.showo.model.embed_tokens.weight.dtype
    else:
        mask_dtype = model.showo.model.embed_tokens.weight.dtype

    train_dataset = DPODataset(config=config, device=device)

    data_collator = DPODataCollator()

    dataloader_params = {
        "batch_size": config.training.batch_size_t2i,
        "collate_fn": data_collator,
        "num_workers": config.dataset.params.num_workers,
        "pin_memory": config.dataset.params.pin_memory,
        "persistent_workers": config.dataset.params.persistent_workers,
    }

    data_loader = DataLoader(train_dataset, **dataloader_params)

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")
    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=len(data_loader)*config.training.num_epoch,
        num_warmup_steps=int(config.lr_scheduler.params.warmup_ratio * len(data_loader)*config.training.num_epoch),
    )

    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    @torch.no_grad()
    def prepare_inputs_and_labels(
            pixel_values_or_image_ids: Union[torch.FloatTensor, torch.LongTensor],
            texts: Union[str, str],
            min_masking_rate: float = 0.0,
            is_train: bool = True,
    ):
        image_tokens = vq_model.get_code(pixel_values_or_image_ids)
        image_tokens = image_tokens + len(uni_prompting.text_tokenizer)

        # create MLM mask and labels
        input_ids, labels, loss_weight, mask_prob = mask_or_random_replace_tokens(
            image_tokens,
            mask_id,
            config,
            mask_schedule=mask_schedule,
            is_train=is_train,
        )
        input_ids, masks, labels = uni_prompting((texts, input_ids, labels), 't2i')
        return input_ids, labels, mask_prob, image_tokens

    global_step = 0
    for epoch in range(0, config.training.num_epoch):
        model.train()
        epoch_iterator = tqdm(data_loader, desc=f"Epoch {epoch}", position=0, leave=True)
        for step, batch in enumerate(epoch_iterator):
            batch_prompt_list = batch['batch_prompt_list']
            batch_chosen_image = batch['batch_chosen_image'].to(device)
            batch_rejected_image = batch['batch_rejected_image'].to(device)
            chosen_input_ids, chosen_labels, chosen_mask_prob, chosen_image_tokens = prepare_inputs_and_labels(
                batch_chosen_image,
                batch_prompt_list, config.training.min_masking_rate)

            rejected_input_ids, rejected_labels, rejected_mask_prob, rejected_image_tokens = prepare_inputs_and_labels(
                batch_rejected_image, batch_prompt_list, config.training.min_masking_rate)

            chosen_attention_mask = create_attention_mask_predict_next(chosen_input_ids,
                                                                       pad_id=int(
                                                                           uni_prompting.sptids_dict['<|pad|>']),
                                                                       soi_id=int(
                                                                           uni_prompting.sptids_dict['<|soi|>']),
                                                                       eoi_id=int(
                                                                           uni_prompting.sptids_dict['<|eoi|>']),
                                                                       rm_pad_in_image=True,
                                                                       return_inverse_mask=True).to(mask_dtype)
            rejected_attention_mask = create_attention_mask_predict_next(rejected_input_ids,
                                                                         pad_id=int(
                                                                             uni_prompting.sptids_dict['<|pad|>']),
                                                                         soi_id=int(
                                                                             uni_prompting.sptids_dict['<|soi|>']),
                                                                         eoi_id=int(
                                                                             uni_prompting.sptids_dict['<|eoi|>']),
                                                                         rm_pad_in_image=True,
                                                                         return_inverse_mask=True).to(mask_dtype)

            def concatenated_forward(model: nn.Module, chosen_input_ids, rejected_input_ids, chosen_labels,
                                     rejected_labels, chosen_attention_mask, rejected_attention_mask):
                len_chosen = chosen_input_ids.shape[0]
                concatenated_input_ids = torch.cat((chosen_input_ids, rejected_input_ids), dim=0)
                concatenated_labels = torch.cat((chosen_labels, rejected_labels), dim=0)

                concatenated_attention_mask = torch.cat((chosen_attention_mask, rejected_attention_mask), dim=0)
                concatenated_attention_mask = concatenated_attention_mask.to(
                    torch.bfloat16) if model.training else concatenated_attention_mask.to(torch.float32)
                all_logits, new_labels = model(
                    input_ids=concatenated_input_ids,
                    input_embeddings=None,
                    attention_mask=concatenated_attention_mask,
                    labels=concatenated_labels,
                    label_smoothing=config.training.label_smoothing,
                    dpo_forward=True
                )
                all_logits = all_logits.to(torch.float32)
                all_logps = get_batch_logps(
                    all_logits,
                    new_labels,
                    label_pad_token_id=-100,
                    max_seq_length=config.dataset.preprocessing.max_seq_length
                )

                chosen_logps = all_logps[:len_chosen]
                rejected_logps = all_logps[len_chosen:]

                chosen_logits = all_logits[:len_chosen]
                rejected_logits = all_logits[len_chosen:]

                chosen_labels = new_labels[:len_chosen]
                rejected_labels = new_labels[len_chosen:]
                return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_labels, rejected_labels)

            with accelerator.accumulate(model):
                (
                    policy_chosen_logps,
                    policy_rejected_logps,
                    policy_chosen_logits,
                    policy_rejected_logits,
                    chosen_labels,
                    rejected_labels,
                ) = concatenated_forward(model, chosen_input_ids, rejected_input_ids, chosen_labels, rejected_labels,
                                         chosen_attention_mask, rejected_attention_mask)
                with torch.no_grad():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                    ) = concatenated_forward(
                        ref_model, chosen_input_ids, rejected_input_ids, chosen_labels, rejected_labels,
                        chosen_attention_mask, rejected_attention_mask
                    )[:2]

                pi_logratios = policy_chosen_logps - policy_rejected_logps
                ref_logratios = reference_chosen_logps - reference_rejected_logps

                pi_logratios = pi_logratios.to(device)
                ref_logratios = ref_logratios.to(device)
                logits = pi_logratios - ref_logratios
                unscaled_dpo_losses = -F.logsigmoid(config.training.beta * logits)
                sft_loss = F.cross_entropy(policy_chosen_logits[:, config.dataset.preprocessing.max_seq_length + 1:].contiguous().view(-1, config.model.showo.vocab_size),
                                           chosen_labels[:, config.dataset.preprocessing.max_seq_length + 1:].contiguous().view(-1),ignore_index=-100)

                if config.training.reward_coef != 0:
                    batch_image_tokens = torch.ones((len(batch_prompt_list), config.model.showo.num_vq_tokens),
                                                    dtype=torch.long, device=device) * mask_id
                    input_ids, _ = uni_prompting((batch_prompt_list, batch_image_tokens), 't2i_gen')

                    if config.training.guidance_scale > 0:
                        uncond_input_ids, _ = uni_prompting(([''] * len(batch_prompt_list), batch_image_tokens),
                                                            't2i_gen')
                        attention_mask = create_attention_mask_predict_next(
                            torch.cat([input_ids, uncond_input_ids], dim=0),
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
                    attention_mask = attention_mask.to(torch.bfloat16)
                    with torch.no_grad():
                        gen_token_ids, _, _ = model.t2i_generate_dpo(
                            input_ids=input_ids,
                            uncond_input_ids=uncond_input_ids,
                            attention_mask=attention_mask,
                            guidance_scale=config.inference.guidance_scale,
                            temperature=config.inference.get("generation_temperature", 1.0),
                            timesteps=config.inference.generation_timesteps,
                            noise_schedule=mask_schedule,
                            noise_type=config.training.get("noise_type", "mask"),
                            seq_len=config.model.showo.num_vq_tokens,
                            uni_prompting=uni_prompting,
                            config=config,
                        )
                    gen_token_ids = torch.clamp(gen_token_ids, max=config.model.showo.codebook_size - 1, min=0)
                    images = vq_model.decode_code(gen_token_ids)

                    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
                    images *= 255.0
                    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                    pil_images = [Image.fromarray(image) for image in images]
                    del gen_token_ids
                    reward_loss = []
                    for per_prompt_index, per_prompt in enumerate(batch_prompt_list):
                        per_prompt = batch_prompt_list[per_prompt_index]
                        per_image = pil_images[per_prompt_index]
                        per_images = [per_image]
                        image_tensors = process_images(per_images, reward_image_processor, reward_model.config)
                        image_tensors = [_image.to(dtype=torch.float16, device=device) for _image in image_tensors]

                        # Prepare interleaved text-image input
                        conv_template = "qwen_1_5"
                        question = f"{DEFAULT_IMAGE_TOKEN}\nThis image is generated by prompt {per_prompt}. Does this image accurately represent the prompt? Please answer yes or no with no explanation.(yes or no in lowercase and no full point)"
                        conv = copy.deepcopy(conv_templates[conv_template])
                        conv.append_message(conv.roles[0], question)
                        conv.append_message(conv.roles[1], None)
                        prompt_question = conv.get_prompt()

                        per_input_ids = tokenizer_image_token(prompt_question, reward_tokenizer, IMAGE_TOKEN_INDEX,
                                                              return_tensors="pt").unsqueeze(0).to(device)
                        image_sizes = [image.size for image in images]
                        succeed = False
                        while not succeed:
                            cont = reward_model.generate(
                                per_input_ids,
                                images=image_tensors,
                                image_sizes=image_sizes,
                                do_sample=True,
                                temperature=1,
                                max_new_tokens=4096,
                                return_dict_in_generate=True,
                                output_scores=True,
                            )

                            scores = torch.cat([score.unsqueeze(1) for score in cont.scores], dim=1)
                            scores = nn.functional.softmax(scores, dim=-1)
                            sequences = cont.sequences
                            selected_prob = torch.gather(scores, dim=-1, index=sequences.unsqueeze(-1)).squeeze(-1)[0][0]
                            succeed = True
                            if reward_tokenizer.convert_ids_to_tokens(sequences[0])[0].lower() not in ['yes', 'no']:
                                succeed = False
                            elif reward_tokenizer.convert_ids_to_tokens(sequences[0])[0].lower() in ['no']:
                                selected_prob = 1 - selected_prob
                        reward_loss.append(selected_prob)
                    reward_loss = -torch.log(torch.clamp(torch.tensor(reward_loss), min=1e-7))
                    loss = unscaled_dpo_losses.mean() + config.training.reward_coef * reward_loss.mean()
                else:
                    loss = config.training.dpo_coef*unscaled_dpo_losses.mean()+config.training.sft_coef*sft_loss

                avg_loss_t2i = accelerator.gather(loss.repeat(config.training.batch_size_t2i)).mean()
                avg_chosen_masking_rate = accelerator.gather(chosen_mask_prob.repeat(config.training.batch_size_t2i)).mean()
                avg_rejected_masking_rate = accelerator.gather(rejected_mask_prob.repeat(config.training.batch_size_t2i)).mean()

                accelerator.backward(loss)

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()

                # log gradient norm before zeroing it
                if (
                        accelerator.sync_gradients
                        and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                        and accelerator.is_main_process
                ):
                    log_grad_norm(model, accelerator, global_step + 1)

                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                if (global_step + 1) % config.experiment.log_every == 0:

                    logs = {
                        "step_loss_t2i": avg_loss_t2i.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "chosen_masking_rate": avg_chosen_masking_rate.item(),
                        "rejected_masking_rate": avg_rejected_masking_rate.item(),
                    }
                    accelerator.log(logs, step=global_step + 1)

                    logger.info(
                        f"Step: {global_step + 1} "
                        f"Loss_t2i: {avg_loss_t2i.item():0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f}"
                    )

                global_step += 1
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(config.experiment.output_dir, safe_serialization=False)

    accelerator.end_training()

def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)

if __name__ == "__main__":
    main()
