import os
import torch
import copy
import warnings
import re
from PIL import Image
import sys
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

warnings.filterwarnings("ignore")

class ImageSelector:
    def __init__(self, model_name="llava_qwen", pretrained="lmms-lab/llava-onevision-qwen2-7b-ov", device="cuda", device_map="auto"):
        # Load model
        llava_model_args = {"multimodal": True}

        overwrite_config = {"image_aspect_ratio": "pad"}
        llava_model_args["overwrite_config"] = overwrite_config
        
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            pretrained, None, model_name, device_map=device_map, **llava_model_args
        )
        self.device = device
        self.model.eval()

    def orm(self, prompt, image_file):
        if not os.path.isfile(image_file) or not image_file.lower().endswith(('png', 'jpg', 'jpeg')):
            raise ValueError(f"Invalid image file: {image_file}")  

        # Load the image
        image = Image.open(image_file)

        # Process the image
        image_tensor = process_images([image], self.image_processor, self.model.config)[0]
        image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

        question = (f"{DEFAULT_IMAGE_TOKEN} This image is generated by a prompt: {prompt[0]}. Does this image accurately represent the prompt? Please answer yes or no without explanation.")

        # Prepare conversation
        conv_template = "qwen_1_5"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        # Input question and image to the model
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        image_size = image.size

        succeed = False
        max_retries = 1
        retry_count = 0

        yes_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        no_token_id = self.tokenizer.convert_tokens_to_ids("no")

        while not succeed and retry_count < max_retries:
            retry_count += 1
            # Generate answer
            with torch.no_grad():
                cont = self.model.generate(
                    input_ids,
                    images=[image_tensor],
                    image_sizes=[image_size],
                    do_sample=True,
                    temperature=1.,
                    max_new_tokens=100,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            sequences = cont.sequences
            cur_reponse = self.tokenizer.convert_ids_to_tokens(sequences[0])[0].lower()
            
            if cur_reponse not in ['yes', 'no']:    break
            else:   succeed = True

            scores = torch.cat([score.unsqueeze(1) for score in cont.scores], dim=1)
            scores = torch.nn.functional.softmax(scores, dim=-1)
            first_token_prob = scores[0, 0] 
            yes_prob = first_token_prob[yes_token_id].item()
              
        if not succeed:
            print("Failed to generate a valid 'yes' or 'no' answer after maximum retries. Reponse:" + cur_reponse)
            return False, 0.

        return (cur_reponse == 'yes'), yes_prob


    def parm(self, prompt, image_file, clear):
        if not os.path.isfile(image_file) or not image_file.lower().endswith(('png', 'jpg', 'jpeg')):
            raise ValueError(f"Invalid image file: {image_file}")  

        # Load the image
        image = Image.open(image_file)

        # Process the image
        image_tensor = process_images([image], self.image_processor, self.model.config)[0]
        image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

        # judge
        if clear == False and self.clarity_judgement(prompt, image_tensor, image) == 'no':
            return False, 1., False 
            # return True, 1., True 
        else:
            potential, no_prob = self.potential_assessment(prompt, image_tensor, image)
            # print(potential, no_prob)
            return potential, no_prob, True  
            # potential, no_score, clear
    
    def clarity_judgement(self, prompt, image_tensor, image):
        # Prepare the question
        question = (f"{DEFAULT_IMAGE_TOKEN} This image is a certain step in the text-to-image generation process with a prompt: {prompt[0]}. It is not the final generated one, and will keep iterating better. Do you think this image can be used to judge whether it has the potential to iterate to the image satisfied the prompt? (The image, which needn't to be confused but can be clear and basically judged the object, can be used to judge the potential) Answer yes or no without explanation.")
        question = f"""{DEFAULT_IMAGE_TOKEN} \n The above image is a certain step  in the text to image generation process. This image is not the final generated one, and will keep iterating better. The image is generated by prompt '{prompt[0]}.'
        Do you think this image can be used to judge whether it has the potential to iterate to the image satisfied the prompt?(The image, which needn't to be confused  but can be clear and basically judged the object, can be used to judge the potential)
        Answer yes or no with no explanation.(yes or no in lowercase and no full point)"""
        # Prepare conversation
        conv_template = "qwen_1_5"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        # Input question and image to the model
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        image_size = image.size

        succeed = False
        max_retries = 1
        retry_count = 0

        yes_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        no_token_id = self.tokenizer.convert_tokens_to_ids("no")

        while not succeed and retry_count < max_retries:
            retry_count += 1
            # Generate answer
            with torch.no_grad():
                cont = self.model.generate(
                    input_ids,
                    images=[image_tensor],
                    image_sizes=[image_size],
                    do_sample=True,
                    temperature=1.,
                    max_new_tokens=100,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            sequences = cont.sequences
            cur_reponse = self.tokenizer.convert_ids_to_tokens(sequences[0])[0].lower()
            
            if cur_reponse not in ['yes', 'no']:    break
            else:   succeed = True

        if not succeed:
            print("Failed to generate a valid 'yes' or 'no' answer after maximum retries.")

        return cur_reponse

    def potential_assessment(self, prompt, image_tensor, image):
        # Prepare the question
        clear_question= f"""{DEFAULT_IMAGE_TOKEN} \n The above image is a certain step  in the text to image generation process. This image is not the final generated one, and will keep iterating better. The image is generated by prompt '{prompt[0]}.'
        Do you think this image can be used to judge whether it has the potential to iterate to the image satisfied the prompt?(The image, which needn't to be confused  but can be clear and basically judged the object, can be used to judge the potential)
        Answer yes or no with no explanation.(yes or no in lowercase and no full point)"""
        question = (f"{DEFAULT_IMAGE_TOKEN} Do you think whether the image has the potential to iterate to the image satisfied the prompt? Please answer yes or no without explanation.")
        question='Do you think whether the image has the potential to iterate to the image satisfied the prompt? Please answer yes or no with no explanation.(yes or no in lowercase and no full point)'
        # Prepare conversation
        conv_template = "qwen_1_5"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], clear_question)
        conv.append_message(conv.roles[1], 'yes')
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        # Input question and image to the model
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        image_size = image.size

        succeed = False
        max_retries = 1
        retry_count = 0

        yes_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        no_token_id = self.tokenizer.convert_tokens_to_ids("no")

        while not succeed and retry_count < max_retries:
            retry_count += 1
            # Generate answer
            with torch.no_grad():
                cont = self.model.generate(
                    input_ids,
                    images=[image_tensor],
                    image_sizes=[image_size],
                    do_sample=True,
                    temperature=1.,
                    max_new_tokens=100,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            sequences = cont.sequences
            cur_reponse = self.tokenizer.convert_ids_to_tokens(sequences[0])[0].lower()
            
            if cur_reponse not in ['yes', 'no']:    break
            else:   succeed = True

            scores = torch.cat([score.unsqueeze(1) for score in cont.scores], dim=1)
            scores = torch.nn.functional.softmax(scores, dim=-1)
            first_token_prob = scores[0, 0] 
            no_prob = first_token_prob[no_token_id].item()
              
        if not succeed:
            print("Failed to generate a valid 'yes' or 'no' answer after maximum retries.")
            return False, 50.

        return ('yes' in cur_reponse), no_prob