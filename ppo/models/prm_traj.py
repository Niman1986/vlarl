import torch
from torch import nn
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor  # NOTE: needs transformers >= 4.46.0 
from peft import PeftModel,PeftConfig


class DummyRM(nn.Module):
    def __init__(self, all_args):
        super().__init__()
        self.all_args = all_args
        
    @torch.no_grad()
    def get_reward(self, text_list: list, image_list: list, actions=None) -> np.ndarray:
        step_scores = []
        for i in range(len(text_list)):
            step_scores.append([0.0])
        step_scores = np.array(step_scores)
        
        return step_scores

class QwenProcessRM(nn.Module):
    def __init__(self, all_args):
        super().__init__()
        self.all_args = all_args
        self.prm_model_name_or_path = all_args.prm_model_name_or_path
        self.prm_checkpoint_path = all_args.prm_checkpoint_path
        print(f"prm_base_model_path: {self.prm_model_name_or_path}")
        print(f"prm_checkpoint_path: {self.prm_checkpoint_path}")
        
        # self.good_token = '+'
        # self.bad_token = '-'
        # self.step_tag = '\n\n\n\n\n'

        self.processor = AutoProcessor.from_pretrained(
            self.prm_model_name_or_path,
            # do_rescale=False,
        )  #, add_eos_token=False, padding_side='left')
        # self.tokenizer.pad_token_id = 151655 # "<|image_pad|>"
        # self.candidate_tokens = self.tokenizer.encode(f" {self.good_token} {self.bad_token}") # [488, 481]
        # self.step_tag_id = self.tokenizer.encode(f" {self.step_tag}")[-1] # 76325

        if self.all_args.use_vllm:
            from vllm import LLM, SamplingParams
            # TODO: vllm with lora sampling, ref: https://docs.vllm.ai/en/latest/usage/lora.html
            # FIXME: assert "factor" in rope_scaling needs transformers 0.45, but lLAMA-Factory merge needs 0.46
            # Ref: https://github.com/huggingface/transformers/issues/33401#issuecomment-2395583328
            self.model = LLM(
                model=self.prm_model_name_or_path,
                limit_mm_per_prompt={"image": 10, "video": 10},
            )
            self.sampling_params = SamplingParams(
                temperature=0.1,
                top_p=0.001,
                repetition_penalty=1.05,
                max_tokens=2,
                stop_token_ids=[],
            )
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(self.prm_model_name_or_path, 
                                                            device_map="auto", 
                                                            torch_dtype=torch.bfloat16,
                                                            attn_implementation="flash_attention_2",
                                                            )
            if self.prm_checkpoint_path is not None:
                self.model = PeftModel.from_pretrained(self.model, self.prm_checkpoint_path)
                # self.model.merge_and_unload()     # for critic training
                self.model.print_trainable_parameters()
            self.model.eval()

        # disable_dropout_in_model(self.model)

    def preprocess(self, text_list: list, image_list: list, actions=None) -> np.ndarray:
        for i in range(len(text_list)):
            # extract task instruction from f"In: What action should the robot take to {task_label.lower()}?\nOut:"
            task_label = text_list[i].split("What action should the robot take to ")[1].split("?\nOut:")[0]
            image_list[i] = Image.fromarray(image_list[i])
            # Sample messages for batch inference
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_list[i]},
                        {"type": "text", "text": f"The task is {task_label}, is it completed?"},
                    ],
                }
            ]
            # Preparation for batch inference
            text_list[i] = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        return text_list, image_list

    @torch.no_grad()
    def get_reward(self, text_list: list, image_list: list, actions=None) -> np.ndarray:
        text_list, image_list = self.preprocess(text_list, image_list)

        # image is [0, 255]
        inputs = self.processor(
            text=text_list,
            images=image_list,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        if self.all_args.use_vllm:
            # TODO: batch inference
            llm_inputs = {
                "prompt": text_list,
                "multi_modal_data": {"image": image_list},
            }
            outputs = self.model.generate([llm_inputs], sampling_params=self.sampling_params)
            output_text = outputs[0].outputs[0].text

        else:
            generated_ids  = self.model.generate(   # just output 0 or 1
                **inputs,
                max_new_tokens=2,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        # print(output_text)

        step_scores = [int(text) for text in output_text]
        step_scores = np.array(step_scores)

        return step_scores


if __name__ == "__main__":
    from ppo.conf.config_traj import RLConfig

    all_args = RLConfig()
    
    text_list = ["Hello, World!"]
    image_list = [np.random.rand(3, 224, 224)]

    prm = DummyRM(all_args)
    print(prm.get_reward(text_list, image_list))

    prm = QwenProcessRM(all_args)
    print(prm.get_reward(text_list, image_list))

# python -m train/models/prm_traj
