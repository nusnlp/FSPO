import argparse
import json
import re
import os
import string
import copy
import random
import hydra
import numpy as np
import torch
import torch.distributed
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoConfig
from collections import defaultdict

from verl import DataProto
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_local_path_from_hdfs
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.workers.reward_manager import NaiveRewardManager
from verl.workers.rollout.hf_rollout import HFRollout


def load_data(input_path):
    if "halluqa" in input_path:
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        with open(input_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    return white_space_fix(remove_articles(remove_punc(lower(replace_underscore(s)))))


def match_answer(solution_str):
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    if not matches:
        return None
    else:
        answer_text = matches[-1].group(1).strip()
        return normalize_answer(answer_text)


def save_answers(output_path, answers):
    with open(output_path, 'w', encoding='utf-8') as f:
        for answer in answers:
            f.write(json.dumps(answer, ensure_ascii=False) + '\n')


def generate_answer(prompt, tokenizer, model, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    outputs = model.generate(
        input_ids,
        max_length=4096,
        temperature=1.0,
        do_sample=True,
        top_p=0.9
    )

    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=False)
    return response


def load_sharded_model(fsdp_checkpoint_path):
    state_dict = defaultdict(list)
    checkpoint_dir = Path(fsdp_checkpoint_path)

    shard_files = list(checkpoint_dir.glob("model_world_size_*_rank_*.pt"))
    if not shard_files:
        raise ValueError(f"No checkpoint files found in {fsdp_checkpoint_path}")

    pattern = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt")
    world_sizes = set()
    for file in shard_files:
        match = pattern.match(file.name)
        if match:
            world_sizes.add(int(match.group(1)))

    if len(world_sizes) != 1:
        raise ValueError(
            f"Inconsistent world_size found in checkpoint files: {world_sizes}"
        )

    world_size = world_sizes.pop()
    print(f"Found checkpoints with world_size = {world_size}")

    for rank in range(world_size):
        filepath = checkpoint_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        if not filepath.exists():
            raise ValueError(f"Missing shard file: {filepath}")

        print(f"Loading shard: {filepath}")
        shard_dict = torch.load(filepath, weights_only=False)

        for key, value in shard_dict.items():
            if hasattr(value, "to_local"):
                value = value.to_local()
            state_dict[key].append(value)

    consolidated_state_dict = {}
    for key in state_dict:
        try:
            consolidated_state_dict[key] = torch.cat(state_dict[key], dim=0)
        except (RuntimeError, TypeError):
            consolidated_state_dict[key] = state_dict[key][0]
            print(
                f"Parameter '{key}' does not need concatenation, using first shard value"
            )

    return consolidated_state_dict


def initialize_model_and_tokenizer(local_path, trust_remote_code=True, torch_dtype=torch.bfloat16):
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
    actor_module = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=local_path,
        torch_dtype=torch_dtype,
        config=actor_model_config,
        attn_implementation="flash_attention_2",
        trust_remote_code=trust_remote_code,
    )

    return tokenizer, actor_module


def main():
    parser = argparse.ArgumentParser(description='Generate answers using HuggingFace model.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to test data file (JSONL format)')
    parser.add_argument('--output_path', type=str, required=True, help='Path to generated text')
    parser.add_argument('--model_type', type=str, required=True, help='Type of the model')
    parser.add_argument('--local_path', type=str, required=True, help='Path or name of original model')
    parser.add_argument('--model_path', type=str, required=True, help='Path or name of HuggingFace model')
    parser.add_argument('--gpu_id', type=int, required=True, help='id of the GPU to use')

    args = parser.parse_args()

    device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu')

    if args.model_type == 'qwen-base':
        prefix = """A conversation between User and Assistant. The user asks a question, and the assistant solves it. 
        The assistant first thinks about the reasoning process in the mind and then provides the user with the final 
        answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, 
        respectively. For example, <think> reasoning process here </think> <answer> answer here </answer>. The Assistant 
        shows the reasoning process within <think> </think> tags, and ONLY return the FINAL ANSWER within <answer> </answer> 
        tags. For example: <answer> Kim Marton </answer>.\n\nUser: {question}\nAssistant: Let me solve this step by step. <think>"""
    elif args.model_type == 'qwen-instruct':
        prefix = """<|im_start|>system\nYou are a helpful assistant. Given a question, you need to first think about 
        the reasoning process in the mind and then provide the final answer. The reasoning process and answer are enclosed 
        within <think> </think> and <answer> </answer> tags, respectively. For example, <think> reasoning process 
        here </think> <answer> answer here </answer>. You must show the reasoning process within <think> </think> tags, 
        and ONLY return the FINAL ANSWER within <answer> </answer> tags. For example: <answer> Kim Marton </answer>.<|im_end|>\n<|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    elif args.model_type == 'llama-instruct':
        prefix = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant. 
        Given a question, you need to first think about the reasoning process in the mind and then provide the final 
        answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, 
        respectively. For example, <think> reasoning process here </think> <answer> answer here </answer>. You must show 
        the reasoning process within <think> </think> tags, and ONLY return the FINAL ANSWER within <answer> </answer> tags. For example: <answer> Kim Marton </answer>.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{question}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nLet me solve this step by step.\n<think>"""
    else:
        raise NotImplementedError

    print(f'Loading model {args.model_path}...')
    tokenizer, model = initialize_model_and_tokenizer(args.local_path)
    data = load_data(args.data_path)

    for step in os.listdir(args.model_path):
        ckpt_path = os.path.join(args.model_path, step, "actor")

        # Loading FSDP checkpoint (optional: these three lines can be skipped. Prerequisite: actor_module must be preloaded)
        state_dict = load_sharded_model(ckpt_path)
        model.load_state_dict(state_dict)
        model.to(torch.bfloat16)
        model.to(device)

        output = []
        for item in tqdm(data, desc='Generating answers'):
            question = item['question']
            prompt = prefix.format(question=question)
            response = generate_answer(prompt, tokenizer, model, device)
            item["predicted_answer"] = response
            output.append(item)

        output_file = os.path.join(args.output_path, args.model_type + "-" + step + ".json")
        save_answers(output_file, output)

if __name__ == '__main__':
    main()
