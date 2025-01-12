import math
import ray
import torch
import transformers
from torch.utils.data import DataLoader

from dataset import get_examples, GSMDataset
from calculator import sample
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GenerationConfig
import random
import re
from tqdm import tqdm
import sys

from train_llama2 import ModelArguments, DataArguments, TrainingArguments

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    if completion.find('\u0000') >= 0:
        completion = completion[0:completion.find('\u0000')]
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        try:
            float(match_str)
        except BaseException:
            return INVALID_ANS
        return match_str
    else:
        return INVALID_ANS


@ray.remote(num_gpus=1)
def parallel_decode(model_path, tokenizer, batch, generation_config):
    # gpu_id = ray.get_gpu_ids()[0]
    # print(gpu_id)
    device = torch.device("cuda")
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()
    
    with torch.no_grad():
        batch_output = model.generate(
            input_ids=[b['q_ids'].cuda() for b in batch],
            attention_mask=[b["q_attention_mask"].cuda() for b in batch],
            generation_config=generation_config,
            return_dict_in_generate=True
        )

    outputs_string = tokenizer.batch_decode(batch_output.sequences, skip_special_tokens=True)
    results = []
    for gold_ans1, pred_ans in zip(batch, outputs_string):
        gold_ans = gold_ans1["example"]["answer"]
        gold_ext = extract_answer(gold_ans)
        pred_ext = extract_answer(pred_ans)
        results.append((gold_ext, pred_ext))
    return results


def main():
    global local_rank
    torch.set_warn_always(False)
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    # model = transformers.AutoModelForCausalLM.from_pretrained(
    #     model_args.model_name_or_path,
    #     config=config,
    #     cache_dir=training_args.cache_dir,
    # )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    print("loading model: ", model_args.model_name_or_path)
    # device = torch.device("cuda")
    # model.to(device)
    print("eval_data_path:", data_args.eval_data_path)
    eval_examples = get_examples(data_args.eval_data_path)
    eval_dset = GSMDataset(tokenizer, eval_examples, loss_on_prefix=data_args.loss_on_prefix)
    eval_loader = DataLoader(eval_dset, batch_size=training_args.per_device_eval_batch_size, shuffle=False, num_workers=4)
    
    ray.init()
    # Divide the eval_loader into chunks based on the number of GPUs
    num_gpus = 2
    print("num_gpus: ", num_gpus, torch.cuda.device_count())
    eval_loader_chunks = list(eval_loader)
    eval_loader_chunks = [eval_loader_chunks[i::num_gpus] for i in range(num_gpus)]

    generation_config = GenerationConfig(
        # temperature=0.7,
        # do_sample=False,
        # num_beams=1,
        max_new_tokens=256,
        # num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    results = [parallel_decode.remote(model_args.model_name_or_path, tokenizer, chunk, generation_config) for chunk in
               eval_loader_chunks]

    # Gather results
    results = ray.get(results)
    all_results = [item for sublist in results for item in sublist]  # Flatten the results

    pred_ans_list = [res[1] for res in all_results]
    gold_ans_list = [res[0] for res in all_results]
    
    print(pred_ans_list)
    print(gold_ans_list)
    
    cor = 0
    invalid = 0
    rg = range(min(len(pred_ans_list), len(gold_ans_list)))
    for i in rg:
        if pred_ans_list[i] != INVALID_ANS and abs(float(pred_ans_list[i]) - float(gold_ans_list[i])) < 1e-4:
            cor += 1
        if pred_ans_list[i] == INVALID_ANS:
            invalid += 1
    print(cor, cor / len(list(rg)))
    print(len(rg), invalid)


if __name__ == "__main__":
    main()
