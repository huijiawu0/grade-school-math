import math

import torch as th
import transformers

from dataset import get_examples, GSMDataset
from calculator import sample
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import random
import re
from tqdm import tqdm
import sys

from grade_school_math.train_llama2 import ModelArguments, DataArguments, TrainingArguments

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


def main():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(training_args)
    local_rank = training_args.local_rank

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    model_name = sys.argv[1]
    print("Loading model: ", model_name)
    device = th.device("cuda")
    model.to(device)
    print("Model Loaded")
    
    mode = sys.argv[2]
    print("mode:", mode)
    test_examples = get_examples(mode)[:100]

    # random_example = random.choice(test_examples)
    # qn = random_example["question"]
    # qn = test_examples[2]["question"]
    sample_len = 150
    
    pred_ans = []
    gold_ans = []
    for example in tqdm(test_examples):
        qn = example["question"]
        ans = sample(model, qn, tokenizer, device, sample_len)
        ans_ext = extract_answer(ans)
        pred_ans.append(ans_ext)
        gold_ans.append(extract_answer(example["answer"]))
        print(ans)
        print(ans_ext, extract_answer(example["answer"]))
    
    print(pred_ans)
    print(gold_ans)
    
    cor = 0
    invalid = 0
    rg = range(min(len(pred_ans), len(gold_ans)))
    for i in rg:
        if pred_ans[i] != INVALID_ANS and abs(float(pred_ans[i]) - float(gold_ans[i])) < 1e-4:
            cor += 1
        if pred_ans[i] == INVALID_ANS:
            invalid += 1
    print(cor, cor / len(list(rg)))
    print(len(rg), invalid)


if __name__ == "__main__":
    main()
