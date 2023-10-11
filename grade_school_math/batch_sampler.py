import math

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


def main():
    global local_rank
    torch.set_warn_always(False)
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
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

    print("Loading model: ", model_args.model_name_or_path)
    device = torch.device("cuda")
    model.to(device)
    print("Model Loaded")
    
    print("mode:", data_args.eval_data_path)
    eval_examples = get_examples("test.jsonl")
    eval_dset = GSMDataset(tokenizer, eval_examples, loss_on_prefix=data_args.loss_on_prefix)
    eval_loader = DataLoader(eval_dset, batch_size=training_args.per_device_eval_batch_size, shuffle=False, num_workers=4)
    generation_config = GenerationConfig(
        temperature=0.7,
        do_sample=True,
        num_beams=1,
        max_new_tokens=256,
        num_return_sequences=1,
    )
    pred_ans = []
    gold_ans = []
    for batch in tqdm(eval_loader):
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        with torch.no_grad():
            batch_output = model.module.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True
            )
        outputs_string = tokenizer.batch_decode(batch_output, skip_special_tokens=True)
        print(batch_output)
        print(outputs_string)
        # ans_ext = extract_answer(ans)
        # pred_ans.append(ans_ext)
        # gold_ans.append(extract_answer(example["answer"]))
        # print(ans)
        # print(ans_ext, extract_answer(example["answer"]))
        #
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
