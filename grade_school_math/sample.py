import torch as th
from dataset import get_examples, GSMDataset
from calculator import sample
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import random
import re
from tqdm import tqdm

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
    device = th.device("cuda")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("model_ckpt_prefix_lr1e3")
    model.to(device)
    print("Model Loaded")
    
    test_examples = get_examples("train")[:100]
    
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
