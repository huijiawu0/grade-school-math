import json
import os
import re
import torch as th


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def get_examples(path):
    path = os.path.join("data/", f"{path}")
    examples = read_jsonl(path)
    
    for ex in examples:
        ex.update(question=ex["question"] + "\n")
        ex.update(answer=ex["answer"] + "<|endoftext|>")
    
    print(f"{path} got {len(examples)} examples")
    return examples


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example["answer"])
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer


class GSMDataset(th.utils.data.Dataset):
    def __init__(self, tokenizer, examples, loss_on_prefix=True):
        self.examples = examples
        self.qa = [ex["question"] + ex["answer"] for ex in self.examples]
        self.input_ids = tokenizer(self.qa,
                                   return_tensors="pt",
                                   padding="max_length",
                                   max_length=tokenizer.model_max_length,
                                   truncation=True).input_ids
        self.loss_on_prefix = loss_on_prefix
        self.targets = self.input_ids.clone()
        self.attention_mask = self.input_ids.ne(tokenizer.pad_token_id),
        self.max_len = max(
            [
                len(self.input_ids[i])
                for i in range(len(self.examples))
            ]
        )
        print("input_ids: ", self.input_ids.shape)
        print(f"1Max tokens: {self.max_len}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return dict(input_ids=self.input_ids[idx], attention_mask=self.attention_mask[idx], labels=self.targets[idx])
        # return dict(input_ids=tokens, attention_mask=mask)
