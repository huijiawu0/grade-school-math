import os
import re

import torch
from transformers.trainer_pt_utils import LabelSmoother

from dataset import read_jsonl
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GenerationConfig
from transformers import GPT2Config, AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


class GSMDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, examples, loss_on_prefix=True):
        self.examples = examples
        self.qns = [ex["question"] for ex in self.examples]
        self.ans = [ex["answer"] for ex in self.examples]
        self.qns = tokenizer(self.qns, padding=False)
        self.ans = tokenizer(self.ans, padding=False)
        self.loss_on_prefix = loss_on_prefix
        self.max_len = max(
            [
                len(self.qns["input_ids"][i]) + len(self.ans["input_ids"][i])
                for i in range(len(self.examples))
            ]
        )
        self.max_qns_len = max(
            [
                len(self.qns["input_ids"][i])
                for i in range(len(self.examples))
            ]
        )
        print(f"Max tokens: {self.max_len}")
        print(f"Max q tokens: {self.max_qns_len}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        qn_tokens = self.qns["input_ids"][idx]
        ans_tokens = self.ans["input_ids"][idx]
        pad_tokens = [0] * (self.max_len - len(qn_tokens) - len(ans_tokens))
        qn_pad_tokens = [0] * (self.max_qns_len - len(qn_tokens))
        qap_tokens = qn_tokens + ans_tokens + pad_tokens
        qp_tokens = qn_tokens + qn_pad_tokens
        mask = (
                ([int(self.loss_on_prefix)] * len(qn_tokens))
                + ([1] * len(ans_tokens))
                + ([0] * len(pad_tokens))
        )
        qn_mask = [1] * len(qn_tokens) + [0] * len(qn_pad_tokens)
        qap_tokens = torch.tensor(qap_tokens)
        mask = torch.tensor(mask)
        qp_tokens = torch.tensor(qp_tokens)
        qn_mask = torch.tensor(qn_mask)
        return dict(
            input_ids=qap_tokens,
            attention_mask=mask,
            q_ids=qp_tokens,
            q_attention_mask=qn_mask,
            examples=self.examples[idx],
        )


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


def eval(model, eval_dataloader, tokenizer):
    generation_config = GenerationConfig(
        # temperature=0.7,
        # do_sample=False,
        # num_beams=1,
        max_new_tokens=256,
        # num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    pred_ans_list = []
    gold_ans_list = []
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            batch_output = model.generate(
                input_ids=batch['q_ids'].to(model.device),
                attention_mask=batch["q_attention_mask"].to(model.device),
                generation_config=generation_config,
                return_dict_in_generate=True
            )
        outputs_string = tokenizer.batch_decode(batch_output.sequences, skip_special_tokens=True)
        for gold_ans, pred_ans in zip(batch['examples']["answer"], outputs_string):
            gold_ext = extract_answer(gold_ans)
            pred_ext = extract_answer(pred_ans)
            gold_ans_list.append(gold_ext)
            pred_ans_list.append(pred_ext)
            # print("GOLD: ", gold_ans)
            # print("PRED: ", pred_ans)
    
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


def get_examples(split):
    path = os.path.join("data/", f"{split}.jsonl")
    examples = read_jsonl(path)
    
    for ex in examples:
        ex.update(question=ex["question"] + "\n")
        ex.update(answer=ex["answer"] + "<|endoftext|>")
    
    print(f"{len(examples)} {split} examples")
    return examples


def main():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.unk_token
    train_examples = get_examples("train")
    train_dset = GSMDataset(tokenizer, train_examples)
    train_loader = DataLoader(train_dset, batch_size=16, shuffle=True)
    eval_examples = get_examples("test")
    eval_dset = GSMDataset(tokenizer, eval_examples)
    eval_loader = DataLoader(eval_dset, batch_size=32, shuffle=False)
    
    device = torch.device("cuda")
    config = GPT2Config.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
    model.to(device)
    model.train()
    optim = AdamW(model.parameters(), lr=1e-5)
    
    num_epochs = 1
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optim,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    
    pbar = tqdm(range(num_training_steps))
    for epoch in range(num_epochs):
        for batch in train_loader:
            optim.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask")}
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs[0]
            loss.backward()
            optim.step()
            lr_scheduler.step()
            pbar.update(1)
            pbar.set_description(f"train_loss: {loss.item():.5f}")
            eval(model, eval_loader, tokenizer)
    
    model.save_pretrained("model_ckpts/")


if __name__ == "__main__":
    main()
