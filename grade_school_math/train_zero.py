import re

import torch
from transformers.trainer_pt_utils import LabelSmoother

from dataset import get_examples, GSMDataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GenerationConfig
from transformers import GPT2Config, AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from torch.utils.data import DataLoader


IGNORE_TOKEN_ID = LabelSmoother.ignore_index
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


def main():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    train_examples = get_examples("train.jsonl")
    train_dset = GSMDataset(tokenizer, train_examples)
    train_loader = DataLoader(train_dset, batch_size=16, shuffle=True)
    eval_examples = get_examples("test.jsonl")
    eval_dset = GSMDataset(tokenizer, eval_examples)
    eval_loader = DataLoader(eval_dset, batch_size=32, shuffle=False)

    device = torch.device("cuda")
    config = GPT2Config.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
    model.to(device)
    model.train()
    optim = AdamW(model.parameters(), lr=1e-5)

    num_epochs = 20
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
            batch = {k: v.to(device) for k, v in batch.items()}
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
