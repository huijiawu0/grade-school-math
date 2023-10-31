# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import math
import pathlib
import re
import time
from dataclasses import dataclass, field
from typing import Optional
import torch.nn.functional as F

import torch
import transformers
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Trainer, GenerationConfig
from transformers.trainer_pt_utils import LabelSmoother
from torch import nn, Tensor
from transformers.utils import ModelOutput

from dataset import get_examples, GSMDataset

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


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False
    loss_on_prefix: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig
    
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
            trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()


from transformers import TrainerCallback


class EvaluationAccuracyCallback(TrainerCallback):
    def __init__(self, model, tokenizer, eval_dataloader, generation_config=None):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.eval_dataloader = eval_dataloader
    
    def on_evaluate(self, args, state, control, **kwargs):
        total_loss = 0
        for batch in tqdm(self.eval_dataloader):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch['q_ids'].to(self.model.device),
                    attention_mask=batch["q_attention_mask"].to(self.model.device))
            predictions = outputs.logits  # (batch_size, 1)
            gold_labels = batch['examples']['answer']  # (batch_size)
            gold_labels = gold_labels.float().unsqueeze(1)
            loss = F.binary_cross_entropy_with_logits(predictions, gold_labels)
            total_loss += loss.item() * len(gold_labels)
            
        avg_loss = total_loss / len(self.eval_dataloader.dataset)
        print(f"Evaluation loss: {avg_loss:.4f}")


def evaluate_on_gpu(model, tokenizer, batch, generation_config):
    with torch.no_grad():
        batch_output = model.generate(
            input_ids=batch['q_ids'].cuda(),
            attention_mask=batch["q_attention_mask"].cuda(),
            generation_config=generation_config,
            return_dict_in_generate=True
        )
    outputs_string = tokenizer.batch_decode(batch_output.sequences, skip_special_tokens=True)
    gold_ans_list = []
    pred_ans_list = []
    for gold_ans, pred_ans in zip(batch['examples']["answer"], outputs_string):
        gold_ext = extract_answer(gold_ans)
        pred_ext = extract_answer(pred_ans)
        gold_ans_list.append(gold_ext)
        pred_ans_list.append(pred_ext)
    return gold_ans_list, pred_ans_list


def get_transformer_hidden_size(model: transformers.PreTrainedModel):
    if isinstance(model, transformers.GPT2LMHeadModel):
        hidden_size_attr_name = "n_embd"
    elif isinstance(model, transformers.OPTForCausalLM):
        hidden_size_attr_name = "word_embed_proj_dim"
    elif isinstance(model, transformers.T5ForConditionalGeneration):
        hidden_size_attr_name = "d_model"
    else:
        # Hack to deal with the fact that transformers library changed the LLaMA model name.
        llama_cls = getattr(
            transformers, "LLaMAForCausalLM" if hasattr(transformers, "LLaMAForCausalLM") else "LlamaForCausalLM"
        )
        if isinstance(model, llama_cls):
            hidden_size_attr_name = "hidden_size"
        else:
            raise ValueError(f"Unknown base_model type: {type(model)}")
        from typing import Any, Mapping
    return getattr(model.config, hidden_size_attr_name)


class RewardModelOutput(ModelOutput):
    rewards: Tensor = None


class RewardModel(transformers.PreTrainedModel):
    def __init__(self, config, model, **kwargs):
        super(RewardModel, self).__init__(config)
        self.backbone_model = model
        hidden_size = get_transformer_hidden_size(self.backbone_model)
        reward_head = nn.Linear(hidden_size, 1)
        torch.nn.init.zeros_(reward_head.bias)
        self.reward_head = reward_head.to(next(self.backbone_model.parameters()).device)

    def forward(self, input_ids, attention_mask=None, return_dict=True, **kwargs):
        outputs = self.backbone_model.model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True, **kwargs
        )
        last_hidden_state = outputs.last_hidden_state
        last_hidden_state_at_the_end = last_hidden_state[:, -1, :]
        rewards = self.reward_head(last_hidden_state_at_the_end).squeeze(-1)
        return RewardModelOutput(rewards=rewards) if return_dict else (rewards,)


def train():
    global local_rank
    
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
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
    )

    reward_model = RewardModel(model)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    eval_tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )
    eval_tokenizer.pad_token = eval_tokenizer.unk_token
    start_time = time.time()
    train_examples = get_examples(data_args.data_path)
    train_dset = GSMDataset(tokenizer, train_examples, loss_on_prefix=data_args.loss_on_prefix, cross_entropy=True)
    # eval_examples = get_examples("test.jsonl")
    eval_examples = get_examples("test.jsonl")[:200]
    eval_dset = GSMDataset(eval_tokenizer, eval_examples, loss_on_prefix=data_args.loss_on_prefix, cross_entropy=True)
    eval_dataloader = DataLoader(eval_dset,
                                 batch_size=training_args.per_device_eval_batch_size,
                                 shuffle=False,
                                 num_workers=4)
    end_time = time.time()
    print(f"Data loading took {(end_time - start_time):.2f} seconds.")
    data_module = dict(train_dataset=train_dset, eval_dataset=eval_dset)
    generation_config = GenerationConfig(
        # temperature=0.7,
        # do_sample=False,
        # num_beams=1,
        max_new_tokens=256,
        # num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    trainer = Trainer(
        model=reward_model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
        callbacks=[EvaluationAccuracyCallback(reward_model, eval_tokenizer, eval_dataloader, generation_config)]
    )
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    # Save model
    model.config.use_cache = True
    trainer.save_state()
    # trainer_save_model_safe(trainer)
    trainer.save_model()


if __name__ == "__main__":
    train()
