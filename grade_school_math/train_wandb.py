import torch
from dataset import get_examples, GSMDataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from transformers import AdamW, TrainingArguments, Trainer
import argparse


def main(args):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    train_examples = get_examples("train")
    train_dset = GSMDataset(tokenizer, train_examples, loss_on_prefix=args.loss_on_prefix)
    
    config = GPT2Config.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
    
    if args.load_from_ckpt:
        model.load_state_dict(torch.load("model_ckpts/pytorch_model.bin"))
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dset,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    
    trainer.save_model("model_ckpts/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using Trainer class.")
    parser.add_argument("--load_from_ckpt", action="store_true",
                        help="Load the model from checkpoint and continue training.")
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Number of warmup steps for learning rate scheduler.")
    parser.add_argument("--loss_on_prefix", action="store_true",
                        help="Compute loss on prefix (for GSMDataset).")
    
    args = parser.parse_args()
    main(args)
