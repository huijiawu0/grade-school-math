import torch as th
from dataset import get_examples, GSMDataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import GPT2Config, AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import argparse


def main(args):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    train_examples = get_examples("train")
    
    # Pass the loss_on_prefix argument to GSMDataset
    train_dset = GSMDataset(tokenizer, train_examples, loss_on_prefix=args.loss_on_prefix)
    
    device = th.device("cuda")
    config = GPT2Config.from_pretrained("gpt2")
    if args.rand_init:
        print("Rand init GPT2 weight")
        model = GPT2LMHeadModel(config)
    else:
        model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
    model.to(device)

    model_checkpoint_path = "%s/pytorch_model.bin" % args.save_path
    optimizer_checkpoint_path = "%s/optimizer_state.pth" % args.save_path
    
    if args.load_from_ckpt:
        model.load_state_dict(th.load(model_checkpoint_path))
        optim = AdamW(model.parameters(), lr=args.learning_rate)
        optim.load_state_dict(th.load(optimizer_checkpoint_path))
    else:
        optim = AdamW(model.parameters(), lr=args.learning_rate)
    
    model.train()
    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True)
    
    num_training_steps = args.num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optim,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    pbar = tqdm(range(num_training_steps))
    for epoch in range(args.num_epochs):
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
    
    model.save_pretrained(args.save_path)
    th.save(optim.state_dict(), optimizer_checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model.")
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
    parser.add_argument("--rand_init", action="store_true",
                        help="Rand init")
    parser.add_argument("--save_path", type=str, default="",
                        help="Model save path")
    
    args = parser.parse_args()
    
    main(args)
