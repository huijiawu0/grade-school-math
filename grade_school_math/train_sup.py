import math

import torch
import torch.nn as nn
from dataset import get_examples, GSMDataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import GPT2Config, AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import argparse



class SimpleTransformer(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(SimpleTransformer, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def main(args):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    train_examples = get_examples("train")
    
    # Pass the loss_on_prefix argument to GSMDataset
    train_dset = GSMDataset(tokenizer, train_examples, loss_on_prefix=args.loss_on_prefix)
    
    device = torch.device("cuda")
    # config = GPT2Config.from_pretrained("gpt2")
    ntokens = tokenizer.vocab_size
    model = SimpleTransformer(ntokens, ninp=768, nhead=12, nhid=3072, nlayers=12)
    model.to(device)

    model_checkpoint_path = "%s/pytorch_model.bin" % args.save_path
    optimizer_checkpoint_path = "%s/optimizer_state.pth" % args.save_path
    
    if args.load_from_ckpt:
        model.load_state_dict(torch.load(model_checkpoint_path))
        optim = AdamW(model.parameters(), lr=args.learning_rate)
        optim.load_state_dict(torch.load(optimizer_checkpoint_path))
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
    torch.save(optim.state_dict(), optimizer_checkpoint_path)


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
    parser.add_argument("--save_path", type=str, default="",
                        help="Model save path")
    
    args = parser.parse_args()
    
    main(args)
