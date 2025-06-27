import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from pathlib import Path

from model import build_transformer

def get_config():
    return {
        "batch_size": 2,
        "num_epochs": 20,
        "head": 2,
        "lr": 10**-4,
        "seq_len": 234,
        "d_model": 88,
        "datasource": 'en_ta',
        "input_lang": "en",
        "output_lang": "ta",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "train_ds": "./en-ta-ds/train.csv",
        "test_ds": "./en-ta-ds/test.csv",
        "val_ds": "./en-ta-ds/val.csv",
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

def get_tokenizer():
    base_tokenizer = tiktoken.get_encoding('gpt2')
    custom_tokens = ["<|startoftext|>", "<|padding|>"]
    custom_token_ids = {
        token: base_tokenizer.n_vocab + i for i, token in enumerate(custom_tokens)
    }
    tokenizer = tiktoken.Encoding(
        name="gpt2_custom",
        pat_str=base_tokenizer._pat_str,
        mergeable_ranks=base_tokenizer._mergeable_ranks,
        special_tokens={**base_tokenizer._special_tokens, **custom_token_ids},
    )
    special_tokens_set = set(custom_tokens) | {"<|endoftext|>"} 

    return tokenizer, special_tokens_set

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer, special_tokens_set, input_lang, output_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer = tokenizer
        self.special_tokens_set = special_tokens_set
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.sos = torch.tensor(tokenizer.encode("<|startoftext|>", allowed_special=special_tokens_set), dtype=torch.int64)
        self.pad = torch.tensor(tokenizer.encode("<|padding|>", allowed_special=special_tokens_set), dtype=torch.int64)
        self.eos = torch.tensor(tokenizer.encode("<|endoftext|>", allowed_special=special_tokens_set), dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def causal_mask(self, size):
        mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
        return mask == 0
    
    def __getitem__(self, idx):
        pair = self.ds[idx]
        input_text = pair[self.input_lang]
        output_text = pair[self.output_lang]

        input_tokens = self.tokenizer.encode(input_text, allowed_special=self.special_tokens_set)
        output_tokens = self.tokenizer.encode(output_text, allowed_special=self.special_tokens_set)

        input_pad = self.seq_len - len(input_tokens) - 2
        output_pad = self.seq_len - len(output_tokens) - 1

        if input_pad < 0 or output_pad < 0:
            raise Exception(f'exceed seq_len input_len: {len(input_tokens)} output_len: {len(output_tokens)}')
        
        encoder_input = torch.cat(
            [
                self.sos,
                torch.tensor(input_tokens, dtype=torch.int64),
                self.eos,
                torch.tensor([self.pad] * input_pad, dtype=torch.int64),
            ],
            dim=0,
        )

        decoder_input = torch.cat(
            [
                self.sos,
                torch.tensor(output_tokens, dtype=torch.int64),
                torch.tensor([self.pad] * output_pad, dtype=torch.int64),
            ],
            dim=0,
        )

        label = torch.cat(
            [
                torch.tensor(output_tokens, dtype=torch.int64),
                self.eos,
                torch.tensor([self.pad] * output_pad, dtype=torch.int64),
            ],
            dim=0,
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad).unsqueeze(0).int() & self.causal_mask(decoder_input.size(0)),
            "label": label,
            "input_text": input_text,
            "output_text": output_text,
        }
    
def get_ds(config, tokenizer, special_tokens_set):
    train_dataset = pd.read_csv(config["train_ds"])[['en', 'ta']].to_dict(orient="records")
    valid_dataset = pd.read_csv(config["val_ds"])[['en', 'ta']].to_dict(orient="records")
    test_dataset = pd.read_csv(config["test_ds"])[['en', 'ta']].to_dict(orient="records")


    train_dataset = BilingualDataset(train_dataset, tokenizer, special_tokens_set, config["input_lang"], config["output_lang"], config["seq_len"])
    valid_dataset = BilingualDataset(valid_dataset, tokenizer, special_tokens_set, config["input_lang"], config["output_lang"], config["seq_len"])
    test_dataset = BilingualDataset(test_dataset, tokenizer, special_tokens_set, config["input_lang"], config["output_lang"], config["seq_len"])

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    return train_dataloader, valid_dataloader, test_dataloader


def get_model(config, tokenizer):
    return build_transformer(tokenizer.n_vocab, config['seq_len'], config["d_model"], 4, 8)

def train_model(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    tokenizer, special_tokens_set = get_tokenizer()

    writer = SummaryWriter(config['experiment_name'])

    train_dataloader, valid_dataloader, test_dataloader = get_ds(config, tokenizer, special_tokens_set)
    model = get_model(config, tokenizer).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    preload = config['preload']

    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.encode("<|startoftext|>", allowed_special=special_tokens_set)[0], label_smoothing=0.1).to(device)
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        torch.cuda.empty_cache()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)

            loss = loss_fn(proj_output.view(-1, tokenizer.n_vocab), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


config = get_config()
train_model(config=config)