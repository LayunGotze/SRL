import os
import argparse
import pickle
import torch
import json
import torch.nn as nn
from torch.utils.data import DataLoader
from esim.data import NLIDataset
import torch.nn as nn
from tqdm import tqdm
from esim.model import ESIM

batch_size=128
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ------------------- Data loading ------------------- #
"""
print(20 * "=", " Preparing for training ", 20 * "=")
print("\t* Loading training data...")
train_file='data/SNLI/train_data.pkl'
with open(train_file, "rb") as pkl:
    train_data = NLIDataset(pickle.load(pkl))

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
"""

valid_file='data/SNLI/dev_data.pkl'
print("\t* Loading validation data...")
with open(valid_file, "rb") as pkl:
    valid_data = NLIDataset(pickle.load(pkl))

valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)

# -------------------- Model definition ------------------- #
embeddings_file='data/SNLI/embeddings.pkl'
print("\t* Building model...")
with open(embeddings_file, "rb") as pkl:
    embeddings = torch.tensor(pickle.load(pkl), dtype=torch.float).to(device)

hidden_size=100
dropout=0.5
num_classes=3

model = ESIM(embeddings.shape[0],
                 embeddings.shape[1],
                 hidden_size,
                 embeddings=embeddings,
                 dropout=dropout,
                 num_classes=num_classes,
                 device=device).to(device)

tqdm_batch_iterator = tqdm(valid_loader)
for batch_index, batch in enumerate(tqdm_batch_iterator):
    premise=batch['premise'].to(device)
    premise_lengths=batch['premise_length'].to(device)
    hypotheses = batch["hypothesis"].to(device)
    hypotheses_lengths = batch["hypothesis_length"].to(device)
    labels = batch["label"].to(device)
    res=model(premise,premise_lengths,hypotheses,hypotheses_lengths)

    break