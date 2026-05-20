import torch
from torch import nn
import torchmetrics as tm

dataset_path = 'G:\myProjects\LanguageModeling\dataset'
model_path = 'G:\myProjects\LanguageModeling\metric 99.49.pt'


seed = 8
batch_size = 80
seq_len = 70
embedding_dim = 300
num_layers = 3
hidden_dim = 1150
dropoute = 0.1
dropouti = 0.65
dropouth = 0.3
dropouto = 0.4
weight_drop = 0.
lr = 30
wd = 1.2e-6
momentum = 0.9
clip = 0.25

device = 'cuda' if torch.cuda.is_available() else 'cpu'

loss_fn = nn.CrossEntropyLoss()
metric = tm.text.Perplexity().to(device)

wandb_enable = False
if wandb_enable: #if you want to use WandB set your key and run name below
    WBKey = ''
    wandb_arg_name = ''


