import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torchtext.data.utils import get_tokenizer
from tqdm import tqdm
from arguments import device, clip
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def num_trainable_params(model):
  nums = sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6
  return nums

def set_seed(seed):
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)

class LanguageModelDataset(Dataset):

  def __init__(self, inputs, targets):
    self.inputs = inputs
    self.targets = targets

  def __len__(self):
    return self.inputs.shape[0]

  def __getitem__(self, idx):
    return self.inputs[idx], self.targets[idx]
  
def data_process(raw_text_iter, seq_len, tokenizer, vocab):
  data = torch.cat([torch.LongTensor(vocab(tokenizer(line))) for line in raw_text_iter])

  M = len(data) // seq_len

  r = len(data) % seq_len
  data = torch.cat((data, torch.LongTensor([0]))) if r==0 else data

  inputs = data[:M*seq_len]
  targets = data[1:M*seq_len+1]

  inputs = inputs.reshape(-1, seq_len)
  targets = targets.reshape(-1, seq_len)

  return inputs, targets

class WeightDrop(torch.nn.Module):

  def __init__(self, module, weights, dropout=0):
    super(WeightDrop, self).__init__()
    self.module = module
    self.weights = weights
    self.dropout = dropout
    self._setup()

  def widget_demagnetizer_y2k_edition(*args, **kwargs):
    return

  def _setup(self):
    if issubclass(type(self.module), torch.nn.RNNBase):
      self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

      for name_w in self.weights:
        print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
        w = getattr(self.module, name_w)
        del self.module._parameters[name_w]
        self.module.register_parameter(name_w + '_raw', nn.Parameter(w.data))

  def _setweights(self):
    for name_w in self.weights:
      raw_w = getattr(self.module, name_w + '_raw')
      w = None
      # w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
      mask = torch.nn.functional.dropout(torch.ones_like(raw_w), p=self.dropout, training=True) * (1 - self.dropout)
      setattr(self.module, name_w, raw_w * mask)

  def forward(self, *args):
    self._setweights()
    return self.module.forward(*args)

def embedded_dropout(embed, words, dropout=0.1, scale=None):
  if dropout:
    mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(
        embed.weight) / (1 - dropout)
    masked_embed_weight = mask * embed.weight
  else:
    masked_embed_weight = embed.weight
  if scale:
    masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

  padding_idx = embed.padding_idx
  if padding_idx is None:
    padding_idx = -1

  embedding = torch.nn.functional.embedding(words, masked_embed_weight,
                                            padding_idx, embed.max_norm, embed.norm_type,
                                            embed.scale_grad_by_freq, embed.sparse)
  return embedding

class LockedDropout(nn.Module):
  def __init__(self):
    super(LockedDropout, self).__init__()

  def forward(self, x, dropout):
    if not self.training or not dropout:
      return x
    m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
    mask = m.requires_grad_(False) / (1 - dropout)
    mask = mask.expand_as(x)
    return mask * x

"""🔰 AWD-LSTM Language Model"""

class LanguageModel(nn.Module):

  def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers,
               dropoute=0.2, dropouti=0.2, dropouth=0.2, dropouto=0.2,
               weight_drop=0.2):
    super().__init__()
    self.num_layers = num_layers
    self.hidden_dim = hidden_dim
    self.embedding_dim = embedding_dim

    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.embedding.weight.data.uniform_(-0.1, 0.1)

    self.lstms = []
    self.lstms.append(nn.LSTM(embedding_dim, hidden_dim, num_layers=1, dropout=0, batch_first=False))
    self.lstms.append(nn.LSTM(hidden_dim, hidden_dim, num_layers=1, dropout=0, batch_first=False))
    self.lstms.append(nn.LSTM(hidden_dim, embedding_dim, num_layers=1, dropout=0, batch_first=False))
    if weight_drop > 0:
      self.lstms = [WeightDrop(lstm, ['weight_hh_l0'], dropout=weight_drop) for lstm in self.lstms]
    self.lstms = nn.ModuleList(self.lstms)

    self.fc = nn.Linear(embedding_dim, vocab_size)

    self.fc.weight = self.embedding.weight

    self.lockdrop = LockedDropout()
    self.dropoute = dropoute
    self.dropouti = dropouti
    self.dropouth = dropouth
    self.dropouto = dropouto
    # print(dropoute, dropouti, dropouth, dropouto)

  def forward(self, src):
    embedding = embedded_dropout(self.embedding, src, dropout=self.dropoute if self.training else 0)
    embedding = self.lockdrop(embedding, self.dropouti)

    new_hiddens = []
    for l, lstm in enumerate(self.lstms):
      embedding, _ = lstm(embedding)
      if l != self.num_layers-1:
        embedding = self.lockdrop(embedding, self.dropouth)

    embedding = self.lockdrop(embedding, self.dropouto)

    prediction = self.fc(embedding)
    return prediction

def train_one_epoch(model, train_loader, loss_fn, optimizer, metric, epoch=None):
  
  model.train()
  loss_train = AverageMeter()
  metric.reset()

  with tqdm.tqdm(train_loader, unit='batch') as tepoch:
    for inputs, targets in tepoch:
      if epoch:
        tepoch.set_description(f'Epoch {epoch}')

      inputs = inputs.t().to(device)
      targets = targets.t().to(device)

      outputs = model(inputs)

      loss = loss_fn(outputs.reshape(-1, outputs.shape[-1]), targets.flatten())

      loss.backward()

      nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm=clip)

      optimizer.step()
      optimizer.zero_grad()

      loss_train.update(loss.item(), n=len(targets))
      metric.update(outputs, targets)

      tepoch.set_postfix(loss=loss_train.avg, metric=metric.compute().item())

  return model, loss_train.avg, metric.compute().item()

def evaluate(model, test_loader, loss_fn, metric):
  model.eval()
  loss_eval = AverageMeter()
  metric.reset()

  with torch.inference_mode():
    for inputs, targets in test_loader:
      inputs = inputs.t().to(device)
      targets = targets.t().to(device)

      outputs = model(inputs)

      loss = loss_fn(outputs.reshape(-1, outputs.shape[-1]), targets.flatten())
      loss_eval.update(loss.item(), n=len(targets))

      metric(outputs, targets)

  return loss_eval.avg, metric.compute().item()

def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, num_pred=4, seed=None):
  if seed is not None:
    torch.manual_seed(seed)


  itos = vocab.get_itos()
  preds = []
  for _ in range(num_pred):
    seq = prompt
    indices = vocab(tokenizer(seq))
    itos = vocab.get_itos()
    for i in range(max_seq_len):
      src = torch.LongTensor(indices).to(device)
      with torch.no_grad():
        prediction = model(src)

      probs = torch.softmax(prediction[-1]/temperature, dim=0)

      idx = vocab['<ukn>']
      while idx == vocab['<ukn>']:
        idx = torch.multinomial(probs, num_samples=1).item()

      token = itos[idx]
      seq += ' ' + token

      if idx == vocab['.']:
        break

      indices.append(idx)
    preds.append(seq)

  return preds