
import torch
import os
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from arguments import dataset_path, seq_len, seed, batch_size
from functions import data_process, LanguageModelDataset, set_seed


train_iter = open(os.path.join(dataset_path, 'wiki.train.tokens'))
valid_iter = open(os.path.join(dataset_path, 'wiki.valid.tokens'))
test_iter = open(os.path.join(dataset_path, 'wiki.test.tokens'))

"""## 🟠 Build vocabulary and save it"""

tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])
torch.save(vocab, 'vocab.pt')

"""## 🟠 Transform the data """


X_train, y_train = data_process(train_iter, seq_len)
X_valid, y_valid = data_process(valid_iter, seq_len)
X_test, y_test = data_process(test_iter, seq_len)

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape

"""## 🟠 Custom dataset"""

train_set = LanguageModelDataset(X_train, y_train)
valid_set = LanguageModelDataset(X_valid, y_valid)
test_set = LanguageModelDataset(X_test, y_test)



"""## 🟠 Define a dataloader if needed"""

set_seed(seed)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
