import torch
import torch.optim as optim
import wandb
from arguments import WBKey, seed, embedding_dim, num_layers, hidden_dim, dropoute, dropouth, dropouti, dropouto, device, weight_drop
from arguments import wd, momentum, batch_size, seq_len, clip, wandb_enable, wandb_arg_name
from arguments import model_path, loss_fn, metric
from functions import set_seed, LanguageModel, train_one_epoch, evaluate
from dataset import train_loader, valid_loader

wandb.login(key=WBKey)
torch.cuda.empty_cache()

"""🔰 Define model."""

# set_seed(seed)
# model = LanguageModel(vocab_size=len(vocab), embedding_dim=embedding_dim,
#                       hidden_dim=hidden_dim, num_layers=num_layers,
#                       dropoute=dropoute, dropouti=dropouti,
#                       dropouth=dropouth, dropouto=dropouto,
#                       weight_drop=weight_drop).to(device)

model = torch.load(model_path).to(device)

"""🔰 Define optimizer and Set learning rate and weight decay."""

set_seed(seed)

lr = 0.5
# wd = 1e-6
# momentum = 0.9

optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
# optimizer = optim.SGD([{'params': model.embedding.parameters(), 'lr': 0.1*lr},
#                        {'params': model.lstms.parameters(), 'lr': lr}],
#                       weight_decay=wd, momentum=momentum)

"""🔰 Initialize `wandb`"""

if wandb_enable:
  wandb.init(
      project='LM-AWD-LSTM',
      name=wandb_arg_name,
      config={
          'lr': lr,
          'momentum': momentum,
          'batch_size': batch_size,
          'seq_len': seq_len,
          'hidden_dim': hidden_dim,
          'embedding_dim': embedding_dim,
          'num_layers': num_layers,
          'dropout_embed': dropoute,
          'dropout_in_lstm': dropouti,
          'dropout_h_lstm': dropouth,
          'dropout_out_lstm': dropouto,
          'clip': clip,
      }
  )

"""🔰 Write code to train the model for `num_epochs` epoches."""

loss_train_hist = []
loss_valid_hist = []

metric_train_hist = []
metric_valid_hist = []

best_loss_valid = torch.inf
epoch_counter = 0

set_seed(seed)
num_epochs = 30

for epoch in range(1, num_epochs+1):
  # Train
  model, loss_train, metric_train = train_one_epoch(model,
                                                    train_loader,
                                                    loss_fn,
                                                    optimizer,
                                                    metric,
                                                    epoch)
  # Validation
  loss_valid, metric_valid = evaluate(model,
                                      valid_loader,
                                      loss_fn,
                                      metric)

  loss_train_hist.append(loss_train)
  loss_valid_hist.append(loss_valid)

  metric_train_hist.append(metric_train)
  metric_valid_hist.append(metric_valid)

  if loss_valid < best_loss_valid:
    torch.save(model, model_path)
    best_loss_valid = loss_valid
    print('Model Saved!')

  print(f'Valid: Loss = {loss_valid:.4}, Metric = {metric_valid:.4}')
  print()

  if wandb_enable:
    wandb.log({"metric_train": metric_train, "loss_train": loss_train,
                "metric_valid": metric_valid, "loss_valid": loss_valid})

  epoch_counter += 1

wandb.finish()

