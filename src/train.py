import tqdm
from torch import nn
from utils.common_functions import AverageMeter


def train_one_epoch(model, train_loader, loss_fn, optimizer, metric, clip, device="cpu", epoch=None):
    model.train()
    loss_train = AverageMeter()
    metric.reset()

    with tqdm.tqdm(train_loader, unit="batch") as tepoch:
        for inputs, targets in tepoch:
            if epoch:
                tepoch.set_description(f"Epoch {epoch}")

            inputs = inputs.t().to(device)  # [seq_len, batch_size]
            targets = targets.t().to(device)  # [seq_len, batch_size]

            outputs = model(inputs)  # [seq_len, batch_size, vocab_size]

            loss = loss_fn(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))

            loss.backward()
            nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm=clip)
            optimizer.step()
            optimizer.zero_grad()

            loss_train.update(loss.item(), n=len(targets.reshape(-1)))

            outputs_permuted = outputs.permute(1, 0, 2)  # [batch_size, seq_len, vocab_size]
            targets_permuted = targets.permute(1, 0)  # [batch_size, seq_len]

            metric.update(outputs_permuted, targets_permuted)

            tepoch.set_postfix(loss=loss_train.avg, metric=metric.compute().item())

    return model, loss_train.avg, metric.compute().item()
