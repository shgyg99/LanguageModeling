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

            inputs = inputs.t().to(device)
            targets = targets.t().to(device)

            outputs, _ = model(inputs)

            loss = loss_fn(outputs.reshape(-1, outputs.shape[-1]), targets.flatten())

            loss.backward()

            nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm=clip)

            optimizer.step()
            optimizer.zero_grad()

            loss_train.update(loss.item(), n=len(targets))
            metric.update(outputs, targets)

            tepoch.set_postfix(loss=loss_train.avg, metric=metric.compute().item())

    return model, loss_train.avg, metric.compute().item()
