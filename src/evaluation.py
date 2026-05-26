import torch
from utils.config_manager import config_manager
from utils.common_functions import AverageMeter


def evaluate(model, test_loader, loss_fn, device, metric):
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
