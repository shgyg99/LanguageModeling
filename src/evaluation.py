import torch
from utils.config_manager import config_manager
from utils.common_functions import AverageMeter

def evaluate(model, test_loader, loss_fn, device, metric):
    model.eval()
    loss_eval = AverageMeter()
    metric.reset()

    with torch.inference_mode():
        for inputs, targets in test_loader:
            inputs = inputs.t().to(device)      # [seq_len, batch_size]
            targets = targets.t().to(device)    # [seq_len, batch_size]

            outputs = model(inputs)          # [seq_len, batch_size, vocab_size]

            loss = loss_fn(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))
            loss_eval.update(loss.item(), n=len(targets.reshape(-1)))

            outputs_permuted = outputs.permute(1, 0, 2)  # [batch_size, seq_len, vocab_size]
            targets_permuted = targets.permute(1, 0)      # [batch_size, seq_len]
            
            metric.update(outputs_permuted, targets_permuted)

    return loss_eval.avg, metric.compute().item()