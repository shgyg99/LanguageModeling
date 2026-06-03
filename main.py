import torch
import os
from torch import nn, optim
import torchmetrics as tm
from utils.common_functions import set_seed
from src.model import LanguageModel
from src.data_ingestion import DataIngestion
from src.data_processing import WikiDataset
from utils.config_manager import config_manager
from utils.logger import setup_logger
from tqdm import tqdm

log_dir = config_manager.get("paths", {}).get("output", {}).get("logs", "./logs")
train_conf = config_manager.get("training", {})
logger = setup_logger(quiet=True, log_file=f"{log_dir}/training.log")

model_save_dir = config_manager.get("paths", {}).get("models", {}).get("saved", "./models/saved")
os.makedirs(model_save_dir, exist_ok=True)
model_save_path = os.path.join(model_save_dir, "best_model.pt")


def main():
    logger.info("=" * 50)
    logger.info("LanguageModeling - Training Pipeline")
    logger.info("=" * 50)

    # 1. Data Loading
    logger.info("[Data Loading]")
    dt = DataIngestion()
    dataset = dt.download_dataset()

    if dataset:
        logger.info(
            f"Train: {len(dataset['train'])}, Validation: {len(dataset['validation'])}, Test: {len(dataset['test'])}"
        )

    # 2. Data Processing
    logger.info("[Data Processing]")
    wiki = WikiDataset()
    train_loader, val_loader, test_loader = wiki.prepare_dataloaders()

    logger.info(f"Successfully created dataloader...")
    logger.info(f"Train loader: {len(train_loader)} batches")
    logger.info(f"Validation loader: {len(val_loader)} batches")
    logger.info(f"Test loader: {len(test_loader)} batches")

    # 3. Model Training
    logger.info("[Model Training]")
    from src.model import LanguageModel
    from src.train import train_one_epoch
    from src.evaluation import evaluate

    set_seed(config_manager.get("system.random_seed", 42))

    architecture = config_manager.get("model", {}).get("architecture", {})
    device = config_manager.get("system.device")

    # Get vocabulary size from tokenizer
    vocab_size = len(wiki.tokenizer)

    model = LanguageModel(
        vocab_size=vocab_size,
        embedding_dim=architecture.get("embedding_dim", 300),
        hidden_dim=architecture.get("hidden_dim", 1150),
        num_layers=architecture.get("num_layers", 3),
        dropoute=architecture.get("dropoute", 0.1),
        dropouti=architecture.get("dropouti", 0.65),
        dropouth=architecture.get("dropouth", 0.3),
        dropouto=architecture.get("dropouto", 0.4),
        weight_drop=architecture.get("weight_drop", 0.2),
        tie_weights=True,
    ).to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=train_conf.get("learning_rate", 30),
        weight_decay=train_conf.get("weight_decay", 1.2e-6),
        momentum=train_conf.get("momentum", 0.9),
    )

    # Learning rate scheduler (بدون verbose)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    loss_fn = nn.CrossEntropyLoss()
    metric = tm.text.Perplexity().to(device)
    clip = train_conf.get("clip")

    loss_train_hist = []
    loss_valid_hist = []
    metric_train_hist = []
    metric_valid_hist = []

    best_loss_valid = torch.inf
    patience_counter = 0
    patience = train_conf.get("early_stopping", {}).get("patience", 5)

    num_epochs = train_conf.get("num_epochs", 10)

    for epoch in range(1, num_epochs + 1):
        logger.info(f"\nEpoch {epoch}/{num_epochs}")
        logger.info("-" * 30)

        # Train
        model, loss_train, metric_train = train_one_epoch(
            model, train_loader, loss_fn, optimizer, metric, clip, device, epoch
        )

        # Validation
        loss_valid, metric_valid = evaluate(model, val_loader, loss_fn, device, metric)

        # Update scheduler
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(loss_valid)
        new_lr = optimizer.param_groups[0]["lr"]

        if new_lr != old_lr:
            logger.info(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")

        # Store history
        loss_train_hist.append(loss_train)
        loss_valid_hist.append(loss_valid)
        metric_train_hist.append(metric_train)
        metric_valid_hist.append(metric_valid)

        # Logging
        logger.info(
            f"Train Loss: {loss_train:.4f} | Train PPL: {metric_train:.2f} | "
            f"Val Loss: {loss_valid:.4f} | Val PPL: {metric_valid:.2f}"
        )

        # Early stopping and model saving
        if loss_valid < best_loss_valid:
            best_loss_valid = loss_valid
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"✅ New best model saved! (Loss: {loss_valid:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter}/{patience} epochs")

        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break

    # Final evaluation on test set
    logger.info("\n" + "=" * 50)
    logger.info("Final Evaluation on Test Set")
    logger.info("=" * 50)

    # Load best model
    model.load_state_dict(torch.load(model_save_path))
    test_loss, test_metric = evaluate(model, test_loader, loss_fn, device, metric)
    logger.info(f"Test Loss: {test_loss:.4f} | Test PPL: {test_metric:.2f}")

    logger.info("\n✅ Training completed!")


if __name__ == "__main__":
    main()
