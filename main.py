from src.data_ingestion import DataIngestion
from src.data_processing import WikiDataset
from utils.config_manager import config_manager
from utils.logger import setup_logger

log_dir = config_manager.get('paths', {}).get('output', {}).get('logs', './logs')
logger = setup_logger(quiet=True, log_file=f"{log_dir}/training.log")

def main():
    logger.info("=" * 50)
    logger.info("LanguageModeling - Training Pipeline")
    logger.info("=" * 50)

    # 1.
    logger.info('[Data Loading]')

    dt = DataIngestion()
    dataset = dt.download_dataset()
    
    if dataset:
        logger.info(f"Train: {len(dataset['train'])}, Validation: {len(dataset['validation'])}, Test: {len(dataset['test'])}")


    # 2.
    logger.info('[Data Processing]')
    wiki = WikiDataset()
    
    train_loader, val_loader, test_loader = wiki.prepare_dataloaders()

    logger.info(f'Successfully created dataloader...')
    logger.info(f'Train loader: {len(train_loader)} batches')
    logger.info(f'Validation loader: {len(val_loader)} batches')
    logger.info(f'Test loader: {len(test_loader)} batches')

    


if __name__ == "__main__":
    main()