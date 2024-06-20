import logging

from .pipeline import Pipeline
from .config import get_config_from_command


def run_UADB(args_us, X, y, y_noisy, dataloader, dim, outlier_score):
    print("Running UADB")
    config = get_config_from_command()
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
    )

    pipeline = Pipeline(config, X, y, outlier_score, args_us)

    pipeline.boost_co_train(args_us)