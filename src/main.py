import argparse
import yaml
import logging

from src.indexing import Indexing

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['index', 'search'], required=True)
    parser.add_argument('--config', type=str, default='config/indexing.yaml')
    parser.add_argument('--max_docs', type=int, default=None, help='Max documents to process')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.mode == 'index':
        indexing_pipeline = Indexing(config)
        indexing_pipeline.run(max_docs=args.max_docs)
    elif args.mode == 'search':
        logger.info("Search module not implemented yet")  # TODO: добавить модуль поиска


if __name__ == '__main__':
    main()