import argparse
from kitsune.trainer import KitsuneTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='JSON file for configuration')
    parser.add_argument('-m', '--model_dir', type=str, required=True, help='Directory to save checkpoints')
    parser.add_argument('-p', '--checkpoint', type=str, default=None, help='Path to starting checkpoint (optional)')
    args = parser.parse_args()

    trainer = KitsuneTrainer(config_path=args.config, model_dir=args.model_dir)
    if args.checkpoint:
        trainer.resume_from_checkpoint(args.checkpoint)
    trainer.train()

if __name__ == '__main__':
    main()
