import argparse
import json
import os
from pathlib import Path

from trainer import train


def main():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="EKPC class-incremental learning")
    parser.add_argument(
        "--config", default=str(project_dir / "exps/adapter_cifar224.json")
    )
    args = parser.parse_args()
    config_path = Path(args.config).expanduser()
    if not config_path.is_file():
        config_path = project_dir / config_path
    config_path = config_path.resolve()
    with config_path.open() as source:
        config = json.load(source)
    os.chdir(project_dir)
    train(config)


if __name__ == "__main__":
    main()
