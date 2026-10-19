import copy
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from data.data_manager import DataManager
from utils import model_factory


def train(args):
    seeds = args["seed"]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seed must be a nonempty list")
    results = []
    for seed in seeds:
        config = copy.deepcopy(args)
        config["seed"] = int(seed)
        results.append(_train(config))
    summary = {
        "runs": results,
        "A_Last_mean": float(np.mean([run["A_Last"] for run in results])),
        "A_Last_std": float(np.std([run["A_Last"] for run in results])),
        "A_Avg_mean": float(np.mean([run["A_Avg"] for run in results])),
        "A_Avg_std": float(np.std([run["A_Avg"] for run in results])),
    }
    destination = (
        Path(args.get("output_dir", "logs")) / args["model_name"] / args["dataset"]
    )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _train(args):
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    logs_dir = (
        Path(args.get("output_dir", "logs"))
        / args["model_name"]
        / args["dataset"]
        / str(init_cls)
        / str(args["increment"])
    )
    logs_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args['prefix'].strip() or 'ekpc'}_{args['seed']}_{args['convnet_type']}"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(logs_dir / f"{name}.log", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    for key, value in args.items():
        logging.info("%s: %s", key, value)
    _set_random(args["seed"])
    _set_device(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        data_root=args.get("data_root"),
    )
    model = model_factory.get_model(args["model_name"], args)
    top1, top5 = ([], [])
    for task in range(data_manager.nb_tasks):
        model.incremental_train(data_manager)
        accuracy = model.eval_task()
        model.after_task()
        top1.append(float(accuracy["top1"]))
        top5.append(float(accuracy["top5"]))
        logging.info("Task %s: %s", task, accuracy["grouped"])
        logging.info("CNN top1 curve: %s", top1)
        logging.info("CNN top5 curve: %s", top5)
        logging.info("Average Accuracy (CNN): %.4f", np.mean(top1))
    result = {
        "seed": args["seed"],
        "top1": top1,
        "top5": top5,
        "A_Last": top1[-1],
        "A_Avg": float(np.mean(top1)),
        "class_order": data_manager._class_order,
    }
    (logs_dir / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _set_device(args):
    devices = args["device"]
    if not isinstance(devices, list) or len(devices) != 1:
        raise ValueError("EKPC expects exactly one device, for example [0] or [-1]")
    index = str(devices[0])
    if index in {"-1", "cpu"}:
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use device: [-1] for CPU checks")
        device = torch.device(f"cuda:{int(index)}")
        torch.cuda.set_device(device)
    args["device"] = [device]


def _set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
