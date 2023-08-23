import argparse
import torch
import json
import os

from datetime import datetime
from scinet.experiments.main_run import SCINetPipeline


if __name__ == "__main__":

    cwd = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(cwd, "config.json")

    with open(cfg_path, "r") as file:
        config = json.load(file)

    args = argparse.Namespace()
    for key, value in config.items():
        setattr(args, key, value)

    if not args.long_term_forecast:
        args.concat_len = args.window_size - args.horizon

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
    torch.backends.cudnn.enabled = True

    exp = SCINetPipeline(args)

    if args.evaluate:
        data = exp._get_data()
        before_evaluation = datetime.now().timestamp()
        if args.stacks == 1:
            rse, rae, correlation = exp.validate(
                data, data.test[0], data.test[1], evaluate=True
            )
        else:
            rse, rae, correlation, rse_mid, rae_mid, correlation_mid = exp.validate(
                data, data.test[0], data.test[1], evaluate=True
            )
        after_evaluation = datetime.now().timestamp()
        print(f"Evaluation took {(after_evaluation - before_evaluation) / 60} minutes")

    elif args.train or args.resume:
        data = exp._get_data()
        before_train = datetime.now().timestamp()
        print("===================Normal-Start=========================")
        normalize_statistic = exp.train()
        after_train = datetime.now().timestamp()
        print(f"Training took {(after_train - before_train) / 60} minutes")
        print("===================Normal-End=========================")
        exp.validate(data, data.test[0], data.test[1], evaluate=True)
    else:
        raise NotImplementedError()
