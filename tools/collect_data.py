import os
import os.path as osp
import re
import argparse
import numpy as np
from pdb import set_trace

def parse_args():
    parser = argparse.ArgumentParser(description='Experiment summary parser')
    parser.add_argument('--save_dir', default='checkpoints', type=str)
    parser.add_argument('--exp_format', type=str)
    return parser.parse_args()


def read_num(saveDir, exp):
    import json
    path = osp.join(saveDir, exp, 'all_results.json')
    if not osp.isfile(path):
        return -1
    with open(path, 'r') as file:
        result = json.load(file)

    return result["eval_accuracy"]

def main():
    args = parse_args()
    exps_all = os.listdir(args.save_dir)
    exps_select = []
    # set_trace()
    for exp in exps_all:
        if re.match(args.exp_format, exp) is not None:
            exps_select.append(exp)

    numbers = []
    for exp in exps_select:
        acc = read_num(args.save_dir, exp)
        if acc > 0:
            print("{}: {}".format(exp, acc))
            numbers.append(acc)
        else:
            print("read fail for {}".format(exp))

    if len(numbers) > 0:
        print("mean is {}, std is {} for {}".format(np.mean(np.array(numbers)), np.std(np.array(numbers)), numbers))


if __name__ == "__main__":
    main()
