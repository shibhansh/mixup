import sys
import json
import pickle
import argparse
import numpy as np
from utils import *
import matplotlib.pyplot as plt


def add_cfg_performance(cfg=''):
    with open(cfg, 'r') as f:
        params = json.load(f)
    list_params, param_settings = get_configurations(params=params)
    per_param_setting_performance = []
    for idx in range(len(param_settings)):
        file = params['data_dir'] + str(idx) + '/0'
        with open(file, 'rb') as f:
            data = pickle.load(f)

        # Online performance
        per_param_setting_performance.append((data[0], data[1]))

    return np.array(per_param_setting_performance)


def main(arguments):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg_file', help="Path of the file containing the parameters of the experiment", type=str,
                            default='cfg/gap_reg.json')
    args = parser.parse_args(arguments)
    cfg_file = args.cfg_file

    with open(cfg_file, 'r') as f:
        params = json.load(f)

    list_params, param_settings = get_configurations(params=params)

    performances = add_cfg_performance(cfg=cfg_file)

    plt.plot(performances[:, 1], performances[:, 0])
    plt.show()


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

