import sys
import json
import pickle
import argparse
import numpy as np
from utils import *
import matplotlib.pyplot as plt


def add_cfg_performance(cfg='', settings=None):
    with open(cfg, 'r') as f:
        params = json.load(f)
    list_params, param_settings = get_configurations(params=params)
    per_param_setting_performance = []
    if settings is None:
        settings = [i for i in range(len(param_settings))]
    for idx in settings:
        file = params['data_dir'] + str(idx) + '/0'
        with open(file, 'rb') as f:
            data = pickle.load(f)

        # Online performance
        per_param_setting_performance.append((data[0], data[1]))

    return np.array(per_param_setting_performance)


def main(arguments):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg_file', help="Path of the file containing the parameters of the experiment", type=str,
                            default='cfg/l2_gap.json')
    args = parser.parse_args(arguments)
    cfg_file = args.cfg_file

    with open(cfg_file, 'r') as f:
        params = json.load(f)
    list_params, param_settings = get_configurations(params=params)


    performances = add_cfg_performance(cfg=cfg_file, settings=[i for i in range(0, 5)])
    plt.plot(performances[:, 1], performances[:, 0], label=params[list_params[0]][0])

    performances = add_cfg_performance(cfg=cfg_file, settings=[i for i in range(5, 10)])
    plt.plot(performances[:, 1], performances[:, 0], label=params[list_params[0]][1])

    performances = add_cfg_performance(cfg=cfg_file, settings=[i for i in range(10, 15)])
    plt.plot(performances[:, 1], performances[:, 0], label=params[list_params[0]][2])

    performances = add_cfg_performance(cfg=cfg_file, settings=[i for i in range(15, 20)])
    plt.plot(performances[:, 1], performances[:, 0], label=params[list_params[0]][3])

    performances = add_cfg_performance(cfg='cfg/gap_reg.json', settings=[i for i in range(0, 5)])
    plt.plot(performances[:, 1], performances[:, 0], label='0')

    plt.legend()
    plt.show()
    plt.savefig('cmpr.png')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

