from generator import DataGenerator, TicksGenerator
import hydra
from typing import Dict, List, Any, Union, Tuple
from omegaconf.dictconfig import DictConfig
from setting import create_setting
from pprint import pprint

import matplotlib.pyplot as plt

import numpy as np
import random


@hydra.main(config_path='config', config_name='config')
def main(cfg):

    setting = create_setting(cfg)

    pprint(setting)

    tick_x, tick_y = TicksGenerator.generate(setting)
    data_x, data_y = DataGenerator.generate(setting)

    fig, ax = plt.subplots(figsize=(5., 5.))

    ax.set_xticklabels(tick_x)
    ax.set_yticklabels(tick_y)

    ax.scatter(data_x, data_y)

    plt.savefig(
        '/Users/nakagawaayato/compe/kaggle/mga/src/7_generate_chart/gen0000/test.png')


if __name__ == '__main__':
    main()
