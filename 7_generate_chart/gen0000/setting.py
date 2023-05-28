import random
import numpy as np

from omegaconf.dictconfig import DictConfig
from typing import Dict, List, Any, Union, Tuple


def choose_key_by_weight(weighted_dict):
    # 辞書のキーと値をそれぞれ別のリストに格納する
    keys = list(weighted_dict.keys())
    weights = list(weighted_dict.values())

    # 値の総和を計算し、重みを確率として正規化する
    total_weight = sum(weights)
    probabilities = [weight / total_weight for weight in weights]

    # 確率的にキーを選択する
    chosen_key = random.choices(keys, probabilities)[0]
    return chosen_key


def get_n_data(n_data_dict):
    n_min, n_max = n_data_dict.min, n_data_dict.max
    return np.random.randint(n_min, n_max + 1)


def create_setting(cfg: DictConfig) -> Dict:
    setting = {}

    # set chart type
    setting['chart_type'] = choose_key_by_weight(cfg.chart_type.type_weight)

    # chartの設定
    chart_cfg = cfg[setting['chart_type']]

    # set n_data
    setting['n_data'] = get_n_data(chart_cfg.n_data)

    # generate data from 2d or 1d distribution
    setting['dim_distribution'] = '2d' \
        if np.random.rand() < chart_cfg.rate_2d else '1d'
    setting['distribution'] = choose_key_by_weight(chart_cfg.distribution)

    # generate ticks
    ticks_cfg = chart_cfg.ticks
    margin_cfg = chart_cfg.margin
    for ax in ['x', 'y']:
        setting[ax] = {}
        # TODO: categoricalの設定をここに作りif文で分ける
        setting[ax]['value_type'] = 'numerical'
        setting[ax]['num_type'] = choose_key_by_weight(chart_cfg.data_type[ax])

        setting[ax]['n_ticks'] = get_n_data(ticks_cfg.n_ticks)
        digit = int(choose_key_by_weight(ticks_cfg.digit).replace('d', ''))

        # create start and interval
        int_or_float = choose_key_by_weight(ticks_cfg.type)
        if int_or_float == 'int':
            v0, v1 = np.random.randint(1, 10), np.random.randint(1, 10)
            while v0 == v1:
                v1 = np.random.randint(1, 10)
        elif int_or_float == 'float':
            def gen_float():
                return np.round(np.clip(np.random.rand() * 9. + 1., 1.0, 9.9999999), 1)
            v0, v1 = gen_float(), gen_float()
            while v0 == v1:
                v1 = gen_float(1, 9)
        start_flag = choose_key_by_weight(ticks_cfg.start)
        if start_flag == 'zero':
            v0 = 0
        elif start_flag == 'minus':
            v0 *= -1

        setting[ax]['tick_start'] = v0 * 10 ** digit
        setting[ax]['tick_interval'] = v1 * 10 ** digit
        setting[ax]['tick_end'] =  \
            setting[ax]['tick_start'] + \
            setting[ax]['tick_interval'] * \
            (setting[ax]['n_ticks'] - 1)

        # margin
        margin_dict = {}
        for pos in ['bl', 'tr']:
            margin_type = 'small' \
                if np.random.rand() < margin_cfg['rate_small'] else 'large'
            margin_min, margin_max = margin_cfg[margin_type]['min'], margin_cfg[margin_type]['max']
            setting[ax][f'margin_{pos}'] = \
                np.random.rand() * (margin_max - margin_min) + margin_min

    return setting
