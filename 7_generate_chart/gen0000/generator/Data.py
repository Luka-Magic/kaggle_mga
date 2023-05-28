from .base import BaseGenerator
from typing import Dict, List, Any, Union, Tuple
import numpy as np


class DataGenerator(BaseGenerator):
    @classmethod
    def generate(cls, setting: Dict) -> Tuple[List[Any], List[Any]]:
        '''
        Args:
            setting (Dict)
        Returns:
            data (Tuple[List[Any], List[Any]])
                data consists of list of x, y
                len(x) == len(y)
        '''
        n_data = setting['n_data']

        # distribution
        if setting['dim_distribution'] == '2d':
            distribution = setting['distribution']
            # 分布からデータの生成
            if distribution == "normal":
                # 正規分布（平均0、標準偏差1）から生成
                data = np.random.normal(0, 1, (2, n_data))
            elif distribution == "uniform":
                # 一様分布（0以上1未満）から生成
                data = np.random.uniform(0, 1, (2, n_data))
            elif distribution == "poisson":
                # ポアソン分布（λ=1）から生成
                data = np.random.poisson(1, (2, n_data))
            elif distribution == "binomial":
                # 二項分布（n=10, p=0.5）から生成
                data = np.random.binomial(10, 0.5, (2, n_data))
            elif distribution == "gamma":
                # ガンマ分布（k=2, θ=2）から生成
                data = np.random.gamma(2, 2, (2, n_data))
            elif distribution == "beta":
                # ベータ分布（α=2, β=5）から生成
                data = np.random.beta(2, 5, (2, n_data))
            else:
                raise ValueError("Invalid distribution specified.")

            data_dict = {}

            # [0, 1]に正規化
            # 列ごとに最小値と最大値を計算
            min_data = np.min(data, axis=1, keepdims=True)
            max_data = np.max(data, axis=1, keepdims=True)
            normalized_data = (data - min_data) / (max_data - min_data)
            data_dict['x'], data_dict['y'] = normalized_data

            # ticksとmarginからデータを生成
            for ax in ['x', 'y']:
                data_start = setting[ax]['tick_start'] + \
                    setting[ax][f'margin_bl']
                data_end = setting[ax]['tick_end'] - setting[ax][f'margin_tr']
                data_dict[ax] = data_dict[ax] * \
                    (data_end - data_start) + data_start
            x, y = data_dict['x'], data_dict['y']

        assert len(x) == len(y)
        assert len(set((x_i, y_i) for x_i, y_i in zip(x, y))) == n_data
        return x, y
