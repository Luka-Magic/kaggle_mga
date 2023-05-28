from .base import BaseGenerator
from typing import Dict, List, Any, Union, Tuple
import numpy as np
from omegaconf.dictconfig import DictConfig


class TicksGenerator(BaseGenerator):
    @classmethod
    def generate(cls, setting: Dict) -> Tuple[List[Any], List[Any]]:

        axes_setting = {'x': setting['x'], 'y': setting['y']}

        ticks_dict = {}

        for ax, ax_setting in axes_setting.items():
            if ax_setting['value_type'] == 'categorical':
                n_data = setting['n_data']
            elif ax_setting['value_type'] == 'numerical':
                n_ticks = ax_setting['n_ticks']
                interval = ax_setting['tick_interval']
                ticks = [ax_setting['tick_start'] +
                         interval * i for i in range(n_ticks)]
            else:
                ValueError
            ticks_dict[ax] = ticks
        print(ticks_dict)

        return ticks_dict['x'], ticks_dict['y']
