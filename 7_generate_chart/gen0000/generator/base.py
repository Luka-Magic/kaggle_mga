from abc import ABCMeta, abstractmethod
from typing import Dict, List, Any, Union, Tuple


class BaseGenerator(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def generate(cls, setting: Dict) -> Any:
        """
            Args: 
                setting (Dict)
        """
