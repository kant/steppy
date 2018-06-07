from enum import Enum
from typing import Dict, Any, Type, List, Union, cast, Tuple
from pathlib import Path
import joblib
import shutil

from steppy.base import BaseTransformer

Timestamp = float
DataPacket = Dict[str, Any]


class Node:
    def __init__(self, *,
                 dependency_timestamps: Dict[str, Timestamp]=None,
                 timestamp: Timestamp=0):
        self._dependency_timestamps = dependency_timestamps or {}
        self._timestamp = timestamp

    def get_dependency_differences(self, new_dependency_timestamps: Dict[str, Timestamp]) -> List:
        return get_dict_differences(self._dependency_timestamps, new_dependency_timestamps)

    def get_version_differences(self, other_node: 'Node') -> List:
        return []

    def set_dependency_timestamps(self, dependency_timestamps: Dict[str, Timestamp]):
        self._dependency_timestamps = dependency_timestamps

    def get_timestamp(self) -> Timestamp:
        return self._timestamp

    def set_timestamp(self, timestamp: Timestamp):
        self._timestamp = timestamp


class DataNode(Node):
    def __init__(self, *,
                 data_path: Union[str, Path],
                 dependency_timestamps: Dict[str, Timestamp]=None,
                 timestamp: Timestamp=0):
        super().__init__(dependency_timestamps=dependency_timestamps, timestamp=timestamp)
        self._data_path = Path(data_path)

    def set_content(self, *,
                    content: Any,
                    timestamp: Timestamp,
                    dependency_timestamps: Dict[str, Timestamp]):
        self.set_timestamp(timestamp)
        self.set_dependency_timestamps(dependency_timestamps)
        self._data_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(content, str(self._data_path))

    def get_content(self) -> Any:
        return joblib.load(str(self._data_path))


class TransformerNode(Node):
    def __init__(self, *,
                 tr_type: Type[BaseTransformer],
                 tr_init_kwargs: Dict[str, Any],
                 tr_path: Union[str, Path],
                 dependency_timestamps: Dict[str, Timestamp]=None,
                 timestamp: Timestamp=0
                 ):
        super().__init__(dependency_timestamps=dependency_timestamps, timestamp=timestamp)
        self._tr_type = tr_type
        self._tr_init_kwargs = tr_init_kwargs
        self._tr_path = Path(tr_path)
        self._fitted = False

    # def reset_transformer(self, timestamp: Timestamp):
    #     self._transformer = self._tr_type(**self._tr_init_kwargs)
    #     self._fitted = False
    #     self._timestamp = timestamp

    def get_version_differences(self, other_node: 'TransformerNode'):
        diffs = []
        if self._tr_type != other_node._tr_type:
            diffs.append("current transformer class: {}, new class: {}".format(
                self._tr_type, other_node._tr_type))
        kwargs_diffs = get_dict_differences(self._tr_init_kwargs, other_node._tr_init_kwargs)
        if len(kwargs_diffs) > 0:
            diffs.append("Initializer argument differences: {}".format(', '.join(kwargs_diffs)))
        return diffs

    def is_fitted(self):
        return self._fitted

    def fit(self, *,
            dependency_timestamps,
            timestamp: Timestamp=None,
            fit_args: Dict[str, Any]
            ) -> None:
        self.set_timestamp(timestamp)
        self.set_dependency_timestamps(dependency_timestamps)
        transformer = self._tr_type(**self._tr_init_kwargs)
        transformer.fit(**fit_args)
        self._fitted = True
        self._tr_path.parent.mkdir(parents=True, exist_ok=True)
        transformer.save(self._tr_path)

    def transform(self, tr_args: Dict[str, Any]) -> DataPacket:
        transformer = self._tr_type(**self._tr_init_kwargs)
        if self._tr_path.exists():
            transformer.load(self._tr_path)
        result = transformer.transform(**tr_args)
        return cast(DataPacket, result)

    def copy(self, *,
             copied_tr_path: Union[str, Path],
             dependency_timestamps: Dict[str, Timestamp]=None,
             timestamp: Timestamp=0):
        shutil.copy(str(self._tr_path), str(copied_tr_path))
        return TransformerNode(tr_type=self._tr_type,
                               tr_init_kwargs=self._tr_init_kwargs,
                               tr_path=self._tr_path,
                               dependency_timestamps=dependency_timestamps,
                               timestamp=timestamp)


class Evaluator:
    class GoodValue(Enum):
        HIGH = "high"
        LOW = "low"

    def evaluate(self, result: DataPacket) -> float:
        raise NotImplementedError

    def good_value(self) -> GoodValue:
        raise NotImplementedError


class SelectorNode(Node):
    def __init__(self, *,
                 evaluator,
                 timestamp: Timestamp=None):
        super().__init__(timestamp=timestamp)
        self._evaluator = evaluator

    def get_version_differences(self, other_node: 'SelectorNode'):
        diffs = []
        if self._evaluator.__class__ != other_node._evaluator.__class__:
            diffs.append("current evaluator class: {}, new class: {}".format(
                self._evaluator.__class__, other_node._evaluator.__class__))
        return diffs

    def select(self, inputs_results: Dict[str, DataPacket]) -> Tuple[str, Dict[str, float]]:
        """
        Args:
            inputs_results: mapping input node's name to its output

        Returns:
            pair containing name of the best input, and dictionary with evaluation of all inputs
        """
        evals = {name: self._evaluator.evaluate(inputs_results[name]) for name in inputs_results}
        best_value = float("-inf") if self._is_high_value_good() else float("inf")
        best_name = None
        for name, val in evals.items():
            if (self._is_high_value_good() and val > best_value) or\
                    (self._is_low_value_good() and val < best_value):
                best_value = val
                best_name = name
        return best_name, evals

    def good_value(self) -> Evaluator.GoodValue:
        return self._evaluator.good_value()

    def _is_high_value_good(self) -> bool:
        return self._evaluator.good_value() == Evaluator.GoodValue.HIGH

    def _is_low_value_good(self) -> bool:
        return self._evaluator.good_value() == Evaluator.GoodValue.LOW


def get_dict_differences(curr: dict, new: dict):
    diffs = []
    for key, val in curr.items():
        if key not in new:
            diffs.append("key '{}' present in current but not in new".format(key))
        else:
            new_val = new[key]
            if val != new_val:
                diffs.append("key '{}' mapped to '{}' in current, and to '{}' in new"
                             .format(key, val, new_val))

    for key in new:
        if key not in curr:
            diffs.append("key '{}' present in new but not in current".format(key))

    return diffs
