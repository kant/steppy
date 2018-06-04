from enum import Enum
from typing import Dict, Any, Type, List

from steppy.base import StepsError, BaseTransformer

Timestamp = float


class Node:
    def __init__(self, *,
                 dependency_timestamps: Dict[str, Timestamp]=None,
                 timestamp: Timestamp=None):
        self._dependency_timestamps = dependency_timestamps
        self._timestamp = timestamp

    def get_dependency_differences(self, new_dependency_timestamps: Dict[str, Timestamp]) -> List:
        return get_dict_differences(self._dependency_timestamps, new_dependency_timestamps)

    def get_version_differences(self, other_node: 'Node') -> List:
        return []

    def get_timestamp(self) -> Timestamp:
        return self._timestamp


class DataNode(Node):
    def __init__(self, *,
                 content: Any=None,
                 transformer_name: str=None,
                 dependency_timestamps: Dict[str, Timestamp]=None,
                 timestamp: Timestamp=None):
        super().__init__(dependency_timestamps=dependency_timestamps,
                         timestamp=timestamp)
        self._content = content
        self._transformer_name = transformer_name,

    def get_content(self) -> Any:
        return self._content

    def get_source_transformer(self):
        return self._transformer_name


class TransformerNode(Node):
    def __init__(self, *,
                 tr_type: Type[BaseTransformer],
                 tr_init_kwargs: Dict[str, Any],
                 dependency_timestamps: Dict[str, Timestamp]=None,
                 timestamp: Timestamp=None):
        super().__init__(dependency_timestamps=dependency_timestamps,
                         timestamp=timestamp)
        self._tr_type = tr_type
        self._tr_init_kwargs = tr_init_kwargs
        self._transformer = tr_type(**tr_init_kwargs)
        self._fitted = False

    def __getstate__(self):
        state = dict(self.__dict__)
        del state['_transformer']
        return state

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)
        self._transformer = self._tr_type(self._tr_init_kwargs)

    def get_version_differences(self, other_node: 'TransformerNode'):
        diffs = []
        if self._tr_type != other_node._tr_type:
            diffs.append("current transformer class: {}, new class: {}".format(
                self._tr_type, other_node._tr_type))
        kwargs_diffs = get_dict_differences(self._tr_init_kwargs, other_node._tr_init_kwargs)
        if len(kwargs_diffs) > 0:
            diffs.append("Initializer argument differences: {}".format(', '.join(kwargs_diffs)))
        return diffs

    def _is_fitted(self):
        return self._fitted

    def fit(self, dependency_timestamps, fit_args: Dict[str, Any]) -> None:
        if self._fitted:
            msg = "Fitting a node for a second time."
            raise StepsError(msg)
        self._transformer.fit(**fit_args)
        self._dependency_timestamps = dependency_timestamps
        self._fitted = True

    def transform(self,
                  tr_name: str,
                  dependency_timestamps: Dict[str, Timestamp],
                  timestamp: Timestamp,
                  tr_args: Dict[str, Any]) -> DataNode:
        result = self._transformer.transform(**tr_args)
        return DataNode(content=result,
                        transformer_name=tr_name,
                        dependency_timestamps=dependency_timestamps,
                        timestamp=timestamp)

    def save_transformer(self, path):
        self._transformer.save(path)

    def load_transformer(self, path):
        self._transformer.load(path)


class Evaluator:
    class GoodValue(Enum):
        HIGH = "high"
        LOW = "low"

    def evaluate(self, input_results: Dict[str, Any]) -> float:
        raise NotImplementedError

    def good_value(self) -> GoodValue:
        raise NotImplementedError


class SelectorNode(Node):
    def __init__(self, *,
                 evaluator,
                 dependency_timestamps: Dict[str, Timestamp]=None,
                 timestamp: Timestamp=None):
        super().__init__(dependency_timestamps=dependency_timestamps,
                         timestamp=timestamp)
        self._evaluator = evaluator

    def get_version_differences(self, other_node: 'SelectorNode'):
        diffs = []
        if self._evaluator.__class__ != other_node._evaluator.__class__:
            diffs.append("current evaluator class: {}, new class: {}".format(
                self._evaluator.__class__, other_node._evaluator.__class__))
        return diffs


def get_dict_differences(curr: dict, new: dict):
    diffs = []
    for key, val in curr.items():
        if key not in new:
            diffs.append("key {} present in current but not in new")
        else:
            new_val = new[key]
            if val != new_val:
                diffs.append("key {} mapped to {} in current, and to {} in new"
                             .format(key, val, new_val))

    for key in new:
        if key not in curr:
            diffs.append("key {} present in new but not in current")

    return diffs
