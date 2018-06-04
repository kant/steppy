from collections import defaultdict
from time import time
from pathlib import Path
import joblib
from typing import NamedTuple, Type, Dict, List, Union, Any, cast
from enum import Enum
import shutil
import traceback

from steppy.base import StepsError
from steppy.utils import get_logger
from steppy.adapter import Adapter, AdapterError, AllInputs

from steppy.grzes.nodes import Node, DataNode, TransformerNode, SelectorNode, Evaluator, Timestamp
from steppy.grzes.data_equality import data_equals

logger = get_logger()


TransformerDesc = NamedTuple('TransformerDesc', [('name', str), ('type', Type),
                                                 ('init_kwargs', Dict[str, Any])])
SelectorDesc = NamedTuple('SelectorDesc', [('name', str), ('evaluator', Evaluator)])
PathLike = Union[Path, str]


class NodeType(Enum):
    DATA = "data"
    TRANSFORMER = "transformer"
    SELECTOR = "selector"


class Pipeline:
    def __init__(self, working_dir: PathLike):
        self.working_dir = working_dir
        self.data_nodes = {}
        self.transformer_nodes = {}
        self.selector_nodes = {}
        self.node_dicts = {
            NodeType.DATA: self.data_nodes,
            NodeType.TRANSFORMER: self.transformer_nodes,
            NodeType.SELECTOR: self.selector_nodes
        }
        self.agenda = []

    def run(self, input_data: Dict[str, Any], from_scratch: bool=False) -> Dict[str, Any]:
        self._clear()
        self._init_input_nodes(input_data, from_scratch)

        for task, kwargs in self.agenda:
            try:
                task(**kwargs)
            except Exception as e:
                msg = "Error while executing {}, {}".format(task.__name__, kwargs)
                traceback.print_exc()
                raise StepsError(msg) from e

        return {name: node.get_content() for name, node in self.data_nodes.items()}

    def put(self, node_desc: TransformerDesc):
        self.agenda.append((self._create_transformer,
                            dict(tr_desc=node_desc)))

    def fit(self, *,
            input_names: List[str],
            transformer: Union[str, TransformerDesc],
            adapter: Adapter=None) -> None:
        if isinstance(transformer, TransformerDesc):
            self.agenda.append((self._create_transformer,
                                dict(tr_desc=transformer)))
            tr_name = transformer.name
        else:
            tr_name = transformer
        self.agenda.append((self._scheduled_fit,
                            dict(input_names=input_names, tr_name=tr_name, adapter=adapter)))

    def transform(self, *,
                  input_names: List[str],
                  tr_name: str,
                  output_name: str,
                  adapter: Adapter=None) -> None:
        self.agenda.append((self._scheduled_transform,
                            dict(input_names=input_names, tr_name=tr_name,
                                 output_name=output_name, adapter=adapter)))

    def select(self, *,
               selector: Union[str, SelectorDesc],
               selected_name: str,
               input_names: List[str],
               tr_names: List[str]) -> None:
        if isinstance(selector, SelectorDesc):
            self.agenda.append((self._create_selector,
                               dict(sel_desc=selector)))
            sel_name = selector.name
        else:
            sel_name = selector
        self.agenda.append((self._scheduled_select, dict(
            selected_name=selected_name, selector_name=sel_name,
            input_names=input_names, tr_names=tr_names)))

    def _clear(self) -> None:
        self.data_nodes = {}
        self.transformer_nodes = {}
        self.selector_nodes = {}
        self.node_dicts = {
            NodeType.DATA: self.data_nodes,
            NodeType.TRANSFORMER: self.transformer_nodes,
            NodeType.SELECTOR: self.selector_nodes
        }

    def _init_input_nodes(self, input_data, from_scratch=False) -> None:
        timestamp = self._make_timestamp()
        for name, content in input_data.items():
            if not from_scratch:
                try:
                    self._load_node(name, NodeType.DATA)
                    if data_equals(self.data_nodes[name].get_content(), content):
                        logger.info("Input node '{}' up-to-date.".format(name))
                        continue
                    else:
                        logger.info("Input node '{}' out-of-date.".format(name))
                except FileNotFoundError:
                    logger.info("Input node '{}' not found on disk.".format(name))

            logger.info("Creating a new input node '{}'".format(name))
            self.data_nodes[name] = DataNode(content=content,
                                             timestamp=timestamp)
            self._save_node(name)

    def _create_transformer(self, tr_desc: TransformerDesc) -> None:
        if self._is_node_loaded(tr_desc.name):
            msg = "There is already a node named '{}'.".format(tr_desc.name)
            raise StepsError(msg)

        new_node = TransformerNode(tr_type=tr_desc.type,
                                   tr_init_kwargs=tr_desc.init_kwargs,
                                   timestamp=self._make_timestamp())
        try:
            loaded_node = self._load_node(tr_desc.name, NodeType.TRANSFORMER)
            ver_diffs = loaded_node.get_version_differences(new_node)
            if len(ver_diffs) == 0:
                logger.info("Transformer '{}' loaded.".format(tr_desc.name))
                return
            else:
                logger.info("Transformer '{}' out-of-date because transformer definition "
                            "changed: {}".format(tr_desc.name, '; '.join(ver_diffs)))
        except FileNotFoundError:
            logger.info("No saved transformer '{}' found.".format(tr_desc.name))

        logger.info("Creating a new transformer.")
        self.transformer_nodes[tr_desc.name] = new_node
        self._save_node(tr_desc.name)

    def _scheduled_fit(self, input_names, tr_name, adapter) -> None:
        self._raise_if_inputs_dont_exist(input_names)
        if tr_name not in self.transformer_nodes:
            msg = "No such transformer: '{}'.".format(tr_name)
            raise StepsError(msg)

        input_timestamps = self._get_timestamps(input_names)
        tr_node = self.transformer_nodes[tr_name]
        if not tr_node.is_fitted():
            logger.info("Transformer '{}' hasn't been fitted yet.".format(tr_name))
        else:
            dep_diffs = tr_node.get_dependency_differences(input_timestamps)
            if len(dep_diffs) == 0:
                logger.info("Transformer '{}' up-to-date. Skip fitting.".format(tr_name))
                return
            else:
                logger.info("Transformer '{}' out-of-date because of dependency changes: {}"
                            .format(tr_name, '; '.join(dep_diffs)))
                tr_node.reset_transformer(self._make_timestamp())

        logger.info("Fitting transformer '{}'.".format(tr_name))
        self._do_fit(tr_name, input_names, adapter)
        self._save_node(tr_name)

    def _do_fit(self, tr_name, input_names, adapter) -> None:
        inputs_contents = {name: self.data_nodes[name].get_content() for name in input_names}
        input_timestamps = self._get_timestamps(input_names)
        if adapter:
            fit_args = self._adapt(adapter, inputs_contents)
        else:
            fit_args = self._unpack(inputs_contents)

        logger.info("Fitting '{}'.".format(tr_name))
        self.transformer_nodes[tr_name].fit(input_timestamps, fit_args)

    def _scheduled_transform(self, input_names, tr_name, output_name, adapter=None) -> None:
        self._raise_if_inputs_dont_exist(input_names)
        if tr_name not in self.transformer_nodes:
            msg = "No such transformer: '{}'.".format(tr_name)
            raise StepsError(msg)

        dependency_timestamps = self._get_timestamps([tr_name] + input_names)
        try:
            loaded_node = self._load_node(output_name, NodeType.DATA)
            dep_diffs = loaded_node.get_dependency_differences(dependency_timestamps)
            if len(dep_diffs) == 0:
                logger.info("Data '{}' up-to-date, skip transformation.".format(output_name))
                return
            else:
                logger.info("Data '{}' out of date because of dependency changes: {}"
                            .format(output_name, '; '.join(dep_diffs)))
        except FileNotFoundError:
                logger.info("Data '{}' did not exist.".format(output_name))

        logger.info("Transformer '{}' is creating data '{}'.".format(tr_name, output_name))
        new_node = self._do_transform(tr_name, input_names, adapter)
        self.data_nodes[output_name] = new_node
        self._save_node(output_name)

    def _do_transform(self, tr_name: str, input_names: List[str], adapter: Adapter) -> DataNode:
        inputs_contents = {name: self.data_nodes[name].get_content() for name in input_names}
        dependency_timestamps = self._get_timestamps([tr_name] + input_names)
        if adapter:
            tr_args = self._adapt(adapter, inputs_contents)
        else:
            tr_args = self._unpack(inputs_contents)
        return self.transformer_nodes[tr_name].transform(tr_name, dependency_timestamps,
                                                         self._make_timestamp(), tr_args)

    def _create_selector(self, sel_desc: SelectorDesc) -> None:
        if self._is_node_loaded(sel_desc.name):
            msg = "There is already a node named '{}'.".format(sel_desc.name)
            raise StepsError(msg)

        new_node = SelectorNode(evaluator=sel_desc.evaluator,
                                timestamp=self._make_timestamp())
        try:
            loaded_node = self._load_node(sel_desc.name, NodeType.SELECTOR)
            ver_diffs = loaded_node.get_version_differences(new_node)
            if len(ver_diffs) == 0:
                logger.info("Selector '{}' loaded.".format(sel_desc.name))
                return
            else:
                logger.info("Selector '{}' out-of-date because selector definition "
                            "changed: {}".format(sel_desc.name, '; '.join(ver_diffs)))
        except FileNotFoundError:
            logger.info("No saved selector '{}' found.".format(sel_desc.name))

        logger.info("Creating a new selector.")
        self.selector_nodes[sel_desc.name] = new_node
        self._save_node(sel_desc.name)

    def _scheduled_select(self,
                          selector_name: str,
                          selected_name: str,
                          input_names: List[str],
                          tr_names: List[str]
                          ) -> None:
        self._raise_if_inputs_dont_exist(input_names)
        if selector_name not in self.selector_nodes:
            msg = "No such selector: '{}'.".format(selector_name)
            raise StepsError(msg)

        dependency_timestamps = self._get_timestamps([selector_name] + input_names)
        try:
            loaded_node = self._load_node(selected_name, NodeType.TRANSFORMER)
            dep_diffs = loaded_node.get_dependency_differences(dependency_timestamps)
            if len(dep_diffs) == 0:
                logger.info("Transformer '{}' up-to-date, skip selection.".format(selected_name))
                return
            else:
                logger.info("Transformer '{}' out of date because of dependency changes: {}"
                            .format(selected_name, '; '.join(dep_diffs)))
        except FileNotFoundError:
                logger.info("Transformer '{}' not found on disk.".format(selected_name))

        logger.info("Selector '{}' is selecting transformer '{}'.".format(selector_name, selected_name))
        best_tr_name = self._do_select(selector_name=selector_name,
                                       selected_name=selected_name,
                                       input_names=input_names,
                                       tr_names=tr_names)
        self._copy_transformer(best_tr_name, selected_name)
        self._load_node(selected_name, NodeType.TRANSFORMER)

    def _do_select(self, selector_name: str,
                   selected_name: str,
                   input_names: List[str],
                   tr_names: List[str]
                   ) -> str:
        sel_node = self.selector_nodes[selector_name]
        best_value = float("-inf") if sel_node.is_high_value_good() else float("inf")
        best_index = -1
        for i, name in enumerate(input_names):
            data_node = self.data_nodes[name]
            val = sel_node.evaluate(data_node.get_content())
            if (sel_node.is_high_value_good() and val > best_value) or\
                    (sel_node.is_low_value_good() and val < best_value):
                best_value = val
                best_index = i
        return tr_names[best_index]

    def _get_node_path(self, node_name: str) -> Path:
        return Path(self.working_dir) / 'nodes' / '{}.pkl'.format(node_name)

    def _get_transformer_path(self, tr_name: str) -> Path:
        return Path(self.working_dir) / 'transformers' / '{}.sav'.format(tr_name)

    def _is_node_saved(self, node_name: str) -> bool:
        path_node = self._get_node_path(node_name)
        return path_node.exists()

    def _load_node(self, name: str, node_type: NodeType):
        if self._is_node_loaded(name):
            raise RuntimeError("Trying to load an already loaded node.")

        return {
            NodeType.DATA: self._load_data_node,
            NodeType.TRANSFORMER: self._load_transformer_node,
            NodeType.SELECTOR: self._load_selector_node,
        }.get(node_type)(name)

    def _load_data_node(self, name: str):
        path_node = self._get_node_path(name)
        node = joblib.load(str(path_node))
        self.data_nodes[name] = node
        return node

    def _load_transformer_node(self, name):
        path_node = self._get_node_path(name)
        path_tran = self._get_transformer_path(name)
        node = joblib.load(str(path_node))
        node.load_transformer(path_tran)
        self.transformer_nodes[name] = node
        return node

    def _load_selector_node(self, name):
        path_node = self._get_node_path(name)
        node = joblib.load(str(path_node))
        self.selector_nodes[name] = node
        return node

    def _save_node(self, node_name: str):
        logger.info("Saving node '{}'.".format(node_name))
        node = self._get_node(node_name)
        path = self._get_node_path(node_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if node_name in self.transformer_nodes:
            node = cast(TransformerNode, node)
            path_tran = self._get_transformer_path(node_name)
            path_tran.parent.mkdir(parents=True, exist_ok=True)
            node.save_transformer(str(path_tran))
        joblib.dump(node, str(path))

    def _get_node(self, node_name: str) -> Node:
        for node_type, node_dict in self.node_dicts.items():
            if node_name in node_dict:
                return node_dict[node_name]
        print(self.node_dicts)
        raise RuntimeError("Requested unloaded node '{}'.".format(node_name))

    def _adapt(self, adapter: Adapter, inputs_content: AllInputs):
        logger.info('Adapting inputs')
        try:
            return adapter.adapt(inputs_content)
        except AdapterError as e:
            msg = "Error while adapting step"
            raise StepsError(msg) from e

    def _unpack(self, inputs_contents):
        logger.info('Unpacking inputs')
        unpacked_steps = {}
        key_to_input_names = defaultdict(list)
        for input_name, input_dict in inputs_contents.items():
            unpacked_steps.update(input_dict)
            for key in input_dict.keys():
                key_to_input_names[key].append(input_name)

        repeated_keys = [(key, input_names) for key, input_names in key_to_input_names.items()
                         if len(input_names) > 1]
        if len(repeated_keys) == 0:
            return unpacked_steps
        else:
            msg = "Could not unpack inputs. Following keys are present in multiple input nodes:\n"\
                "\n".join(["  '{}' present in inputs {}".format(key, input_names)
                           for key, input_names in repeated_keys])
            raise StepsError(msg)

    def _is_node_loaded(self, node_name):
        for node_type, node_dict in self.node_dicts.items():
            if node_name in node_dict:
                return True
        return False

    def _raise_if_inputs_dont_exist(self, input_names):
        for name in input_names:
            if name not in self.data_nodes:
                msg = "Input node '{}' does not exist".format(name)
                raise StepsError(msg)

    def _get_timestamps(self, node_names):
        return {node_name: self._get_node(node_name).get_timestamp()
                for node_name in node_names}

    def _make_timestamp(self) -> Timestamp:
        return time()

    def _copy_transformer(self, source_name: str, destination_name: str):
        source_node_path = self._get_node_path(source_name)
        source_tran_path = self._get_transformer_path(source_name)
        destination_node_path = self._get_node_path(destination_name)
        destination_tran_path = self._get_transformer_path(destination_name)
        shutil.copy(str(source_node_path), str(destination_node_path))
        shutil.copy(str(source_tran_path), str(destination_tran_path))
