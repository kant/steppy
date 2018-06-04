import numpy as np

from steppy.base import BaseTransformer

from steppy.grzes.pipeline import Pipeline, TransformerDesc, SelectorDesc
from steppy.grzes.nodes import Evaluator

from ..steppy_test_utils import CACHE_DIRPATH


class CenteringTransformer(BaseTransformer):
    def __init__(self):
        super().__init__()
        self.mean = None
        self.id = np.random.randint(10**12)

    def fit(self, numbers):
        self.mean = np.mean(numbers)

    def transform(self, numbers):
        if self.mean is None:
            raise RuntimeError("Running transform before fit")
        return {'centered': numbers - self.mean, 'id': self.id}


class ConstantTransformer(BaseTransformer):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.id = np.random.randint(10**12)

    def transform(self, *args, **kwargs):
        return {'value': self.c, 'id': self.id}


class DummyEvaluator(Evaluator):
    def evaluate(self, input_results):
        return input_results['value']

    def good_value(self):
        return Evaluator.GoodValue.HIGH


def simple_pipeline():
    p = Pipeline(CACHE_DIRPATH)
    p.fit(input_names=['train'],
          transformer=TransformerDesc('centerer', CenteringTransformer, {}))
    p.transform(input_names=['valid'],
                tr_name='centerer',
                output_name='transformed')
    return p


def simple_data():
    return {
        'train': {
            'numbers': np.array([1, 2, 3])
        },
        'valid': {
            'numbers': np.array([0, 5, 3])
        }
    }


def test_simple_pipeline():
    p = simple_pipeline()
    data = simple_data()
    results = p.run(data)
    expected = np.array([0, 5, 3]) - np.mean(data['train']['numbers'])
    assert np.array_equal(results['transformed']['centered'], expected)


def test_transformer_not_refitted_for_same_data():
    p = simple_pipeline()
    data = simple_data()
    results = p.run(data)
    original_id = results['transformed']['id']

    p = simple_pipeline()
    results = p.run(data)
    assert results['transformed']['id'] == original_id


def test_transformer_refitted_for_different_data():
    p = simple_pipeline()
    data = simple_data()
    results = p.run(data)
    original_id = results['transformed']['id']

    p = simple_pipeline()
    data['train'] = {'numbers': np.array([2, -1, 1])}
    results = p.run(data)
    assert results['transformed']['id'] != original_id


def selector_pipeline():
    lst = [3, 1, 4, 2]
    p = Pipeline(CACHE_DIRPATH)
    for c in lst:
        tr_name = 'const_{}'.format(c)
        out_name = 'output_{}'.format(c)
        p.put(TransformerDesc(tr_name, ConstantTransformer, dict(c=c)))
        p.transform(input_names=['train'],
                    tr_name=tr_name,
                    output_name=out_name)
    p.select(selector=SelectorDesc('selector', DummyEvaluator()),
             selected_name='best_model',
             input_names=['output_{}'.format(val) for val in lst],
             tr_names=['const_{}'.format(val) for val in lst])
    p.transform(input_names=['valid'],
                tr_name='best_model',
                output_name='final_result')
    return p


def test_selector():
    p = selector_pipeline()
    result = p.run(simple_data())
    assert result['final_result']['value'] == 4
