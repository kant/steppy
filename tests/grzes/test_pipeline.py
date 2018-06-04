import numpy as np

from steppy.base import BaseTransformer
from steppy.adapter import Adapter, E

from steppy.grzes.pipeline import Pipeline, TransformerDesc

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

    def transform(self, *args, **kwargs):
        return self.c


def test_simple_pipeline():
    def simple_pipeline():
        p = Pipeline(CACHE_DIRPATH)
        centerer_adapter = Adapter({
                  'numbers': E('train', 'data')
        })
        p.fit(input_names=['train'],
              transformer=TransformerDesc('centerer', CenteringTransformer, {}),
              adapter=centerer_adapter)
        p.transform(input_names=['valid'],
                    tr_name='centerer',
                    output_name='valid_transformed',
                    adapter=centerer_adapter)
        return p

    data = {
        'train': {
            'numbers': np.array([1,2,3])
        },
        'valid': {
            'numbers': np.array([0, 5, 3])
        }
    }
    p = simple_pipeline()
    results = p.run(data)
    expected = np.array([0, 5, 3]) - 2
    original_id = results['id']
    assert results['valid']['centered'] == expected

    p = simple_pipeline()
    results = p.run(data)
    assert results['valid']['centered'] == expected
    assert results['valid']['id'] == original_id


# def test_selector():
#     p = Pipeline(CACHE_DIRPATH)
