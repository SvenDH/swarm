"""Native models and command data need no checkpoint serialization."""
import unittest
from unittest.mock import patch

from wrappers.registry import available


class WrapperRamTests(unittest.TestCase):
    def test_commands_copy_containers_and_treat_markers_as_data(self):
        from wrappers.common import record
        data = {'__recipe_value__': {'module': 'os', 'data': ['literal']}}
        commands = []
        record(commands, 'example', (data,), {})
        data['__recipe_value__']['data'].append('later')
        self.assertEqual(commands[0][1][0]['__recipe_value__']['data'], ['literal'])

    @unittest.skipUnless('numpy' in available(), 'NumPy not installed')
    def test_commands_keep_native_array_copies_not_encoded_payloads(self):
        import numpy as np
        from wrappers.common import record
        array = np.arange(6).reshape(2, 3)
        commands = []
        record(commands, 'plot', (array,), {})
        saved = commands[0][1][0]
        self.assertIsInstance(saved, np.ndarray)
        self.assertFalse(np.shares_memory(array, saved))
        array[:] = 0
        self.assertEqual(saved.tolist(), [[0, 1, 2], [3, 4, 5]])
        with self.assertRaises(TypeError):
            record(commands, 'unsafe', (np.array([lambda: None], dtype=object),), {})

    @unittest.skipUnless('sklearn' in available(), 'Scikit-learn not installed')
    def test_fitted_model_is_reused_and_counted_against_budget(self):
        import numpy as np
        from wrappers import sklearn
        from wrappers.protocol import NativeMemory
        model = sklearn.factory('StandardScaler')()
        native = model.model
        values = np.arange(300).reshape(100, 3)
        self.assertIs(model.call('fit', values), model)
        with patch.dict(sklearn.FACTORIES, {'StandardScaler': lambda **kw: self.fail('reconstructed model')}):
            np.testing.assert_allclose(model.call('transform', values), native.transform(values))
        self.assertIs(model.model, native)
        self.assertIs(sklearn.members(model, None, 'mean_')['mean_'], native.mean_)
        memory = NativeMemory(1024 * 1024)
        memory.entries['test'] = ('owner', 'sklearn', model)
        self.assertGreater(memory.size(), native.mean_.nbytes + native.scale_.nbytes)
        self.assertNotIn('model', sklearn.members(model, None, 'model'))
