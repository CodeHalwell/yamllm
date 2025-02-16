import unittest
from yamllm.core.config import Config

class TestConfig(unittest.TestCase):

    def setUp(self):
        self.config = Config()

    def test_load_config(self):
        # Test loading a valid config file
        result = self.config.load_config('path/to/valid_config.yaml')
        self.assertIsNotNone(result)

    def test_validate_config(self):
        # Test validating a valid config
        valid_config = {'key': 'value'}
        self.assertTrue(self.config.validate_config(valid_config))

        # Test validating an invalid config
        invalid_config = {'invalid_key': 'value'}
        self.assertFalse(self.config.validate_config(invalid_config))

if __name__ == '__main__':
    unittest.main()