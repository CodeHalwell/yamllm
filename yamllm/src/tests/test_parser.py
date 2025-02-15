import unittest
from yamllm.parser import parse_yaml, extract_items

class TestParser(unittest.TestCase):

    def test_parse_yaml(self):
        # Test parsing a valid YAML file
        yaml_content = """
        key1: value1
        key2:
          - item1
          - item2
        """
        result = parse_yaml(yaml_content)
        expected = {'key1': 'value1', 'key2': ['item1', 'item2']}
        self.assertEqual(result, expected)

    def test_extract_items(self):
        # Test extracting items from a parsed YAML dictionary
        yaml_dict = {'key1': 'value1', 'key2': ['item1', 'item2']}
        result = extract_items(yaml_dict)
        expected = ['value1', 'item1', 'item2']
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()