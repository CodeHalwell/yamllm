import unittest
from yamllm.llm import LLM

class TestLLM(unittest.TestCase):

    def setUp(self):
        self.llm = LLM()

    def test_initialize(self):
        self.llm.initialize()
        self.assertTrue(self.llm.is_initialized)

    def test_query(self):
        self.llm.initialize()
        response = self.llm.query("What is the capital of France?")
        self.assertIsInstance(response, str)

    def test_get_response(self):
        self.llm.initialize()
        response = self.llm.get_response("Tell me a joke.")
        self.assertIsInstance(response, str)

if __name__ == '__main__':
    unittest.main()