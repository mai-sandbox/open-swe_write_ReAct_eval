import unittest
from agent import calculator, compiled_graph
from langchain_tavily import TavilySearch


class TestCalculatorTool(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(calculator(2, 3, 'add'), 5)

    def test_subtraction(self):
        self.assertEqual(calculator(5, 3, 'subtract'), 2)


    def test_multiplication(self):
        self.assertEqual(calculator(2, 3, 'multiply'), 6)

    def test_division(self):
        self.assertEqual(calculator(6, 3, 'divide'), 2)

    def test_division_by_zero(self):
        with self.assertRaises(ValueError):
            calculator(6, 0, 'divide')

    def test_invalid_operation(self):
        with self.assertRaises(ValueError):
            calculator(6, 3, 'modulus')


class TestSearchTool(unittest.TestCase):
    def setUp(self):
        self.search_tool = TavilySearch(max_results=2)

    def test_search(self):
        result = self.search_tool.invoke("LangGraph AI tool information")
        self.assertIsInstance(result, dict)
        self.assertIn('results', result)


class TestAssistantBehavior(unittest.TestCase):
    def test_chatbot_response(self):
        state = {"messages": [{"role": "user", "content": "What is LangGraph?"}]}
        response = compiled_graph.invoke(state)
        self.assertIn("messages", response)
        self.assertIsInstance(response["messages"], list)


if __name__ == '__main__':
    unittest.main()

