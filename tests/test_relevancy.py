from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from helpers.llm_client import get_relevancy_metric

def test_relevancy_basic():
    test_case = LLMTestCase(
        input="How do I run pytest in Python?",
        actual_output="You can run tests using the command python -m pytest."
    )
    metric = get_relevancy_metric(threshold=0.7)
    assert_test(test_case, [metric])

def test_relevancy_with_noise():
    test_case = LLMTestCase(
        input="Hey, I've been working on my project layout all morning and my team lead wants everything standardized. Specifically, what directory name should we use to store our automated test files?",
        actual_output="You should store your automated test files in a dedicated 'tests/' directory at the root of your project."
    )
    metric = get_relevancy_metric(threshold=0.7)
    assert_test(test_case, [metric])

def test_relevancy_edge_case():
    test_case = LLMTestCase(
        input="Can you explain what virtual environments are and why we use them?",
        actual_output="Virtual environments isolate your project dependencies and python interpreter version locally to avoid conflicts."
    )
    metric = get_relevancy_metric(threshold=0.7)
    assert_test(test_case, [metric])