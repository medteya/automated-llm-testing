from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from helpers.llm_client import get_accuracy_metric

def test_llm_technical_accuracy_syntax():
    test_case = LLMTestCase(
        input="What command is used to run a specific pytest file in Python?",
        actual_output="You can run a specific test file using python -m pytest path/to/test.py.",
        expected_output="pytest path/to/test.py or python -m pytest path/to/test.py"
    )
    metric = get_accuracy_metric(threshold=0.7)
    assert_test(test_case, [metric])

def test_llm_technical_accuracy_config():
    test_case = LLMTestCase(
        input="How do you configure a DeepEval metric to run synchronously in Python so it doesn't use async loops?",
        actual_output="You can configure a DeepEval metric to run synchronously by passing async_mode=False when initializing the metric object.",
        expected_output="Set async_mode=False in the metric's constructor parameters."
    )
    metric = get_accuracy_metric(threshold=0.7)
    assert_test(test_case, [metric])

def test_llm_technical_accuracy_fixture():
    test_case = LLMTestCase(
        input="What is the purpose of a pytest fixture?",
        actual_output="A pytest fixture is used to set up a baseline state or provide reusable data and setup logic for test functions.",
        expected_output="Fixtures provide a fixed baseline, setup, or reusable state for tests."
    )
    metric = get_accuracy_metric(threshold=0.7)
    assert_test(test_case, [metric])