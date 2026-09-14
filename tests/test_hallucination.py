from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from helpers.llm_client import get_hallucination_metric, get_faithfulness_metric

def test_hallucination_database_extraction():
    test_case = LLMTestCase(
        input="Where are persistent user profile details stored in our application architecture?",
        actual_output="User profile details are stored in a PostgreSQL relational database instance.",
        context=["The microservice architecture utilizes FastAPI for handling routing and validation. User session tokens are cryptographically signed and verified, while persistent state, profile details, and audit logs are safely committed to a PostgreSQL relational database instance."]
    )
    metric = get_hallucination_metric(threshold=0.7)
    assert_test(test_case, [metric])

def test_hallucination_testing_standards():
    test_case = LLMTestCase(
        input="Where should our automated test scripts and verification files be located?",
        actual_output="Automated test scripts and verification files must be organized under the dedicated tests directory located at the project root.",
        context=["Our continuous integration pipeline relies heavily on GitHub Actions for running automated workflows. All functional test scripts, end-to-end suites, and unit assertions must be organized under the dedicated tests directory located at the project root."]
    )
    metric = get_hallucination_metric(threshold=0.7)
    assert_test(test_case, [metric])

def test_hallucination_faithfulness_check():
    test_case = LLMTestCase(
        input="What specific port number and host configuration must the local model experimentation client target?",
        actual_output="The local model experimentation client must target port 1234 on localhost.",
        retrieval_context=["For local model experimentation, the service connects directly to an LM Studio instance acting as an OpenAI-compatible provider. The API endpoint base URL must explicitly target port 1234 on localhost."]
    )
    metric = get_faithfulness_metric(threshold=0.7)
    assert_test(test_case, [metric])