import os
from deepeval.metrics import (
    AnswerRelevancyMetric, 
    HallucinationMetric, 
    GEval,
    ToxicityMetric,
    BiasMetric,
    FaithfulnessMetric
)
from deepeval.test_case import SingleTurnParams

IS_CI = os.getenv("CI") == "true"

if IS_CI:
    os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"
    os.environ["OPENAI_API_KEY"] = os.getenv("GROQ_API_KEY")
    MODEL_NAME = "openai/gpt-oss-20b"
else:
    os.environ["OPENAI_BASE_URL"] = "http://localhost:1234/v1"
    os.environ["OPENAI_API_KEY"] = "lm-studio"
    MODEL_NAME = "google/gemma-4-e4b"

def get_relevancy_metric(threshold=0.7):
    return AnswerRelevancyMetric(
        threshold=threshold,
        model=MODEL_NAME,
        async_mode=False
    )

def get_hallucination_metric(threshold=0.7):
    return HallucinationMetric(
        threshold=threshold,
        model=MODEL_NAME,
        async_mode=False
    )

def get_accuracy_metric(threshold=0.7):
    return GEval(
        name="Accuracy",
        criteria="Determine whether the actual output is factually correct and accurately answers the input based on the expected output.",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT
        ],
        threshold=threshold,
        model=MODEL_NAME,
        async_mode=False
    )

def get_toxicity_metric(threshold=0.7):
    return ToxicityMetric(
        threshold=threshold,
        model=MODEL_NAME,
        async_mode=False
    )

def get_bias_metric(threshold=0.7):
    return BiasMetric(
        threshold=threshold,
        model=MODEL_NAME,
        async_mode=False
    )

def get_faithfulness_metric(threshold=0.7):
    return FaithfulnessMetric(
        threshold=threshold,
        model=MODEL_NAME,
        async_mode=False
    )