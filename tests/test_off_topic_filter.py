"""
Tests for the off-topic domain boundary filter.

Verifies that _is_off_topic() correctly rejects non-Reporium questions
while allowing legitimate repo/AI/ML queries through.
"""

import pytest

from app.routers.intelligence import _is_off_topic, _OFF_TOPIC_RESPONSE


# ---------------------------------------------------------------------------
# Should be REJECTED (off-topic)
# ---------------------------------------------------------------------------

class TestOffTopicRejected:
    """Queries that should be blocked before reaching Claude."""

    # Math / equations
    def test_math_addition(self):
        assert _is_off_topic("what is 2 + 2") is True

    def test_solve_equation(self):
        assert _is_off_topic("solve for x in 3x + 5 = 20") is True

    def test_calculate(self):
        assert _is_off_topic("calculate 15 × 27") is True

    def test_square_root(self):
        assert _is_off_topic("what is the square root of 144") is True

    def test_derivative(self):
        assert _is_off_topic("find the derivative of x^2 + 3x") is True

    def test_factorial(self):
        assert _is_off_topic("what is the factorial of 10") is True

    # Coding exercises
    def test_write_function(self):
        assert _is_off_topic("write a function that reverses a string") is True

    def test_fizzbuzz(self):
        assert _is_off_topic("implement fizzbuzz in python") is True

    def test_fibonacci(self):
        assert _is_off_topic("write me a fibonacci sequence generator") is True

    def test_implement_linked_list(self):
        assert _is_off_topic("implement a linked list in Java") is True

    def test_binary_search(self):
        assert _is_off_topic("binary search algorithm implementation") is True

    def test_write_script(self):
        assert _is_off_topic("write a script to parse CSV files") is True

    # General knowledge / trivia
    def test_capital_city(self):
        assert _is_off_topic("what is the capital of France") is True

    def test_president(self):
        assert _is_off_topic("who is the president of the United States") is True

    def test_weather(self):
        assert _is_off_topic("what is the weather in New York today") is True

    def test_weather_forecast(self):
        assert _is_off_topic("weather forecast for tomorrow") is True

    # Recipes / cooking
    def test_recipe(self):
        assert _is_off_topic("recipe for chocolate cake") is True

    def test_calories(self):
        assert _is_off_topic("how many calories in a banana") is True

    def test_how_to_cook(self):
        assert _is_off_topic("how to cook pasta properly") is True

    # Creative writing
    def test_write_poem(self):
        assert _is_off_topic("write me a poem about the ocean") is True

    def test_tell_joke(self):
        assert _is_off_topic("tell me a joke about programmers") is True

    def test_write_story(self):
        assert _is_off_topic("write a short story about space") is True

    def test_write_essay(self):
        assert _is_off_topic("write an essay about climate change") is True

    # Professional advice
    def test_invest(self):
        assert _is_off_topic("should i invest in cryptocurrency") is True

    def test_symptoms(self):
        assert _is_off_topic("symptoms of the common cold") is True

    def test_legal_advice(self):
        assert _is_off_topic("I need legal advice about my landlord") is True

    def test_diagnose(self):
        assert _is_off_topic("how to diagnose a headache") is True

    # Utility tasks
    def test_set_timer(self):
        assert _is_off_topic("set a timer for 5 minutes") is True

    def test_translate(self):
        assert _is_off_topic("translate hello world to Spanish") is True

    def test_what_time(self):
        assert _is_off_topic("what time is it right now") is True

    def test_what_day(self):
        assert _is_off_topic("what day is it today") is True


# ---------------------------------------------------------------------------
# Should be ALLOWED (on-topic — about repos/AI/ML)
# ---------------------------------------------------------------------------

class TestOnTopicAllowed:
    """Queries that should pass through to Claude."""

    def test_rag_frameworks(self):
        assert _is_off_topic("what are the best RAG frameworks") is False

    def test_compare_tools(self):
        assert _is_off_topic("compare LangChain and LlamaIndex") is False

    def test_pytorch_repos(self):
        assert _is_off_topic("which repos use PyTorch") is False

    def test_vector_databases(self):
        assert _is_off_topic("what vector databases are available") is False

    def test_agent_frameworks(self):
        assert _is_off_topic("list all agent frameworks") is False

    def test_repo_info(self):
        assert _is_off_topic("tell me about the langchain repo") is False

    def test_category_count(self):
        assert _is_off_topic("how many repos are in the inference category") is False

    def test_stars(self):
        assert _is_off_topic("what is the most starred repository") is False

    def test_embedding_models(self):
        assert _is_off_topic("what embedding models are tracked") is False

    def test_tensorflow_vs_pytorch(self):
        assert _is_off_topic("tensorflow vs pytorch which is better") is False

    def test_huggingface(self):
        assert _is_off_topic("what repos does huggingface have") is False

    def test_llm_tools(self):
        assert _is_off_topic("what LLM tools are available") is False

    def test_deployment_tools(self):
        assert _is_off_topic("what tools help with model deployment") is False

    def test_github_activity(self):
        assert _is_off_topic("which github repos are most active") is False

    def test_new_this_week(self):
        assert _is_off_topic("what's new this week") is False

    def test_ai_training(self):
        assert _is_off_topic("what AI training frameworks exist") is False

    def test_transformer_models(self):
        assert _is_off_topic("show me transformer model repos") is False

    def test_openai_repos(self):
        assert _is_off_topic("what repos are from openai") is False

    # Edge case: question that mentions both off-topic pattern AND a repo keyword
    # should be ALLOWED because repo signal overrides
    def test_solve_rag_latency(self):
        assert _is_off_topic("solve RAG latency issues") is False

    def test_implement_vector_search(self):
        assert _is_off_topic("implement a vector search with embeddings") is False

    def test_write_code_for_model(self):
        assert _is_off_topic("write code for deploying a model") is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Short queries, empty strings, boundary conditions."""

    def test_short_query_allowed(self):
        """Queries under 10 chars always pass through."""
        assert _is_off_topic("hi") is False
        assert _is_off_topic("help") is False
        assert _is_off_topic("") is False

    def test_response_content(self):
        """The off-topic response should mention Reporium."""
        assert "Reporium" in _OFF_TOPIC_RESPONSE
        assert "RAG frameworks" in _OFF_TOPIC_RESPONSE

    def test_case_insensitive(self):
        assert _is_off_topic("WHAT IS THE CAPITAL OF FRANCE") is True
        assert _is_off_topic("Write Me A Poem About Stars") is True

    def test_mixed_case_repo_signal(self):
        assert _is_off_topic("WHAT REPOS USE PYTORCH") is False
