"""Tests for ensemble execution."""

import pytest
from unittest.mock import Mock, MagicMock
from yamllm.core.ensemble import (
    EnsembleManager,
    EnsembleStrategy,
    ModelResponse,
    EnsembleResult
)


@pytest.fixture
def mock_llms():
    """Create mock LLMs."""
    llm1 = Mock()
    llm1.query = Mock(return_value="Response from model 1")
    llm1.model = "gpt-4"

    llm2 = Mock()
    llm2.query = Mock(return_value="Response from model 2")
    llm2.model = "claude-3-sonnet"

    llm3 = Mock()
    llm3.query = Mock(return_value="Response from model 3")
    llm3.model = "gemini-pro"

    return {
        "openai": llm1,
        "anthropic": llm2,
        "google": llm3
    }


def test_ensemble_manager_initialization(mock_llms):
    """Test ensemble manager initialization."""
    manager = EnsembleManager(mock_llms)

    assert manager.llms == mock_llms
    assert manager.logger is not None


def test_ensemble_execute_all_models(mock_llms):
    """Test executing across all models."""
    manager = EnsembleManager(mock_llms)

    result = manager.execute(
        prompt="What is 2+2?",
        strategy=EnsembleStrategy.CONSENSUS
    )

    # All LLMs should be called
    assert mock_llms["openai"].query.called
    assert mock_llms["anthropic"].query.called
    assert mock_llms["google"].query.called

    # Result should be returned
    assert result.final_response is not None
    assert len(result.responses) == 3


def test_ensemble_specific_providers(mock_llms):
    """Test executing with specific providers."""
    manager = EnsembleManager(mock_llms)

    result = manager.execute(
        prompt="What is 2+2?",
        strategy=EnsembleStrategy.CONSENSUS,
        providers=["openai", "anthropic"]
    )

    # Only specified LLMs should be called
    assert mock_llms["openai"].query.called
    assert mock_llms["anthropic"].query.called
    assert not mock_llms["google"].query.called

    assert len(result.responses) == 2


def test_consensus_strategy(mock_llms):
    """Test consensus strategy."""
    manager = EnsembleManager(mock_llms)

    result = manager.execute(
        prompt="What is 2+2?",
        strategy=EnsembleStrategy.CONSENSUS
    )

    assert result.strategy == EnsembleStrategy.CONSENSUS
    assert result.agreement_score >= 0.0
    assert result.agreement_score <= 1.0


def test_best_of_n_strategy(mock_llms):
    """Test best-of-N strategy."""
    # Create responses with different quality
    mock_llms["openai"].query.return_value = "Short"
    mock_llms["anthropic"].query.return_value = """
    A detailed and comprehensive response with multiple sentences and paragraphs.
    This response contains more structure and information.

    It also has multiple paragraphs which increases the quality score.
    """
    mock_llms["google"].query.return_value = "Medium response"

    manager = EnsembleManager(mock_llms)

    result = manager.execute(
        prompt="Explain quantum computing",
        strategy=EnsembleStrategy.BEST_OF_N
    )

    assert result.strategy == EnsembleStrategy.BEST_OF_N
    # Should select the longer, more detailed response
    assert "detailed" in result.final_response.lower() or "comprehensive" in result.final_response.lower()


def test_first_success_strategy(mock_llms):
    """Test first success strategy."""
    manager = EnsembleManager(mock_llms)

    result = manager.execute(
        prompt="What is 2+2?",
        strategy=EnsembleStrategy.FIRST_SUCCESS
    )

    assert result.strategy == EnsembleStrategy.FIRST_SUCCESS
    assert result.final_response is not None


def test_handle_model_errors(mock_llms):
    """Test handling errors from individual models."""
    # Make one model fail
    mock_llms["openai"].query.side_effect = Exception("API error")

    manager = EnsembleManager(mock_llms)

    result = manager.execute(
        prompt="What is 2+2?",
        strategy=EnsembleStrategy.CONSENSUS
    )

    # Should still return result from working models
    assert result.final_response is not None
    assert len(result.responses) == 3

    # Check that error was recorded
    openai_response = next(r for r in result.responses if r.provider == "openai")
    assert openai_response.error is not None


def test_all_models_fail():
    """Test behavior when all models fail."""
    llm1 = Mock()
    llm1.query.side_effect = Exception("API error")
    llm1.model = "gpt-4"

    llm2 = Mock()
    llm2.query.side_effect = Exception("API error")
    llm2.model = "claude-3-sonnet"

    llms = {"openai": llm1, "anthropic": llm2}

    manager = EnsembleManager(llms)

    result = manager.execute(
        prompt="What is 2+2?",
        strategy=EnsembleStrategy.CONSENSUS
    )

    assert result.final_response == "All models failed"
    assert result.agreement_score == 0.0


def test_text_similarity():
    """Test text similarity calculation."""
    manager = EnsembleManager({})

    # Identical texts
    sim = manager._text_similarity("hello world", "hello world")
    assert sim == 1.0

    # No overlap
    sim = manager._text_similarity("hello", "world")
    assert sim < 1.0

    # Partial overlap
    sim = manager._text_similarity("hello world", "hello there")
    assert 0.0 < sim < 1.0


def test_quality_score():
    """Test quality scoring."""
    manager = EnsembleManager({})

    # Short text
    score1 = manager._quality_score("Short")

    # Longer structured text
    score2 = manager._quality_score("""
    This is a longer response with multiple sentences.
    It has better structure and more content.

    It also has multiple paragraphs.
    """)

    # Structured text should score higher
    assert score2 > score1


def test_model_response_dataclass():
    """Test ModelResponse dataclass."""
    response = ModelResponse(
        provider="openai",
        model="gpt-4",
        response="Test response",
        confidence=0.95,
        execution_time=1.5
    )

    assert response.provider == "openai"
    assert response.model == "gpt-4"
    assert response.response == "Test response"
    assert response.confidence == 0.95
    assert response.execution_time == 1.5
    assert response.error is None
    assert response.metadata == {}


def test_ensemble_result_dataclass():
    """Test EnsembleResult dataclass."""
    responses = [
        ModelResponse("openai", "gpt-4", "Response 1"),
        ModelResponse("anthropic", "claude-3", "Response 2")
    ]

    result = EnsembleResult(
        strategy=EnsembleStrategy.CONSENSUS,
        final_response="Final response",
        responses=responses,
        agreement_score=0.85,
        selected_model="openai/gpt-4",
        reasoning="Best agreement"
    )

    assert result.strategy == EnsembleStrategy.CONSENSUS
    assert result.final_response == "Final response"
    assert len(result.responses) == 2
    assert result.agreement_score == 0.85
    assert result.selected_model == "openai/gpt-4"


def test_voting_strategy(mock_llms):
    """Test voting strategy (delegates to consensus)."""
    manager = EnsembleManager(mock_llms)

    result = manager.execute(
        prompt="What is 2+2?",
        strategy=EnsembleStrategy.VOTING
    )

    assert result.strategy == EnsembleStrategy.VOTING
    assert result.final_response is not None


def test_execution_time_tracking(mock_llms):
    """Test that execution time is tracked."""
    manager = EnsembleManager(mock_llms)

    result = manager.execute(
        prompt="What is 2+2?",
        strategy=EnsembleStrategy.CONSENSUS
    )

    # All responses should have execution time
    for response in result.responses:
        if not response.error:
            assert response.execution_time >= 0


def test_empty_llms_dict():
    """Test behavior with empty LLMs dictionary."""
    manager = EnsembleManager({})

    with pytest.raises(ValueError, match="No LLMs available"):
        manager.execute(
            prompt="What is 2+2?",
            strategy=EnsembleStrategy.CONSENSUS
        )


def test_similarity_matrix_calculation(mock_llms):
    """Test similarity matrix calculation."""
    manager = EnsembleManager(mock_llms)

    responses = [
        ModelResponse("openai", "gpt-4", "The answer is four"),
        ModelResponse("anthropic", "claude-3", "The answer is four"),
        ModelResponse("google", "gemini", "The result is 4")
    ]

    matrix = manager._calculate_similarities(responses)

    # Matrix should be NxN
    assert len(matrix) == 3
    assert len(matrix[0]) == 3

    # Diagonal should be 1.0
    assert matrix[0][0] == 1.0
    assert matrix[1][1] == 1.0
    assert matrix[2][2] == 1.0

    # Similar responses should have high similarity
    assert matrix[0][1] > 0.5  # Both say "answer is four"
