"""Multi-model ensemble system for consensus and voting."""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging


class EnsembleStrategy(Enum):
    """Ensemble aggregation strategies."""
    CONSENSUS = "consensus"  # Require agreement from majority
    BEST_OF_N = "best_of_n"  # Return highest quality response
    VOTING = "voting"  # Vote on best approach
    AVERAGE = "average"  # Average/merge responses
    FIRST_SUCCESS = "first_success"  # Return first successful response


@dataclass
class ModelResponse:
    """Response from a single model."""

    provider: str
    model: str
    response: str
    confidence: float = 0.0
    execution_time: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class EnsembleResult:
    """Result from ensemble execution."""

    strategy: EnsembleStrategy
    final_response: str
    responses: List[ModelResponse]
    agreement_score: float
    selected_model: Optional[str] = None
    reasoning: str = ""


class EnsembleManager:
    """Manage multi-model ensemble execution."""

    def __init__(self, llms: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize ensemble manager.

        Args:
            llms: Dictionary of {provider_name: LLM_instance}
            logger: Optional logger
        """
        self.llms = llms
        self.logger = logger or logging.getLogger(__name__)

    def execute(
        self,
        prompt: str,
        strategy: EnsembleStrategy = EnsembleStrategy.CONSENSUS,
        providers: Optional[List[str]] = None,
        timeout: float = 30.0
    ) -> EnsembleResult:
        """
        Execute prompt across multiple models.

        Args:
            prompt: Prompt to execute
            strategy: Aggregation strategy
            providers: List of providers to use (None = all)
            timeout: Timeout for each model

        Returns:
            EnsembleResult
        """
        self.logger.info(f"Executing ensemble with strategy: {strategy.value}")

        # Select providers
        if providers:
            selected_llms = {k: v for k, v in self.llms.items() if k in providers}
        else:
            selected_llms = self.llms

        if not selected_llms:
            raise ValueError("No LLMs available for ensemble")

        # Execute all models in parallel (or sequentially if async not available)
        responses = self._execute_all(prompt, selected_llms, timeout)

        # Aggregate based on strategy
        result = self._aggregate(responses, strategy)

        self.logger.info(f"Ensemble complete. Agreement score: {result.agreement_score:.2f}")

        return result

    def _execute_all(
        self,
        prompt: str,
        llms: Dict[str, Any],
        timeout: float
    ) -> List[ModelResponse]:
        """Execute prompt on all models."""
        responses = []

        for provider, llm in llms.items():
            try:
                import time
                start_time = time.time()

                # Execute query
                response_text = llm.query(prompt)

                execution_time = time.time() - start_time

                # Get model name
                model_name = getattr(llm, 'model', 'unknown')

                responses.append(ModelResponse(
                    provider=provider,
                    model=model_name,
                    response=response_text,
                    execution_time=execution_time
                ))

                self.logger.info(f"Got response from {provider} in {execution_time:.2f}s")

            except Exception as e:
                self.logger.error(f"Error from {provider}: {e}")
                responses.append(ModelResponse(
                    provider=provider,
                    model="unknown",
                    response="",
                    error=str(e)
                ))

        return responses

    def _aggregate(
        self,
        responses: List[ModelResponse],
        strategy: EnsembleStrategy
    ) -> EnsembleResult:
        """
        Aggregate responses based on strategy.

        Args:
            responses: List of model responses
            strategy: Aggregation strategy

        Returns:
            EnsembleResult
        """
        # Filter out errors
        valid_responses = [r for r in responses if not r.error]

        if not valid_responses:
            return EnsembleResult(
                strategy=strategy,
                final_response="All models failed",
                responses=responses,
                agreement_score=0.0,
                reasoning="No valid responses"
            )

        if strategy == EnsembleStrategy.CONSENSUS:
            return self._consensus(valid_responses, responses)
        elif strategy == EnsembleStrategy.BEST_OF_N:
            return self._best_of_n(valid_responses, responses)
        elif strategy == EnsembleStrategy.VOTING:
            return self._voting(valid_responses, responses)
        elif strategy == EnsembleStrategy.FIRST_SUCCESS:
            return self._first_success(valid_responses, responses)
        else:
            # Default to first valid response
            return EnsembleResult(
                strategy=strategy,
                final_response=valid_responses[0].response,
                responses=responses,
                agreement_score=0.5,
                reasoning="Default strategy"
            )

    def _consensus(
        self,
        valid_responses: List[ModelResponse],
        all_responses: List[ModelResponse]
    ) -> EnsembleResult:
        """
        Consensus strategy - find common ground.

        Looks for agreement between responses.
        """
        if len(valid_responses) == 1:
            return EnsembleResult(
                strategy=EnsembleStrategy.CONSENSUS,
                final_response=valid_responses[0].response,
                responses=all_responses,
                agreement_score=1.0,
                selected_model=f"{valid_responses[0].provider}/{valid_responses[0].model}",
                reasoning="Only one valid response"
            )

        # Calculate similarity between responses
        similarity_matrix = self._calculate_similarities(valid_responses)

        # Find response with highest average similarity to others
        avg_similarities = []
        for i, response in enumerate(valid_responses):
            avg_sim = sum(similarity_matrix[i]) / len(similarity_matrix[i])
            avg_similarities.append((avg_sim, i, response))

        avg_similarities.sort(reverse=True, key=lambda x: x[0])

        best_sim, best_idx, best_response = avg_similarities[0]

        return EnsembleResult(
            strategy=EnsembleStrategy.CONSENSUS,
            final_response=best_response.response,
            responses=all_responses,
            agreement_score=best_sim,
            selected_model=f"{best_response.provider}/{best_response.model}",
            reasoning=f"Highest agreement ({best_sim:.1%}) with other models"
        )

    def _best_of_n(
        self,
        valid_responses: List[ModelResponse],
        all_responses: List[ModelResponse]
    ) -> EnsembleResult:
        """
        Best-of-N strategy - select highest quality response.

        Uses heuristics: length, structure, clarity.
        """
        scored_responses = []

        for response in valid_responses:
            score = self._quality_score(response.response)
            scored_responses.append((score, response))

        scored_responses.sort(reverse=True, key=lambda x: x[0])

        best_score, best_response = scored_responses[0]

        return EnsembleResult(
            strategy=EnsembleStrategy.BEST_OF_N,
            final_response=best_response.response,
            responses=all_responses,
            agreement_score=best_score,
            selected_model=f"{best_response.provider}/{best_response.model}",
            reasoning=f"Highest quality score: {best_score:.2f}"
        )

    def _voting(
        self,
        valid_responses: List[ModelResponse],
        all_responses: List[ModelResponse]
    ) -> EnsembleResult:
        """
        Voting strategy - models vote on approaches.

        For code/structured output, votes on best approach.
        """
        # For simplicity, use consensus-like approach
        # In practice, would use LLM to vote on responses
        return self._consensus(valid_responses, all_responses)

    def _first_success(
        self,
        valid_responses: List[ModelResponse],
        all_responses: List[ModelResponse]
    ) -> EnsembleResult:
        """First success strategy - return first valid response."""
        first = valid_responses[0]

        return EnsembleResult(
            strategy=EnsembleStrategy.FIRST_SUCCESS,
            final_response=first.response,
            responses=all_responses,
            agreement_score=1.0,
            selected_model=f"{first.provider}/{first.model}",
            reasoning="First successful response"
        )

    def _calculate_similarities(
        self,
        responses: List[ModelResponse]
    ) -> List[List[float]]:
        """
        Calculate similarity matrix between responses.

        Simple implementation using word overlap.
        """
        n = len(responses)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    matrix[i][j] = self._text_similarity(
                        responses[i].response,
                        responses[j].response
                    )

        return matrix

    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.

        Simple implementation using word overlap (Jaccard similarity).
        """
        if not text1 or not text2:
            return 0.0

        # Tokenize
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        if union == 0:
            return 0.0

        return intersection / union

    def _quality_score(self, text: str) -> float:
        """
        Calculate quality score for text.

        Heuristics:
        - Length (not too short, not too long)
        - Structure (paragraphs, sentences)
        - Clarity (simple words, good grammar)
        """
        score = 0.0

        # Length score (prefer 100-1000 chars)
        length = len(text)
        if 100 <= length <= 1000:
            score += 30.0
        elif length > 1000:
            score += 20.0
        elif length > 50:
            score += 10.0

        # Structure score
        sentences = text.count('.') + text.count('!') + text.count('?')
        if sentences > 0:
            score += min(sentences * 5, 25.0)

        # Paragraph score
        paragraphs = text.count('\n\n') + 1
        if paragraphs > 1:
            score += min(paragraphs * 5, 15.0)

        # Code block score (if contains code)
        if '```' in text or 'def ' in text or 'class ' in text:
            score += 10.0

        # List score
        list_items = text.count('\n- ') + text.count('\n* ')
        if list_items > 0:
            score += min(list_items * 2, 10.0)

        return score


class ParallelEnsembleManager(EnsembleManager):
    """Async version for true parallel execution."""

    async def execute_async(
        self,
        prompt: str,
        strategy: EnsembleStrategy = EnsembleStrategy.CONSENSUS,
        providers: Optional[List[str]] = None,
        timeout: float = 30.0
    ) -> EnsembleResult:
        """
        Execute prompt across multiple models in parallel.

        Args:
            prompt: Prompt to execute
            strategy: Aggregation strategy
            providers: List of providers to use
            timeout: Timeout for each model

        Returns:
            EnsembleResult
        """
        self.logger.info(f"Executing async ensemble with strategy: {strategy.value}")

        # Select providers
        if providers:
            selected_llms = {k: v for k, v in self.llms.items() if k in providers}
        else:
            selected_llms = self.llms

        if not selected_llms:
            raise ValueError("No LLMs available for ensemble")

        # Execute all models in parallel
        tasks = []
        for provider, llm in selected_llms.items():
            task = self._execute_one_async(provider, llm, prompt, timeout)
            tasks.append(task)

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter exceptions
        valid_responses = []
        for r in responses:
            if isinstance(r, ModelResponse):
                valid_responses.append(r)
            elif isinstance(r, Exception):
                self.logger.error(f"Async error: {r}")

        # Aggregate
        result = self._aggregate(valid_responses, strategy)

        return result

    async def _execute_one_async(
        self,
        provider: str,
        llm: Any,
        prompt: str,
        timeout: float
    ) -> ModelResponse:
        """Execute one model asynchronously."""
        import time
        start_time = time.time()

        try:
            # Check if LLM has async method
            if hasattr(llm, 'query_async'):
                response_text = await llm.query_async(prompt)
            else:
                # Fallback to sync in thread
                loop = asyncio.get_event_loop()
                response_text = await loop.run_in_executor(
                    None,
                    llm.query,
                    prompt
                )

            execution_time = time.time() - start_time

            model_name = getattr(llm, 'model', 'unknown')

            return ModelResponse(
                provider=provider,
                model=model_name,
                response=response_text,
                execution_time=execution_time
            )

        except Exception as e:
            return ModelResponse(
                provider=provider,
                model="unknown",
                response="",
                error=str(e)
            )
