import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.opensir.utils import openai_request_with_retry
from collections import deque
from typing import Any, Dict, List, Optional, Union, Literal
import numpy as np
import openai
import logging

logger = logging.getLogger(__name__)


class DynamicExamplePool:
    """
    Maintains a dynamic pool of training examples and their embeddings for diversity calculation.
    Efficiently caches embeddings to avoid recomputing for existing examples.
    Assumes the 'question' field of each example is a string, enforced by upstream logic
    (e.g., json_reward and filtering in TeacherGRPOTrainer for runtime additions,
    and pre-processing for initial_examples).

    This implementation uses vLLM for embedding extraction via an OpenAI-compatible API.
    """

    def __init__(
        self,
        initial_examples: Optional[
            List[Dict[str, Any]]
        ] = None,  # Assumes 'question' is already a string for initial_examples
        embedding_model: str = "meta-llama/Meta-Llama-3.1-8B",
        port: Optional[int] = None,
        use_vllm: bool = True,
        similarity_metric: Literal["cosine", "l2"] = "cosine",
        max_size: Optional[int] = None,
        enable_embeddings: bool = True,
    ):
        """
        Initializes the DynamicExamplePool.

        Args:
            initial_examples (Optional[List[Dict[str, Any]]]): Initial examples to add to the pool.
                                                              It's assumed their 'question' field is already a string.
            embedding_model (str): The name or path of the model to use for generating embeddings.
            port (Optional[int]): Port number for the vLLM embedding server. Required only if enable_embeddings is True.
            use_vllm (bool): Whether to use VLLM for embedding extraction. Defaults to True (only supported option).
            similarity_metric (str): The similarity metric to use for diversity calculation.
            max_size (Optional[int]): Maximum size of the example pool. If None or 0, the pool size is unlimited.
            enable_embeddings (bool): Whether to enable embedding computation for diversity calculations.
        """
        self.max_size = max_size
        self.enable_embeddings = enable_embeddings

        # initial_examples are assumed to have 'question' as string.
        # Examples added via add_example will also have 'question' as string
        # due to upstream processing by json_reward and filtering.
        self.examples: deque[Dict[str, Any]] = deque(initial_examples or [])
        self.embedding_model_name = embedding_model
        self.use_vllm = use_vllm
        self.similarity_metric = similarity_metric.lower()

        if self.similarity_metric not in ["l2", "cosine"]:
            raise ValueError(
                f"Unsupported similarity metric: {similarity_metric}. Choose"
                " either 'l2' or 'cosine'."
            )

        # Validate port requirement
        if self.enable_embeddings and port is None:
            raise ValueError(
                "Port number is required when embeddings are enabled"
            )

        self.embeddings: Optional[np.ndarray] = None
        self.n_dims: Optional[int] = None

        if self.enable_embeddings:
            self._initialize_model_and_tokenizer(port)
            if len(self.examples) > 0:
                self.embeddings = self.get_embeddings(
                    [self._example_to_str(example) for example in self.examples]
                )


    def _initialize_model_and_tokenizer(self, port):
        """Initialize the vLLM client for embedding extraction."""
        openai_api_key = "EMPTY"
        openai_api_base = f"http://localhost:{port}/v1"
        self.client = openai.OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

    def _example_to_str(self, example: Dict[str, Any]) -> str:
        # 'question' is assumed to be a string here.
        return example["question"]

    def _compute_initial_embeddings(self):
        """
        Computes embeddings for all examples in the pool.
        Only called during initialization.
        """
        if not self.examples:
            self.embeddings = None
            return

        # Get embeddings for all examples
        self.embeddings = self.get_embeddings(
            [self._example_to_str(example) for example in self.examples]
        )

    def _get_embeddings_from_str(self, texts: List[str], is_query: bool=False) -> np.ndarray:
        """Extract embeddings using vLLM embedding server."""
        if is_query:
            task = "Find the most similar math problem."
            texts = [f"Instruct: {task}\nQuery: {text}" for text in texts]

        # Sanitize texts for UTF-8 encoding
        texts = [
            text.encode("utf-8", errors="replace").decode("utf-8")
            for text in texts
        ]

        responses = openai_request_with_retry(
            lambda: self.client.embeddings.create(
                input=texts,
                model=self.embedding_model_name,
            )
        )
        return np.array([data.embedding for data in responses.data])


    def get_embeddings(
        self, examples: Union[List[str],Any], is_query: bool = False
    ) -> np.ndarray:
        if not isinstance(examples[0], str):
            examples = [self._example_to_str(example) for example in examples]
        final_embeddings= self._get_embeddings_from_str(examples, is_query)
        if self.similarity_metric == "cosine":
            norms = np.linalg.norm(final_embeddings, axis=1, keepdims=True)
            final_embeddings = final_embeddings / np.maximum(norms, 1e-10)
        return final_embeddings





    def add_example(
        self,
        example: Dict[str, Any],
        new_embedding: Union[None, np.ndarray] = None,
    ):  # 'example' comes from new_examples in _generate_and_score_completions
        """
        Adds a new example to the pool and optionally updates embeddings and the FAISS index.
        Assumes 'example["question"]' is already a string due to upstream processing.
        If the pool is full (and max_size is set), the oldest example is removed.
        Only computes embeddings for the new example and updates the index incrementally if embeddings are enabled.

        Args:
            example (Dict[str, Any]): The new training example to add. 'question' field must be a string.
        """
        # 'example' is used directly as its 'question' field is expected to be a string.

        # Check if the pool is full and needs eviction
        if (
            self.max_size
            and self.max_size > 0
            and len(self.examples) >= self.max_size
        ):
            # Remove the oldest example from the deque
            self.examples.popleft()

            # Remove the corresponding embedding from the numpy array if embeddings are enabled
            if (
                self.enable_embeddings
                and self.embeddings is not None
                and self.embeddings.shape[0] > 0
            ):
                self.embeddings = np.delete(self.embeddings, 0, axis=0)
                # Rebuild the FAISS index after removing an embedding
            elif self.enable_embeddings:
                self.embeddings = None

        # Add the example to the deque
        self.examples.append(example)  # example is used directly

        # Only compute embeddings if they are enabled
        if self.enable_embeddings:
            # Compute embedding only for the new example
            if new_embedding is None:
                new_embedding = self.get_embeddings(
                    [self._example_to_str(example)]  # example is used directly
                )

            # Initialize embeddings if this is the first example or if embeddings became None after eviction
            if self.embeddings is None or self.embeddings.shape[0] == 0:
                self.embeddings = new_embedding
            else:
                # Append the new embedding to existing embeddings
                self.embeddings = np.vstack([self.embeddings, new_embedding])


    def add_examples_batch(
        self,
        examples: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray] = None,
    ):
        """
        Adds multiple examples to the pool in batch.
        Optimized to avoid O(N²) complexity when adding many examples.

        Args:
            examples (List[Dict[str, Any]]): List of examples to add. Each example's
                'question' field must be a string.
            embeddings (Optional[np.ndarray]): Pre-computed embeddings for the examples.
                If None, embeddings will be computed for each example.
                Shape should be (len(examples), embedding_dim) if provided.

        Returns:
            None

        Raises:
            ValueError: If embeddings shape doesn't match number of examples.
        """
        if not examples:
            return

        n_new = len(examples)

        # Validate embeddings shape if provided
        if embeddings is not None and embeddings.shape[0] != n_new:
            raise ValueError(
                f"Number of embeddings ({embeddings.shape[0]}) doesn't match "
                f"number of examples ({n_new})"
            )

        # Handle evictions upfront if pool will exceed capacity
        if self.max_size and self.max_size > 0:
            n_current = len(self.examples)
            n_total = n_current + n_new

            if n_total > self.max_size:
                n_to_evict = n_total - self.max_size

                # Remove oldest examples from deque
                for _ in range(n_to_evict):
                    self.examples.popleft()

                # Remove corresponding embeddings if enabled
                if (
                    self.enable_embeddings
                    and self.embeddings is not None
                    and self.embeddings.shape[0] > 0
                ):
                    self.embeddings = self.embeddings[n_to_evict:]
                elif self.enable_embeddings:
                    self.embeddings = None

        # Add all examples to deque efficiently
        self.examples.extend(examples)

        # Handle embeddings in a single batch operation
        if self.enable_embeddings:
            # Get or compute embeddings for all new examples
            if embeddings is None:
                new_embeddings = self.get_embeddings(
                    [self._example_to_str(ex) for ex in examples]
                )
            else:
                new_embeddings = embeddings

            # Concatenate all embeddings in one operation instead of one-by-one
            if self.embeddings is None or self.embeddings.shape[0] == 0:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])


    # def get_nearest_distance(
    #     self, query_embedding: np.ndarray
    # ) -> tuple[float, Optional[Dict[str, str]]]:
    #     """
    #     Finds the distance to the nearest neighbor in the pool for a given query embedding.

    #     Args:
    #         query_embedding (np.ndarray): The embedding of the query example.

    #     Returns:
    #         tuple[float, Optional[Dict[str, str]]]: A tuple containing the distance to the
    #         nearest neighbor and the nearest example itself. Returns (large value, None)
    #         if the pool is empty.
    #     """
    #     if self.faiss_index is None or len(self.examples) == 0:
    #         return (
    #             float("inf"),
    #             None,
    #         )  # Return a large distance and None if no examples

    #     # Reshape query for FAISS search
    #     query_embedding = query_embedding.astype("float32").reshape(1, -1)

    #     # Perform search
    #     if self.similarity_metric == "cosine":
    #         norms = np.linalg.norm(query_embedding, axis=1, keepdims=True)
    #         query_embedding = query_embedding / np.maximum(norms, 1e-10)
    #     distances, indices = self.faiss_index.search(query_embedding, 1)

    #     nearest_example = self.examples[indices[0][0]]

    #     if self.similarity_metric == "cosine":
    #         # Inner product gives similarity in [-1, 1], convert to distance in [0, 1]
    #         # Original distance: 1.0 - distances[0][0], which was in [0, 2]
    #         # Normalize to [0, 1] by dividing by 2
    #         return (1.0 - distances[0][0]) / 2.0, nearest_example

    #     return distances[0][0], nearest_example

    def sample_examples(self, k: int = 5, get_indexes:bool=False) -> List[str]:
        """
        Randomly samples K examples from the pool for in-context learning.

        Args:
            k (int): The number of examples to sample.

        Returns:
            List[str]: A list of sampled examples.
        """
        if not self.examples:
            return []
        k = min(k, len(self.examples))
        sampled_examples_indices = np.random.choice(
            len(self.examples), k, replace=False
        )
        if get_indexes:
            return sampled_examples_indices
        return [self.examples[i] for i in sampled_examples_indices]

    def __len__(self):
        return len(self.examples)
