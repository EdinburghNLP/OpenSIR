# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict, deque, Counter
from dataclasses import dataclass, field
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import logging
import os
import sys
import re
import time
from typing import Any, Callable, Dict, List, Optional, Union, Literal
from transformers import PreTrainedTokenizerBase

import pickle
import numpy as np
import torch
import transformers
import json
import pandas as pd  # Added for DataFrame operations
from accelerate.utils import (
    gather_object,
    broadcast_object_list,
    is_peft_model,
)
from datasets import Dataset, IterableDataset, load_dataset
from torch import nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    set_seed,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import is_peft_available
from transformers import is_wandb_available, Trainer

from src.open_r1.configs import GRPOConfig, GRPOScriptArguments
from src.open_r1.utils import get_tokenizer
from src.open_r1.utils.callbacks import get_callbacks
from src.open_r1.utils.wandb_logging import init_wandb_training
from src.utils.text_processing import clean_code_block
from src.opensir.utils import openai_request_with_retry
from src.opensir.rewards import DifficultyScorer, ResponseLengthScorer
from src.opensir.example_pool import DynamicExamplePool
from src.utils.math_processing import is_math_expression

# Import needed for embedding functionality
import torch
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config
from trl.data_utils import maybe_apply_chat_template
from trl.import_utils import (
    is_deepspeed_available,
    is_rich_available,
)
from trl.trainer.utils import (
    pad,
    print_prompt_completions_sample,
)
from math_verify import (
    LatexExtractionConfig,
    parse,
)

if is_deepspeed_available():
    import deepspeed

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

logger = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

import torch
import torch.nn.functional as F


def _contains_with_boxed_answer(text: str):
    """
    Check if a string contains 'Therefore, the final answer is: $\\boxed{whatever}$'
    where 'whatever' can be any non-empty content.

    Args:
        text (str): The string to check

    Returns:
        bool: True if the string ends with the pattern, False otherwise
    """
    pattern = r"Therefore, the final answer is: \$\\boxed\{(.+)\}\$"
    return bool(re.search(pattern, text))


def parse_xml(texts):
    parsed = []
    tags = ["problem", "concepts"]
    for text in texts:
        text_results = {}

        for tag in tags:
            pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

            content = None
            if match:
                content = match.group(1).strip()

            text_results[tag] = content

        problem_value = text_results.get("problem")
        if problem_value is None:
            logger.warning(f"Problem not found: '{text}'")
            parsed.append({})
            continue
        if len(str(problem_value).strip()) == 0:
            logger.warning(f"Empty problem found: '{text}'")
            parsed.append({})
            continue
        if str(problem_value).strip() == "...":
            logger.warning(f"Invalid problem found: '{text}'")
            parsed.append({})
            continue

        # Check if problem is under 10 characters
        problem_text = (
            problem_value.encode("utf-8", "replace").decode("utf-8").strip()
        )
        if len(problem_text) < 10:
            logger.warning(
                f"Problem is under 10 characters ({len(problem_text)} chars):"
                f" '{text}'"
            )
            parsed.append({})
            continue

        text_results["problem"] = problem_text

        concepts_value = text_results.get("concepts")
        if concepts_value is None:
            logger.warning(f"Concepts not found: '{text}'")
            parsed.append({})
            continue
        concepts_value = (
            concepts_value.encode("utf-8", "replace").decode("utf-8").strip()
        )
        if len(concepts_value) == 0:
            logger.warning(f"Empty concepts found: '{text}'")
            parsed.append({})
            continue
        if len(concepts_value.split(",")) > 3:
            logger.warning(f"Concepts list has more than 3 items: '{text}'")
            parsed.append({})
            continue
        if "\n" in concepts_value or "\r" in concepts_value:
            logger.warning(f"Concepts string contains newlines: '{text}'")
            parsed.append({})
            continue

        concepts_list = [
            s.strip()
            for s in concepts_value.encode("utf-8", "replace")
            .decode("utf-8")
            .strip()
            .lower()
            .split(",")
        ]

        # Filter out concepts that are under 10 characters
        valid_concepts = [c for c in concepts_list if len(c) >= 10]

        if not valid_concepts:
            logger.warning(
                f"All concepts are under 10 characters: {concepts_list} in"
                f" text: '{text}'"
            )
            parsed.append({})
            continue

        # Log if some concepts were filtered out
        if len(valid_concepts) < len(concepts_list):
            filtered_out = [c for c in concepts_list if len(c) < 10]
            logger.warning(
                "Filtered out concepts under 10 characters:"
                f" {filtered_out} from text: '{text}'"
            )

        concepts_list = valid_concepts

        text_results["concepts"] = concepts_list

        parsed.append(text_results)

    return parsed


def get_student_prompt(problem: str, include_system=True):
    sys_prompt = (
        "You are a helpful AI Assistant, designed to provide"
        " well-reasoned and detailed responses. You FIRST think about the"
        " reasoning process step by step and then provide the user with"
        " the answer. The last line of your response should be 'Therefore,"
        " the final answer is: $\\boxed{ANSWER}$' (without quotes) where"
        " ANSWER is just the final number or expression that solves the"
        " problem."
    )
    if include_system:
        return {
            "prompt": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": problem},
            ]
        }
    else:
        return {
            "prompt": [
                {"role": "user", "content": sys_prompt + "\n\n" + problem},
            ]
        }


def get_concept_prompt(problem: str, include_system=True):
    """
    Generate a prompt for extracting concepts from a given problem.

    Args:
        problem: The math problem to extract concepts from
        include_system: Whether to include system message

    Returns:
        A formatted prompt dict
    """
    system_content = "You are a helpful AI Assistant"
    user_content = f"""You are given a math problem: {problem}

Identify at most three math concepts required to solve this problem. Provide these concepts in a comma separated list inside the <concepts>...</concepts> tags."""

    if include_system:
        return {
            "prompt": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]
        }
    return {
        "prompt": [
            {"role": "user", "content": user_content},
        ]
    }


def get_teacher_prompt(question: str, include_system=True):
    """
    Generate a prompt for creating a new math problem based in-context learning.

    Args:
        question: The reference math problem to create a new problem based on
        include_system: Whether to include system message

    Returns:
        A formatted prompt dict
    """
    system_content = "You are a helpful AI Assistant"
    user_content = """You are given a math problem: {problem}

Your task is to create a math problem that is conceptually different from the provided problem. The new problem must be answerable with a numerical value or mathematical expression.

First, explain how your new problem differs conceptually from the original problem inside the <think>...</think> tags. Then, present your new problem inside the <problem>...</problem> tags. Finally, identify at most three math concepts required to solve your problem. Provide these concepts in a comma separated list inside the <concepts>...</concepts> tags.""".replace(
        "{problem}", question
    )

    if include_system:
        return {
            "prompt": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]
        }
    return {
        "prompt": [
            {"role": "user", "content": user_content},
        ]
    }


@dataclass
class TeacherGRPOConfig(GRPOConfig):
    """
    Configuration for the TeacherGRPOTrainer with specific parameters for the teacher model.
    """

    # In-context learning parameters

    # student
    num_student_generations: int = field(
        default=8,
        metadata={
            "help": (
                "Number of completions to generate per problem by the student"
            )
        },
    )
    num_student_generations_to_train: int = field(
        default=8,
        metadata={
            "help": (
                "Number of completions to train on by the student per prompt. "
                "Must be a multiple of `num_student_generations`."
            )
        },
    )
    student_reward_funcs: list[str] = field(
        default_factory=lambda: [
            "accuracy",
        ],
        metadata={
            "help": "The student reward functions to use.",
            "choices": ["accuracy", "format"],
        },
    )
    student_reward_weights: list[float] = field(
        default_factory=lambda: [1.0],
        metadata={"help": "The student reward weights to use."},
    )

    student_train_sampling_strategy: str = field(
        default="random",
        metadata={
            "help": (
                "Sampling strategy for student completions used for training"
            ),
            "choices": [
                "random",
                "sr-filtered-random",
                "top",
                "sr-filtered-top",
            ],
        },
    )

    # teacher
    num_teacher_generations: int = field(
        default=1,
        metadata={
            "help": (
                "Number of completions to generate per prompt by the teacher"
            )
        },
    )
    num_teacher_generations_to_train: int = field(
        default=16,
        metadata={
            "help": "Number of teacher generations to use for training. "
        },
    )
    teacher_generation_upscale_ratio: int = field(
        default=1,
        metadata={"help": "Up-sampling ratio for teacher completions"},
    )

    num_icl_examples_teacher_generation: int = field(
        default=1,
        metadata={
            "help": (
                "Number of examples to sample from the pool for in-context"
                " learning."
            )
        },
    )
    embedding_port_number: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Port number for the embedding server. Required only if"
                " teacher_diversity reward is used."
            )
        },
    )
    example_pool_max_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Maximum size of the example pool. None or 0 for unlimited."
            )
        },
    )
    max_recent_problems_for_diversity: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Maximum number of most recent problems to consider when "
                "calculating embedding distance. None to use all problems. "
                "Only used when embedding_port_number is provided."
            )
        },
    )
    teacher_reward_funcs: list[str] = field(
        default_factory=lambda: ["solvability"],
        metadata={
            "help": "The teacher reward functions to use.",
            "choices": [
                "solvability",
                "diversity",
                "format",
                "response_length",
            ],
        },
    )
    teacher_reward_weights: list[float] = field(
        default_factory=lambda: [1.0],
        metadata={"help": "The teacher reward weights to use."},
    )
    example_pool_append_strategy: str = field(
        default="all_legal",
        metadata={
            "help": (
                "Sampling strategy for appending examples to the example pool."
                " Either 'all_legal', 'never', or 'top-x' where x is an"
                " integer.."
            ),
            "choices": ["all_legal", "top-x", "never"],
        },
    )
    initial_examples_path: str = field(
        default=None,
        metadata={"help": "Path to the initial examples jsonl"},
    )
    teacher_advantage_global_mean: bool = field(
        default=False,
        metadata={
            "help": (
                "If true, normalise teacher advantages using reward mean/std"
                " computed from all generated problems, not just the training"
                " subset."
            )
        },
    )
    normalize_advantages: bool = field(
        default=True,
        metadata={
            "help": (
                "If true, normalize advantages by dividing by standard"
                " deviation. If false, only center advantages by subtracting"
                " the mean."
            )
        },
    )
    # scoring
    normalise_difficulty_reward: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to normalize difficulty rewards in teacher reward"
                " calculation."
            )
        },
    )
    solve_rate_upper_limit: float = field(
        default=0.9,
        metadata={
            "help": (
                "Upper limit for acceptable solve rates. Problems with solve"
                " rates above this will receive 0 difficulty score."
            )
        },
    )
    solve_rate_lower_limit: float = field(
        default=0.6,
        metadata={
            "help": (
                "Lower limit for acceptable solve rates. Problems with solve"
                " rates below this will receive 0 difficulty score."
            )
        },
    )
    diversity_score_weight: float = field(
        default=1.0,
        metadata={
            "help": (
                "Weight for diversity score when combining with difficulty"
                " score."
            )
        },
    )
    difficulty_score_weight: float = field(
        default=1.0,
        metadata={
            "help": (
                "Weight for difficulty score when combining with diversity"
                " score."
            )
        },
    )

    # Response length reward parameters
    response_length_baseline: float = field(
        default=1000.0,
        metadata={
            "help": (
                "Baseline response length for neutral reward (1.0x). "
                "Longer responses get proportionally higher rewards."
            )
        },
    )
    response_length_cap: Optional[float] = field(
        default=1000.0,
        metadata={
            "help": (
                "Maximum response length to consider. Caps outliers to "
                "prevent extremely long responses from dominating."
            )
        },
    )

    # Random seed configuration
    seed: int = field(
        default=42,
        metadata={
            "help": (
                "Random seed for reproducibility. In distributed training, "
                "each process will use seed + process_index."
            )
        },
    )


# Define the TeacherGRPOTrainer class
class TeacherGRPOTrainer(GRPOTrainer):

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[Callable, list[Callable], list[str]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[
                Dataset,
                IterableDataset,
                dict[str, Union[Dataset, IterableDataset]],
            ]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer],
            Optional[torch.optim.lr_scheduler.LambdaLR],
        ] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        """
        Initialize the TeacherGRPOTrainer.

        This extends the GRPOTrainer initialization with example pool handling and
        special processing for the diversity reward function.

        Args:
            model (Union[str, PreTrainedModel]): The model to train.
            reward_funcs (Union[Callable, list[Callable], list[str]]): Reward functions to use.
                Can include the string "diversity" to use the diversity reward.
            args (Optional[GRPOConfig]): Configuration for the trainer.
            train_dataset (Optional[Union[Dataset, IterableDataset]]): Dataset for training.
            eval_dataset (Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]]):
                Dataset for evaluation.
            processing_class (Optional[PreTrainedTokenizerBase]): Tokenizer for processing inputs.
            reward_processing_classes (Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]]):
                Tokenizers for reward models.
            callbacks (Optional[list[TrainerCallback]]): Callbacks for training events.
            optimizers (tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]]):
                Optimizer and scheduler.
            peft_config (Optional["PeftConfig"]): PEFT configuration.
            embedding_model (str): Name of the model to use for embeddings.
            stop_tokens (Optional[List[str]]): Tokens to stop generation.
        """
        # First, initialize the parent class to get access to the accelerator

        self.use_prover = False

        args.reward_weights = []
        self.model_name = model
        self.use_system_prompt = "gemma" not in model
        super().__init__(
            model=model,
            # Pass an empty list for reward_funcs, we'll set them after creating the example pool
            reward_funcs=[],
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
        self._validate_args(args)

        # Check if processing_class is a tokenizer
        if isinstance(processing_class, PreTrainedTokenizerBase):
            if processing_class.pad_token_id is None:
                processing_class.pad_token = processing_class.eos_token

        # Validate diversity requirement
        if (
            "diversity" in args.teacher_reward_funcs
            and args.embedding_port_number is None
        ):
            raise ValueError(
                "'diversity' reward requires embedding_port_number"
                " to be set"
            )
        if args.initial_examples_path:
            initial_examples = [
                json.loads(row) for row in open(args.initial_examples_path)
            ]
            for row in initial_examples:
                if "question" not in row and "problem" in row:
                    row["question"] = row.pop("problem")

        # Initialize the example pool
        self.example_pool = DynamicExamplePool(
            initial_examples=initial_examples,
            embedding_model="Linq-AI-Research/Linq-Embed-Mistral",
            similarity_metric="cosine",
            use_vllm=True,
            max_size=args.example_pool_max_size,
            port=args.embedding_port_number,
            enable_embeddings=args.embedding_port_number
            is not None,  # Only enable if diversity reward is used
        )

        # student
        self.num_student_generations = args.num_student_generations
        self.student_reward_funcs = args.student_reward_funcs
        self.student_reward_weights = torch.tensor(
            args.student_reward_weights, dtype=torch.float32
        )
        self.num_student_generations_to_train = (
            args.num_student_generations_to_train
        )
        self.student_train_sampling_strategy = (
            args.student_train_sampling_strategy
        )

        # teacher
        self.example_pool_max_size = args.example_pool_max_size
        self.max_recent_problems_for_diversity = (
            args.max_recent_problems_for_diversity
        )
        self.num_teacher_generations = args.num_teacher_generations
        self.num_teacher_generations_to_train = (
            args.num_teacher_generations_to_train
        )
        self.teacher_generation_upscale_ratio = (
            args.teacher_generation_upscale_ratio
        )

        self.num_icl_examples_teacher_generation = (
            args.num_icl_examples_teacher_generation
        )
        self.embedding_port_number = args.embedding_port_number
        self.teacher_reward_funcs = args.teacher_reward_funcs

        self.teacher_reward_weights = torch.tensor(
            args.teacher_reward_weights, dtype=torch.float32
        )
        self.normalize_advantages = args.normalize_advantages
        self.example_pool_append_strategy = args.example_pool_append_strategy
        if 'solvability' in self.teacher_reward_funcs:
            self.difficulty_scorer = DifficultyScorer(
                lower_limit=args.solve_rate_lower_limit,
                upper_limit=args.solve_rate_upper_limit,
                n_generations=self.num_student_generations,
            )

        # Initialize response length scorer if used
        if "response_length" in self.teacher_reward_funcs:
            self.response_length_scorer = ResponseLengthScorer(
                baseline=args.response_length_baseline,
                cap=args.response_length_cap,
            )

        # scoring
        self.normalise_difficulty_reward = args.normalise_difficulty_reward
        self.solve_rate_lower_limit = args.solve_rate_lower_limit
        self.solve_rate_upper_limit = args.solve_rate_upper_limit
        self.diversity_score_weight = args.diversity_score_weight
        self.difficulty_score_weight = args.difficulty_score_weight

        # Pre-encode concept tag IDs for masking
        self._concepts_start_ids = self.processing_class(
            "<concepts>", add_special_tokens=False
        ).input_ids
        self._concepts_end_ids = self.processing_class(
            "</concepts>", add_special_tokens=False
        ).input_ids

        self.num_student_generations_to_train = (
            args.num_student_generations_to_train
        )
        if (
            self.num_student_generations_to_train
            % self.num_student_generations
            != 0
        ):
            raise ValueError(
                "`num_student_generations_to_train` must be a multiple of"
                " `num_student_generations`."
            )

    def _save_example_pool(self, output_dir: str):
        """Save example pool state in a distributed-training-safe manner."""
        if not self.accelerator.is_main_process:
            return

        try:
            pool_dir = os.path.join(output_dir, "example_pool")
            os.makedirs(pool_dir, exist_ok=True)

            # Save examples as JSONL (human-readable, future-proof)
            with open(
                os.path.join(pool_dir, "examples.jsonl"), "w", encoding="utf-8"
            ) as f:
                for example in self.example_pool.examples:
                    f.write(json.dumps(example) + "\n")

            # Save embeddings if they exist
            if self.example_pool.embeddings is not None:
                np.save(
                    os.path.join(pool_dir, "embeddings.npy"),
                    self.example_pool.embeddings,
                )

            # Save metadata for versioning
            metadata = {
                "version": "1.0",
                "pool_size": len(self.example_pool),
                "has_embeddings": self.example_pool.embeddings is not None,
                "timestamp": time.time(),
            }
            with open(os.path.join(pool_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(
                f"Saved example pool with {len(self.example_pool)} examples"
            )
        except Exception as e:
            logger.error(f"Failed to save example pool: {e}")
            # Don't fail the entire save, just log the error

    def _load_example_pool(self, checkpoint_dir: str):
        """Load example pool state, handling missing files gracefully."""
        pool_dir = os.path.join(checkpoint_dir, "example_pool")

        if not os.path.exists(pool_dir):
            logger.warning(
                "No example pool found in checkpoint, keeping current pool"
                " state"
            )
            return

        try:
            # Load metadata first
            metadata_path = os.path.join(pool_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                logger.info(
                    "Loading example pool version"
                    f" {metadata.get('version', 'unknown')}"
                )

            # Load examples
            examples = []
            examples_path = os.path.join(pool_dir, "examples.jsonl")
            if os.path.exists(examples_path):
                with open(examples_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():  # Skip empty lines
                            examples.append(json.loads(line.strip()))

            # Load embeddings
            embeddings = None
            embeddings_path = os.path.join(pool_dir, "embeddings.npy")
            if os.path.exists(embeddings_path):
                embeddings = np.load(embeddings_path)

            # Update the example pool
            self.example_pool.examples = deque(
                examples, maxlen=self.example_pool_max_size
            )
            if embeddings is not None:
                self.example_pool.embeddings = embeddings

            # Synchronize across processes
            self.accelerator.wait_for_everyone()

            logger.info(
                f"Loaded example pool with {len(self.example_pool)} examples"
            )
        except Exception as e:
            logger.error(
                f"Failed to load example pool: {e}. Starting with current pool"
                " state."
            )
            # Don't fail the entire load, just continue with current state

    def save_model(self, output_dir: Optional[str] = None, *args, **kwargs):
        """Save model and example pool state."""
        super().save_model(output_dir=output_dir, *args, **kwargs)

        # Save example pool after model
        if output_dir is not None:
            self._save_example_pool(output_dir)

    def _load_optimizer_and_scheduler(self, checkpoint):
        """Load optimizer, scheduler, and example pool state."""
        super()._load_optimizer_and_scheduler(checkpoint)

        # Load example pool after optimizer/scheduler
        if checkpoint is not None:
            self._load_example_pool(checkpoint)

    def _validate_args(self, args):
        # Validate student_reward_funcs

        valid_student_reward_funcs = ["accuracy", "format"]
        for func in args.student_reward_funcs:
            if func not in valid_student_reward_funcs:
                raise ValueError(
                    f"Invalid student reward function: {func}. "
                    f"Must be one of: {valid_student_reward_funcs}"
                )

        # Validate student_train_sampling_strategy (only "top" supported)
        if args.student_train_sampling_strategy != "top":
            raise ValueError(
                "Invalid student_train_sampling_strategy:"
                f" {args.student_train_sampling_strategy}. Must be 'top'"
            )

        # Validate teacher_reward_funcs (only used functions supported)
        valid_teacher_reward_funcs = [
            "solvability",
            "diversity",
            "format",
            "response_length",
        ]
        for func in args.teacher_reward_funcs:
            if func not in valid_teacher_reward_funcs:
                raise ValueError(
                    f"Invalid teacher reward function: {func}. "
                    f"Must be one of: {valid_teacher_reward_funcs}"
                )

        # Validate group advantage mode requirements
        if args.num_teacher_generations == 1:
            raise ValueError(
                "When using group advantage mode, num_teacher_generations must"
                " be > 1"
            )
        if args.teacher_advantage_global_mean:
            raise ValueError(
                "teacher_advantage_global_mean is not supported for group"
                " advantage mode"
            )

        # Validate example_pool_append_strategy (only "all_legal" supported)
        if args.example_pool_append_strategy != "all_legal":
            raise ValueError(
                "Invalid example_pool_append_strategy:"
                f" {args.example_pool_append_strategy}. Must be 'all_legal'"
            )

    def _get_completion_mask(self, input_ids):
        is_eos = input_ids == self.processing_class.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),),
            is_eos.size(1),
            dtype=torch.long,
            device=input_ids.device,
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[
            is_eos.any(dim=1)
        ]
        sequence_indices = torch.arange(
            is_eos.size(1), device=input_ids.device
        ).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        return completion_mask

    def _apply_concepts_mask(
        self,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        rows_to_process: int,
    ) -> torch.Tensor:
        """
        Apply concepts masking to teacher completions by zeroing out tokens
        between <concepts> and </concepts> tags (inclusive).

        Args:
            completion_ids: Token IDs tensor [batch_size, seq_len]
            completion_mask: Current completion mask [batch_size, seq_len]
            rows_to_process: Number of teacher rows to process (starting from row 0)

        Returns:
            Updated completion mask with concepts tokens zeroed out
        """
        s_ids, e_ids = self._concepts_start_ids, self._concepts_end_ids
        Ls, Le = len(s_ids), len(e_ids)

        for r in range(rows_to_process):  # only teacher rows
            seq = completion_ids[r].tolist()
            try:
                # Find start of <concepts> tag
                s = next(
                    i
                    for i in range(len(seq) - Ls + 1)
                    if seq[i : i + Ls] == s_ids
                )
                # Find end of </concepts> tag after the start tag
                e = next(
                    i
                    for i in range(s + Ls, len(seq) - Le + 1)
                    if seq[i : i + Le] == e_ids
                )
                # Zero out the inclusive span from start to end
                completion_mask[r, s : e + Le] = 0
            except StopIteration:
                # Start or end tag missing - leave mask unchanged
                pass
        return completion_mask

    def _vllm_inference(self, prompts_text: List[str], num_generations: int):
        ori_lengths = gather_object([len(prompts_text)])
        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
        all_prompts_text = gather_object(prompts_text)
        if self.accelerator.is_main_process:
            # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
            # num_generations outputs for each one. This is faster than generating outputs for each duplicate
            # prompt individually.
            ordered_set_of_prompts = all_prompts_text[::num_generations]
            # with profiling_context(self, "vLLM.generate"): # Profiling context might need adjustment
            completion_ids = self.vllm_client.generate(
                prompts=ordered_set_of_prompts,
                n=num_generations,
                repetition_penalty=self.repetition_penalty,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=-1 if self.top_k is None else self.top_k,
                min_p=0.0 if self.min_p is None else self.min_p,
                max_tokens=self.max_completion_length,
                guided_decoding_regex=self.guided_decoding_regex,
            )
        else:
            completion_ids = [None] * len(all_prompts_text)
        # Broadcast the completions from the main process to all processes, ensuring each process receives its
        # corresponding slice.
        completion_ids = broadcast_object_list(completion_ids, from_process=0)
        start_idx = sum(ori_lengths[: self.accelerator.process_index])
        process_slice = slice(
            start_idx, start_idx + ori_lengths[self.accelerator.process_index]
        )
        completion_ids = completion_ids[process_slice]
        return completion_ids

    def _generate_and_score_completions(self, inputs):
        mode = "eval" if self.control.should_evaluate else "train"
        # Initialize profiling timers
        teacher_time = 0.0
        student_time = 0.0

        # Teacher generate problems
        teacher_start_time = time.time()
        n_problems_to_generate: int = (
            self.num_teacher_generations_to_train
            * self.teacher_generation_upscale_ratio
        )
        teacher_prompts: List[List[Dict[str, str]]] = []
        selected_example_idxs_by_teacher: List[int] = []
        for _ in range(n_problems_to_generate // self.num_teacher_generations):
            context_examples_idxs = self.example_pool.sample_examples(
                k=1, get_indexes=True
            )  # Sample K examples
            selected_example_idxs_by_teacher += context_examples_idxs.tolist()
            context_examples = [
                self.example_pool.examples[i] for i in context_examples_idxs
            ]

            # context_examples = self.example_pool.sample_examples(
            #     k=1
            # )  # Sample K examples
            # TODO: Format the prompt properly
            prompt = get_teacher_prompt(
                context_examples[0]["question"],
                include_system=self.use_system_prompt,
            )
            teacher_prompts += [prompt] * self.num_teacher_generations
        teacher_prompts_text: List[str] = [
            maybe_apply_chat_template(prompt, self.processing_class)["prompt"]
            for prompt in teacher_prompts
        ]
        teacher_prompt_inputs = self.processing_class(
            text=teacher_prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        teacher_prompt_inputs = Trainer._prepare_inputs(
            self, teacher_prompt_inputs
        )
        teacher_prompt_ids, teacher_prompt_mask = (
            teacher_prompt_inputs["input_ids"],
            teacher_prompt_inputs["attention_mask"],
        )
        if self.max_prompt_length is not None:
            teacher_prompt_ids = teacher_prompt_ids[
                :, -self.max_prompt_length :
            ]
            teacher_prompt_mask = teacher_prompt_mask[
                :, -self.max_prompt_length :
            ]
        teacher_completion_ids: List[List[int]] = self._vllm_inference(
            teacher_prompts_text, self.num_teacher_generations
        )
        teacher_completions_text: List[str] = (
            self.processing_class.batch_decode(
                teacher_completion_ids, skip_special_tokens=True
            )
        )

        # Parse teacher completions
        problems: List[dict] = parse_xml(teacher_completions_text)
        legal_problem_indexes: List[int] = [
            i for i, problem in enumerate(problems) if problem != {}
        ]

        if len(legal_problem_indexes) == 0:
            raise RuntimeError("No legal problems found.")
        teacher_time = time.time() - teacher_start_time

        # Student generate solutions         student_start_time = time.time()
        student_start_time = time.time()
        student_prompts: List[List[Dict[str, str]]] = [
            get_student_prompt(
                problems[i]["problem"],
                include_system=self.use_system_prompt,
            )
            for i in legal_problem_indexes
        ]
        student_prompts_text: List[str] = [
            maybe_apply_chat_template(prompt, self.processing_class)["prompt"]
            for prompt in student_prompts
            for _ in range(self.num_student_generations)
        ]
        student_prompt_inputs = self.processing_class(
            text=student_prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        student_prompt_ids, student_prompt_mask = (
            student_prompt_inputs["input_ids"],
            student_prompt_inputs["attention_mask"],
        )
        if self.max_prompt_length is not None:
            student_prompt_ids = student_prompt_ids[
                :, -self.max_prompt_length :
            ]
            student_prompt_mask = student_prompt_mask[
                :, -self.max_prompt_length :
            ]
        student_completion_ids: List[List[int]] = self._vllm_inference(
            student_prompts_text, self.num_student_generations
        )
        student_completions_text: List[str] = (
            self.processing_class.batch_decode(
                student_completion_ids, skip_special_tokens=True
            )
        )

        student_answers: List[List[Any]] = [
            parse(
                completion,
                extraction_config=[
                    # ExprExtractionConfig(),
                    LatexExtractionConfig(
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    ),
                ],
                extraction_mode="first_match",
                fallback_mode="first_match",
            )
            for completion in student_completions_text
        ]

        # 2d list of shape (num_problems, num_student_generations)
        solutions: List[Union[None, List[str]]] = [None for _ in problems]
        answers: List[Union[None, List[Any]]] = [None for _ in problems]
        str_answers: List[Union[None, List[str]]] = [None for _ in problems]
        # Reshape student completions into groups for each legal problem
        for i, problem_idx in enumerate(legal_problem_indexes):
            start_idx = i * self.num_student_generations
            end_idx = start_idx + self.num_student_generations
            solutions[problem_idx] = student_completions_text[
                start_idx:end_idx
            ]
            answers[problem_idx] = student_answers[start_idx:end_idx]
            to_add_str_answers = []
            for answer in student_answers[start_idx:end_idx]:
                if answer:
                    try:
                        str_answer = str(answer[0])
                        # Add math expression validation
                        if is_math_expression(str_answer):
                            to_add_str_answers.append(str_answer)
                        else:
                            logger.warning(
                                f"Invalid math expression: {str_answer}"
                            )
                            to_add_str_answers.append(None)
                    except Exception:
                        to_add_str_answers.append(None)
                else:
                    to_add_str_answers.append(None)
            str_answers[problem_idx] = to_add_str_answers

        student_time = time.time() - student_start_time

        # Scoring
        ## solve_rate
        most_common_answers: List[Union[None, str]] = [None for _ in problems]
        solve_rates: List[float] = [None for _ in problems]
        legal_solution_indexes: List[int] = []
        difficulty_scores: List[float] = [None for _ in problems]
        for i, _answers in enumerate(str_answers):
            if _answers is None:
                continue
            valid_answers = [ans for ans in _answers if ans is not None]
            c = Counter(valid_answers)
            if c:
                most_popular_answer = c.most_common(1)[0][0]
                most_common_answers[i] = most_popular_answer
                solve_rates[i] = c.most_common(1)[0][1] / len(valid_answers)
                legal_solution_indexes.append(i)
                difficulty_scores[i] = self.difficulty_scorer.score(
                    solve_rates[i]
                )

        ## response length
        response_length_scores: List[float] = [None for _ in problems]
        if "response_length" in self.teacher_reward_funcs:
            for i, problem_idx in enumerate(legal_solution_indexes):
                # Calculate average student completion length for this problem
                start_idx = i * self.num_student_generations
                end_idx = start_idx + self.num_student_generations
                lengths = [
                    len(student_completion_ids[j])
                    for j in range(start_idx, end_idx)
                ]
                avg_length = sum(lengths) / len(lengths) if lengths else 0

                # Calculate response length reward
                response_length_scores[problem_idx] = (
                    self.response_length_scorer.score(avg_length)
                )

        ## diversity (embedding-based only)
        diversity_scores: List[float] = [None for _ in problems]
        calc_diversity = (
            "diversity" in self.teacher_reward_funcs
        )

        # Calculate diversity pool indices if needed
        if calc_diversity:
            if self.max_recent_problems_for_diversity is not None:
                recent_indices = list(
                    range(
                        len(self.example_pool)
                        - self.max_recent_problems_for_diversity,
                        len(self.example_pool),
                    )
                )
                diversity_pool_indices = list(
                    set(recent_indices + selected_example_idxs_by_teacher)
                )
            else:
                diversity_pool_indices = list(range(len(self.example_pool)))

        # Calculate embeddings if embedding port is provided (needed for pool and possibly diversity)
        candidate_key_embeddings = None
        if self.embedding_port_number is not None and legal_problem_indexes:
            # len = num legal problems
            candidate_key_embeddings: np.ndarray = (
                self.example_pool.get_embeddings(
                    [problems[i]["problem"] for i in legal_problem_indexes],
                    is_query=False,
                )
            )
            norm = np.linalg.norm(
                candidate_key_embeddings, axis=1, keepdims=True
            )
            candidate_key_embeddings = candidate_key_embeddings / norm

        # Calculate embedding diversity if requested
        if calc_diversity and candidate_key_embeddings is not None:
            # key embed
            pool_key_embeddings: np.ndarray = self.example_pool.embeddings
            if self.max_recent_problems_for_diversity is not None:
                pool_key_embeddings = pool_key_embeddings[
                    diversity_pool_indices
                ]
            norm = np.linalg.norm(pool_key_embeddings, axis=1, keepdims=True)
            pool_key_embeddings = pool_key_embeddings / norm

            # not using the instruction of embedding models
            candidate_query_embeddings = candidate_key_embeddings
            candidate_to_candidate_sim = np.dot(
                candidate_query_embeddings, candidate_key_embeddings.T
            )
            # Mask out diagonal (self-similarities)
            np.fill_diagonal(candidate_to_candidate_sim, -np.inf)
            candidate_to_pool_sim = np.dot(
                candidate_query_embeddings, pool_key_embeddings.T
            )
            all_similarities = np.concatenate(
                [candidate_to_candidate_sim, candidate_to_pool_sim], axis=1
            )
            k = min(5, all_similarities.shape[1])
            top_k_similarities = np.partition(all_similarities, -k, axis=1)[
                :, -k:
            ]
            mean_top_k_similarities = np.mean(top_k_similarities, axis=1)
            closest_distances = (1.0 - mean_top_k_similarities) / 2
            for idx, score in zip(legal_problem_indexes, closest_distances):
                diversity_scores[idx] = score
        # Calculate combined scores using embedding diversity
        combined_scores: List[float] = [None for _ in problems]
        for i in range(len(problems)):
            diff_score = difficulty_scores[i]
            div_score = diversity_scores[i]

            if diff_score is not None and div_score is not None:
                combined_scores[i] = (
                    self.difficulty_score_weight * diff_score
                    + self.diversity_score_weight * div_score
                )
        # Add all legal problems to pool
        problems_to_add = []
        embeddings_to_add = []

        for i, problem_idx in enumerate(legal_problem_indexes):
            problems_to_add.append(problems[problem_idx])
            if self.embedding_port_number is not None:
                embeddings_to_add.append(
                    candidate_key_embeddings[i].reshape(1, -1)
                )
            else:
                embeddings_to_add.append(None)

        # Gather problems and embeddings from all processes
        all_problems_to_add = gather_object(problems_to_add)
        all_embeddings_to_add = gather_object(embeddings_to_add)
        if problems_to_add:
            # Prepare all examples with field conversion
            modified_problems = []
            for problem in all_problems_to_add:
                modified_problem = {**problem}
                if "problem" in modified_problem:
                    modified_problem["question"] = modified_problem.pop(
                        "problem"
                    )
                modified_problems.append(modified_problem)

            # Stack embeddings if all are not None
            batch_embeddings = None
            if all(emb is not None for emb in all_embeddings_to_add):
                batch_embeddings = np.vstack(all_embeddings_to_add)

            # Add all examples in one batch operation (O(N) instead of O(N²))
            self.example_pool.add_examples_batch(
                modified_problems, embeddings=batch_embeddings
            )
        self._metrics[mode]["example_pool_size"].append(
            float(len(self.example_pool))
        )

        # Teacher training sample selection
        if self.teacher_generation_upscale_ratio > 1:
            g = self.num_teacher_generations  # generations per prompt
            num_groups = len(problems) // g
            groups_to_take = self.num_teacher_generations_to_train // g

            # compute within-group variance on combined_scores
            group_stats = []
            for gid in range(num_groups):
                start, end = gid * g, (gid + 1) * g
                vals = [
                    s for s in combined_scores[start:end] if s is not None
                ]
                var = np.var(vals) if vals else -1  # (-1 => all illegal)
                group_stats.append((gid, var))

            # pick highest-variance groups
            group_stats.sort(key=lambda x: x[1], reverse=True)
            chosen_groups = [
                gid for gid, _ in group_stats[:groups_to_take]
            ]

            # flatten to generation-level indices
            teacher_selected_problem_idx = []
            for gid in chosen_groups:
                teacher_selected_problem_idx.extend(
                    range(gid * g, (gid + 1) * g)
                )
        else:
            teacher_selected_problem_idx = list(range(len(problems)))

        assert (
            len(teacher_selected_problem_idx)
            == self.num_teacher_generations_to_train
        )

        # Student training sample selection (using "top" strategy)
        num_problems_for_student_training = int(
            self.num_student_generations_to_train
            / self.num_student_generations
        )
        # select the problems with the best combined scores
        valid_score_indexes = [
            idx
            for idx in legal_solution_indexes
            if combined_scores[idx] is not None
        ]
        valid_score_indexes.sort(
            key=lambda x: combined_scores[x], reverse=True
        )
        student_selected_problem_idx = valid_score_indexes[
            :num_problems_for_student_training
        ]
        # If we have fewer selections than needed, randomly select the rest
        if (
            len(student_selected_problem_idx)
            < num_problems_for_student_training
        ):
            remaining_indexes = [
                idx
                for idx in legal_solution_indexes
                if idx not in student_selected_problem_idx
            ]
            additional_needed = num_problems_for_student_training - len(
                student_selected_problem_idx
            )
            additional_selected = random.sample(
                remaining_indexes,
                min(additional_needed, len(remaining_indexes)),
            )
            student_selected_problem_idx.extend(additional_selected)

        # Reward calculation (transformation of score)
        ## student acc (len = len(student_selected_problem_idx))
        student_rewards: List[List[float]] = []
        # Calculate accuracy rewards only for selected student training problems
        for i in student_selected_problem_idx:
            for j, student_answer in enumerate(str_answers[i]):
                rewards = []
                for reward_func in self.student_reward_funcs:
                    if reward_func == "accuracy":
                        most_common_answer = most_common_answers[i]
                        if student_answer is None:
                            # None answers are treated as incorrect
                            rewards.append(0.0)
                        else:
                            # Compare with most common answer
                            rewards.append(
                                1.0
                                if student_answer == most_common_answer
                                else 0.0
                            )
                    elif reward_func == "format":
                        rewards.append(
                            int(_contains_with_boxed_answer(solutions[i][j]))
                        )

                student_rewards.append(rewards)

        ## Teacher reward - compute for ALL problems first
        all_teacher_rewards: List[List[float]] = []
        for i in range(len(problems)):
            rewards = []
            for reward_func in self.teacher_reward_funcs:
                if "solvability" in reward_func:
                    rewards.append(
                        0
                        if solve_rates[i] is None
                        else self.difficulty_scorer.score(solve_rates[i])
                    )
                elif reward_func == "diversity":
                    assert (
                        self.embedding_port_number is not None
                    ), "diversity requires embedding_port_number"
                    rewards.append(
                        diversity_scores[i]
                        if diversity_scores[i] is not None
                        else 0
                    )
                elif reward_func == "format":
                    # further check for think tags
                    correct_format_indexes = []
                    pattern = r"<think>\s*(.*?)\s*</think>"
                    for idx in legal_problem_indexes:
                        match = re.search(
                            pattern,
                            teacher_completions_text[idx],
                            re.DOTALL | re.IGNORECASE,
                        )
                        if match:
                            correct_format_indexes.append(idx)
                    correct_format_indexes = set(correct_format_indexes)
                    rewards.append(i in correct_format_indexes)
                elif reward_func == "response_length":
                    rewards.append(
                        0
                        if response_length_scores[i] is None
                        else response_length_scores[i]
                    )

            all_teacher_rewards.append(rewards)

        # Extract rewards for selected teacher training problems
        teacher_rewards: List[List[float]] = [
            all_teacher_rewards[i] for i in teacher_selected_problem_idx
        ]
        # Calculate advantages
        ## Teacher
        lengths = gather_object([len(teacher_rewards)])
        teacher_rewards = gather_object(teacher_rewards)
        teacher_rewards = torch.tensor(
            teacher_rewards,
            device=self.accelerator.device,
            dtype=torch.float64,
        )
        agg_teacher_rewards = (
            teacher_rewards
            * self.teacher_reward_weights.to(
                self.accelerator.device
            ).unsqueeze(0)
        ).nansum(dim=1)

        # Group advantage mode
        g = self.num_teacher_generations
        if agg_teacher_rewards.numel() % g != 0:
            raise ValueError(
                "Teacher reward count not divisible by group size"
            )
        reward_mean = (
            agg_teacher_rewards.view(-1, g)
            .mean(dim=1)
            .repeat_interleave(g, dim=0)
        )
        if self.normalize_advantages:
            reward_std = (
                agg_teacher_rewards.view(-1, g)
                .std(dim=1)
                .repeat_interleave(g, dim=0)
            )
            teacher_advantages = (agg_teacher_rewards - reward_mean) / (
                reward_std + 1e-4
            )
        else:
            teacher_advantages = agg_teacher_rewards - reward_mean

        # Store global advantages before slicing for logging purposes
        global_teacher_advantages = teacher_advantages.clone()

        start_idx = sum(lengths[: self.accelerator.process_index])
        process_slice = slice(
            start_idx, start_idx + lengths[self.accelerator.process_index]
        )
        teacher_advantages = teacher_advantages[process_slice]

        ## student
        lengths = gather_object([len(student_rewards)])
        student_rewards = gather_object(student_rewards)
        student_rewards = torch.tensor(
            student_rewards,
            device=self.accelerator.device,
            dtype=torch.float64,
        )
        agg_student_rewards = (
            student_rewards
            * self.student_reward_weights.to(
                self.accelerator.device
            ).unsqueeze(0)
        ).nansum(dim=1)

        reward_mean = (
            agg_student_rewards.view(-1, self.num_student_generations)
            .mean(dim=1)
            .repeat_interleave(self.num_student_generations, dim=0)
        )
        if self.normalize_advantages:
            reward_std = (
                agg_student_rewards.view(-1, self.num_student_generations)
                .std(dim=1)
                .repeat_interleave(self.num_student_generations, dim=0)
            )
            student_advantages = (agg_student_rewards - reward_mean) / (
                reward_std + 1e-4
            )
        else:
            student_advantages = agg_student_rewards - reward_mean

        # Store global advantages before slicing for logging purposes
        global_student_advantages = student_advantages.clone()

        start_idx = sum(lengths[: self.accelerator.process_index])
        process_slice = slice(
            start_idx, start_idx + lengths[self.accelerator.process_index]
        )
        student_advantages = student_advantages[process_slice]

        # Create mapping from problem index to student prompt indices
        problem_idx_to_student_prompt_start = {
            problem_idx: i * self.num_student_generations
            for i, problem_idx in enumerate(legal_problem_indexes)
        }

        # Create selected student prompt tensors
        selected_student_prompt_ids = []
        selected_student_prompt_mask = []
        for problem_idx in student_selected_problem_idx:
            start_idx = problem_idx_to_student_prompt_start[problem_idx]
            selected_student_prompt_ids.extend(
                student_prompt_ids[
                    start_idx : start_idx + self.num_student_generations
                ]
            )
            selected_student_prompt_mask.extend(
                student_prompt_mask[
                    start_idx : start_idx + self.num_student_generations
                ]
            )

        # Convert completion_ids to tensors and create masks
        # Convert completion_ids to tensors
        teacher_completion_tensors = [
            torch.tensor(
                completion_ids,
                dtype=torch.long,
                device=self.accelerator.device,
            )
            for completion_ids in teacher_completion_ids
        ]

        student_completion_tensors = [
            torch.tensor(
                completion_ids,
                dtype=torch.long,
                device=self.accelerator.device,
            )
            for completion_ids in student_completion_ids
        ]

        # Create selected student completion tensors
        selected_student_completion_ids = []
        for problem_idx in student_selected_problem_idx:
            start_idx = problem_idx_to_student_prompt_start[problem_idx]
            selected_student_completion_ids.extend(
                student_completion_tensors[
                    start_idx : start_idx + self.num_student_generations
                ]
            )

        # Combine teacher and student prompts/completions
        prompt_ids = pad(
            [teacher_prompt_ids[i] for i in teacher_selected_problem_idx]
            + selected_student_prompt_ids,
            padding_value=self.processing_class.pad_token_id,
            padding_side="left",
        ).to(self.accelerator.device)

        prompt_mask = pad(
            [teacher_prompt_mask[i] for i in teacher_selected_problem_idx]
            + selected_student_prompt_mask,
            padding_value=0,
            padding_side="left",
        ).to(self.accelerator.device)

        completion_ids = pad(
            [
                teacher_completion_tensors[i]
                for i in teacher_selected_problem_idx
            ]
            + selected_student_completion_ids,
            padding_value=self.processing_class.pad_token_id,
            padding_side="right",
        ).to(self.accelerator.device)

        completion_mask = self._get_completion_mask(completion_ids)

        # Apply concepts masking
        num_teacher_completions = len(teacher_selected_problem_idx)
        completion_mask = self._apply_concepts_mask(
            completion_ids, completion_mask, num_teacher_completions
        )

        advantages = torch.cat([teacher_advantages, student_advantages], dim=0)

        # KL related computations
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens
        with torch.no_grad():
            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                )
            else:
                with self.accelerator.unwrap_model(
                    self.model
                ).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                    )

        # metrics logging
        if mode == "train":
            self._total_train_tokens += sum(
                gather_object([attention_mask.sum().item()])
            )
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # Calculate metrics for teacher and student completions
        num_teacher_completions = len(teacher_selected_problem_idx)
        teacher_completion_mask = completion_mask[:num_teacher_completions]
        student_completion_mask = completion_mask[num_teacher_completions:]
        teacher_completion_length = (
            self.accelerator.gather_for_metrics(
                teacher_completion_mask.sum(1).to(self.accelerator.device)
            )
            .float()
            .mean()
            .item()
        )

        student_completion_length = (
            self.accelerator.gather_for_metrics(
                student_completion_mask.sum(1).to(self.accelerator.device)
            )
            .float()
            .mean()
            .item()
        )
        self._metrics[mode]["teacher_completion_length"].append(
            teacher_completion_length
        )
        self._metrics[mode]["student_completion_length"].append(
            student_completion_length
        )
        for i, rew_func in enumerate(self.teacher_reward_funcs):
            self._metrics[mode][f"teacher_reward_{rew_func}"].append(
                teacher_rewards[:, i].mean().item()
            )
        for i, rew_func in enumerate(self.student_reward_funcs):
            self._metrics[mode][f"student_reward_{rew_func}"].append(
                student_rewards[:, i].mean().item()
            )
        self._metrics[mode]["student_reward"].append(
            agg_student_rewards.mean().item()
        )
        self._metrics[mode]["teacher_reward"].append(
            agg_teacher_rewards.mean().item()
        )

        # Log advantage statistics using the stored global advantages
        # Only compute on main process to avoid redundant computation
        if self.accelerator.is_main_process:
            # Teacher advantages
            teacher_advantage_mean = global_teacher_advantages.mean().item()
            teacher_advantage_std = global_teacher_advantages.std().item()

            self._metrics[mode]["teacher_advantage_mean"].append(
                teacher_advantage_mean
            )
            self._metrics[mode]["teacher_advantage_std"].append(
                teacher_advantage_std
            )

            # Student advantages
            student_advantage_mean = global_student_advantages.mean().item()
            student_advantage_std = global_student_advantages.std().item()

            self._metrics[mode]["student_advantage_mean"].append(
                student_advantage_mean
            )
            self._metrics[mode]["student_advantage_std"].append(
                student_advantage_std
            )

        self._metrics[mode]["legal_problem_ratio"].append(
            sum(gather_object([len(legal_problem_indexes)]))
            / sum(gather_object([len(problems)]))
        )
        # Log average solve rates
        legal_solve_rates = [
            solve_rates[i]
            for i in legal_problem_indexes
            if solve_rates[i] is not None
        ]
        legal_solve_rates = gather_object(legal_solve_rates)
        avg_legal_solve_rate = (
            sum(legal_solve_rates) / len(legal_solve_rates)
            if legal_solve_rates
            else 0.0
        )
        teacher_solve_rates = [
            solve_rates[i]
            for i in teacher_selected_problem_idx
            if solve_rates[i] is not None
        ]
        avg_teacher_solve_rate = (
            sum(teacher_solve_rates) / len(teacher_solve_rates)
            if teacher_solve_rates
            else 0.0
        )

        student_solve_rates = [
            solve_rates[i]
            for i in student_selected_problem_idx
            if solve_rates[i] is not None
        ]
        avg_student_solve_rate = (
            sum(student_solve_rates) / len(student_solve_rates)
            if student_solve_rates
            else 0.0
        )

        self._metrics[mode]["total_solve_rate"].append(avg_legal_solve_rate)
        self._metrics[mode]["teacher_solve_rate"].append(
            avg_teacher_solve_rate
        )
        self._metrics[mode]["student_solve_rate"].append(
            avg_student_solve_rate
        )
        self._metrics[mode]["batch_size"].append(prompt_ids.shape[0])

        selected_teacher_solve_rates = [
            solve_rates[i] for i in teacher_selected_problem_idx
        ]
        selected_teacher_most_common_answers = [
            most_common_answers[i] for i in teacher_selected_problem_idx
        ]

        selected_student_solve_rates = []
        selected_student_most_common_answers = []
        for problem_idx in student_selected_problem_idx:
            for _ in range(self.num_student_generations):
                selected_student_solve_rates.append(solve_rates[problem_idx])
                selected_student_most_common_answers.append(
                    most_common_answers[problem_idx]
                )

        solve_rates = gather_object(solve_rates)
        self._metrics[mode]["problems_within_sr_range"].append(
            len(
                [
                    solve_rate
                    for solve_rate in solve_rates
                    if solve_rate is not None
                    and self.solve_rate_lower_limit
                    <= solve_rate
                    <= self.solve_rate_upper_limit
                ]
            )
        )

        profiling_data = {
            "teacher": teacher_time,
            "student": student_time,
        }
        # Gather profiling data from all processes
        all_profiling_data = gather_object(list(profiling_data.values()))

        # Compute mean times across all processes (only for non-zero values)
        section_names = list(profiling_data.keys())
        for i, section_name in enumerate(section_names):
            times = [
                time_value
                for time_value in all_profiling_data[i :: len(section_names)]
                if time_value > 0
            ]
            if times:
                mean_time = sum(times) / len(times)
                self._metrics[mode][
                    f"profiling/Time taken: {section_name}"
                ].append(mean_time)

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
        ):
            # log
            selected_teacher_prompts_text = gather_object(
                [teacher_prompts_text[i] for i in teacher_selected_problem_idx]
            )
            selected_teacher_completions_text = gather_object(
                [
                    teacher_completions_text[i]
                    for i in teacher_selected_problem_idx
                ]
            )

            selected_student_prompts_text = []
            selected_student_completions_text = []
            for problem_idx in student_selected_problem_idx:
                # Find the position of this problem in legal_problem_indexes
                legal_idx = legal_problem_indexes.index(problem_idx)
                start_idx = legal_idx * self.num_student_generations
                end_idx = start_idx + self.num_student_generations
                selected_student_prompts_text.extend(
                    student_prompts_text[start_idx:end_idx]
                )
                selected_student_completions_text.extend(
                    student_completions_text[start_idx:end_idx]
                )
            selected_student_prompts_text = gather_object(
                selected_student_prompts_text
            )
            selected_student_completions_text = gather_object(
                selected_student_completions_text
            )

            # Gather solve rates and most common answers for selected problems
            selected_teacher_solve_rates = gather_object(
                selected_teacher_solve_rates
            )
            selected_teacher_most_common_answers = gather_object(
                selected_teacher_most_common_answers
            )

            selected_student_solve_rates = gather_object(
                selected_student_solve_rates
            )
            selected_student_most_common_answers = gather_object(
                selected_student_most_common_answers
            )

            prompts_to_log = (
                selected_teacher_prompts_text + selected_student_prompts_text
            )
            completions_to_log = (
                selected_teacher_completions_text
                + selected_student_completions_text
            )
            rewards_to_log = (
                agg_teacher_rewards.tolist() + agg_student_rewards.tolist()
            )
            solve_rates_to_log = (
                selected_teacher_solve_rates + selected_student_solve_rates
            )
            most_common_answers_to_log = (
                selected_teacher_most_common_answers
                + selected_student_most_common_answers
            )

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                    )
            if (
                self.args.report_to
                and "wandb" in self.args.report_to
                and wandb.run is not None
            ):
                import pandas as pd

                # Create base table
                table = {
                    "step": (
                        [str(self.state.global_step)] * len(rewards_to_log)
                    ),
                    "prompt": prompts_to_log,
                    "completion": completions_to_log,
                    "reward": rewards_to_log,
                    "solve_rate": solve_rates_to_log,
                    "most_common_answer": most_common_answers_to_log,
                }

                # Add individual teacher rewards
                num_teacher_rewards = len(agg_teacher_rewards)
                for i, reward_func in enumerate(self.teacher_reward_funcs):
                    teacher_reward_values = teacher_rewards[:, i].tolist()
                    # Pad with None for student entries
                    student_padding = [None] * (
                        len(rewards_to_log) - num_teacher_rewards
                    )
                    table[f"teacher_reward_{reward_func}"] = (
                        teacher_reward_values + student_padding
                    )

                # Add individual student rewards
                for i, reward_func in enumerate(self.student_reward_funcs):
                    student_reward_values = student_rewards[:, i].tolist()
                    # Pad with None for teacher entries
                    teacher_padding = [None] * num_teacher_rewards
                    table[f"student_reward_{reward_func}"] = (
                        teacher_padding + student_reward_values
                    )

                # Add advantages to the completions table using stored global advantages
                advantages_to_log = (
                    global_teacher_advantages.tolist()
                    + global_student_advantages.tolist()
                )

                table["advantage"] = advantages_to_log

                df = pd.DataFrame(table)
                # Save table to JSONL file
                wandb_run_name = "default_run"
                if (
                    self.args.report_to
                    and "wandb" in self.args.report_to
                    and wandb.run is not None
                ):
                    wandb_run_name = wandb.run.name.replace("/", "_").replace(
                        " ", "_"
                    )  # Sanitize run name for filename

                output_dir = os.path.join(os.path.dirname(__file__), "outputs")
                os.makedirs(output_dir, exist_ok=True)
                output_filename = f"completions_{wandb_run_name}.jsonl"
                output_filepath = os.path.join(output_dir, output_filename)

                records = df.to_dict(orient="records")
                # Determine file mode based on whether we're resuming
                # If file exists and we're resuming (global_step > 0), append; otherwise overwrite
                if (
                    os.path.exists(output_filepath)
                    and self.state.global_step > 0
                ):
                    file_mode = "a"  # Append if file exists and we're resuming
                else:
                    file_mode = "w"  # Overwrite for new training runs

                with open(output_filepath, file_mode) as f:
                    for record in records:
                        f.write(json.dumps(record) + "\n")
                logger.info(f"Logged completions to {output_filepath}")
                wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": None,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }


def main(
    script_args: GRPOScriptArguments,
    training_args: TeacherGRPOConfig,
    model_args: ModelConfig,
    config_path: Optional[str] = None,
):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device:"
        f" {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)},"
        f" 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if (
        last_checkpoint is not None
        and training_args.resume_from_checkpoint is None
    ):
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint=}."
        )

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load the dataset for training
    if script_args.dataset_name:
        dataset = load_dataset(
            script_args.dataset_name, name=script_args.dataset_config
        )
        # Check if dataset has the expected prompt column
        if (
            script_args.dataset_prompt_column
            not in dataset[script_args.dataset_train_split].column_names
        ):
            logger.warning(
                "Dataset does not contain"
                f" '{script_args.dataset_prompt_column}' column. Creating"
                " empty dataset."
            )
            dataset = Dataset.from_dict({"prompt": []})
    else:
        # Create an empty dataset if no dataset specified
        dataset = Dataset.from_dict({"prompt": []})

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    # Process reward functions
    reward_funcs = []

    # Make a copy of reward_funcs to avoid modifying the original
    reward_func_names = script_args.reward_funcs.copy()
    reward_funcs = reward_func_names

    # Format into conversation (if needed, this might need adjustment for in-context learning)
    def make_conversation(
        example, prompt_column: str = script_args.dataset_prompt_column
    ):
        prompt = []

        if training_args.system_prompt is not None:
            prompt.append(
                {"role": "system", "content": training_args.system_prompt}
            )

        if prompt_column not in example:
            # If the dataset is empty or doesn't have the prompt column, create a dummy prompt
            if not example:
                prompt.append(
                    {"role": "user", "content": "Generate a diverse problem."}
                )
            else:
                raise ValueError(
                    f"Dataset Question Field Error: {prompt_column} is not"
                    " supported."
                )
        else:
            prompt.append({"role": "user", "content": example[prompt_column]})

        return {"prompt": prompt}

    # Apply make_conversation only if the dataset is not empty
    if dataset[script_args.dataset_train_split]:
        dataset = dataset.map(make_conversation)
    else:
        # If dataset is empty, create a dummy dataset with one empty prompt to start generation
        dataset = Dataset.from_dict(
            {
                "prompt": [
                    {"role": "user", "content": "Generate a diverse problem."}
                ]
            }
        )

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs

    #############################
    # Initialize the Teacher GRPO trainer
    #############################
    trainer = TeacherGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != "no"
            else None
        ),
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    dataset_identifier_for_hub = script_args.dataset_name
    if os.path.exists(script_args.dataset_name):
        # It's a local path. Let's use the dataset_config if available and not a path,
        # otherwise, the basename of the file.
        if (
            script_args.dataset_config
            and isinstance(script_args.dataset_config, str)
            and not os.path.exists(script_args.dataset_config)
        ):
            dataset_identifier_for_hub = script_args.dataset_config
        else:
            dataset_identifier_for_hub = os.path.basename(
                script_args.dataset_name
            )
        logger.info(
            "Local dataset path detected. Using"
            f" '{dataset_identifier_for_hub}' as dataset identifier for model"
            " card."
        )

    kwargs = {
        "dataset_name": dataset_identifier_for_hub,
        "tags": ["open-r1"],
    }

    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    if (
        config_path
        and os.path.exists(config_path)
        and "wandb" in training_args.report_to
        and is_wandb_available()
        and wandb.run is not None
    ):
        logger.info(f"Uploading config file {config_path} to W&B.")
        try:
            # wandb.save(config_path, base_path=os.path.dirname(config_path) or ".") # Saves to files dir
            config_artifact = wandb.Artifact(
                name=f"run_config_{wandb.run.id}", type="config"
            )
            config_artifact.add_file(config_path)
            wandb.log_artifact(config_artifact)
            logger.info(
                f"Successfully uploaded {config_path} as a W&B artifact."
            )
        except Exception as e:
            logger.error(f"Failed to upload config file to W&B: {e}")


if __name__ == "__main__":
    # Capture config path before TrlParser modifies sys.argv
    captured_config_path = None
    if "--config" in sys.argv:
        try:
            config_index = sys.argv.index("--config")
            if config_index + 1 < len(sys.argv):
                captured_config_path = sys.argv[config_index + 1]
                # We don't log here yet as logger might not be configured
        except ValueError:
            # This can happen if --config is present without a value,
            # or if TrlParser has unusual behavior with sys.argv.
            # We'll proceed without it, logging can happen later if needed.
            pass

    parser = TrlParser((GRPOScriptArguments, TeacherGRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # Now that logger might be configured via training_args, log the captured path
    if captured_config_path:
        logger.info(f"Captured config path: {captured_config_path}")
    elif "--config" in sys.argv:  # If --config was there but path not captured
        logger.warning(
            "Could not reliably capture --config path from sys.argv, though"
            " --config was present."
        )
    main(
        script_args,
        training_args,
        model_args,
        config_path=captured_config_path,
    )
