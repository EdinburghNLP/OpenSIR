"""
Generate problems using teacher-student consistency filtering.

This script generates mathematical problems by:
1. Creating new problems using a teacher model
2. Generating multiple solutions using a student model
3. Filtering problems based on consistency ratio of student answers

Example usage to start the VLLM server:
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server  \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --trust-remote-code \
    --tensor_parallel_size 4 \
    --seed 1
"""

import rootutils

setup = rootutils.setup_root(
    __file__, indicator=".project-root", pythonpath=True
)
import argparse
import random
import json
import os
import logging
import time
import numpy as np
import openai
import re
from typing import List, Dict, Tuple
from transformers import AutoTokenizer
from collections import Counter, defaultdict
from math_verify import parse, LatexExtractionConfig, ExprExtractionConfig
from tqdm import tqdm

from src.opensir.train import get_teacher_prompt, parse_xml, get_student_prompt


from src.utils.text_processing import sanitize_text

# Suppress logging from json_reward function
logging.getLogger("src.teacher.grpo_teacher").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


class TopKProfiler:
    """Profiler class to track timing for top-k selection milestones."""

    def __init__(self):
        self.timings = defaultdict(list)
        self.counters = defaultdict(int)
        self.start_times = {}
        self.total_start_time = None

    def start_timer(self, milestone: str):
        """Start timing a milestone."""
        self.start_times[milestone] = time.time()

    def end_timer(self, milestone: str):
        """End timing a milestone and record the duration."""
        if milestone in self.start_times:
            duration = time.time() - self.start_times[milestone]
            self.timings[milestone].append(duration)
            del self.start_times[milestone]
            return duration
        return 0

    def increment_counter(self, counter_name: str, value: int = 1):
        """Increment a counter."""
        self.counters[counter_name] += value

    def start_total_timer(self):
        """Start the total process timer."""
        self.total_start_time = time.time()

    def get_total_time(self):
        """Get total elapsed time."""
        if self.total_start_time:
            return time.time() - self.total_start_time
        return 0

    def print_milestone_summary(self, iteration: int):
        """Print timing summary for current iteration."""
        print(f"\n=== TOP-K PROFILING SUMMARY - Iteration {iteration} ===")

        # Print individual milestone timings for this iteration
        for milestone, times in self.timings.items():
            if times:  # Only show milestones that have been recorded
                latest_time = times[-1]
                avg_time = sum(times) / len(times)
                total_time = sum(times)
                print(f"{milestone}:")
                print(f"  Latest: {latest_time:.2f}s")
                print(f"  Average: {avg_time:.2f}s")
                print(f"  Total: {total_time:.2f}s")
                print(f"  Count: {len(times)}")

        # Print counters
        print("\nCounters:")
        for counter, value in self.counters.items():
            print(f"  {counter}: {value}")

        print(f"\nTotal elapsed time: {self.get_total_time():.2f}s")
        print("=" * 55)

    def print_final_summary(self):
        """Print comprehensive final summary."""
        total_time = self.get_total_time()
        print(f"\n{'='*60}")
        print("FINAL TOP-K SELECTION PROFILING SUMMARY")
        print(f"{'='*60}")
        print(f"Total process time: {total_time:.2f}s")

        print(f"\nMILESTONE BREAKDOWN:")
        print(
            f"{'Milestone':<35} {'Total(s)':<10} {'Avg(s)':<10} {'Count':<8} {'%Total':<8}"
        )
        print("-" * 75)

        for milestone, times in sorted(self.timings.items()):
            if times:
                total_milestone = sum(times)
                avg_milestone = total_milestone / len(times)
                percentage = (
                    (total_milestone / total_time * 100)
                    if total_time > 0
                    else 0
                )
                print(
                    f"{milestone:<35} {total_milestone:<10.2f} {avg_milestone:<10.2f} {len(times):<8} {percentage:<8.1f}"
                )

        print(f"\nCOUNTERS:")
        for counter, value in sorted(self.counters.items()):
            print(f"  {counter}: {value}")

        # Calculate efficiency metrics
        if self.counters.get("problems_generated", 0) > 0:
            problems_per_second = (
                self.counters["problems_generated"] / total_time
            )
            print(f"\nEFFICIENCY METRICS:")
            print(
                f"  Problems generated per second: {problems_per_second:.2f}"
            )

        if self.counters.get("problems_selected", 0) > 0:
            selection_efficiency = self.counters[
                "problems_selected"
            ] / self.counters.get("problems_generated", 1)
            print(
                "  Selection efficiency:"
                f" {selection_efficiency:.3f} ({self.counters['problems_selected']}/{self.counters.get('problems_generated', 0)})"
            )


def trim_after_first_boxed(text: str) -> str:
    """
    Trim text after the first complete \\boxed{...} pattern.
    Handles nested braces correctly.

    Args:
        text: Input text that may contain \\boxed{...} pattern

    Returns:
        Text trimmed after the first complete \\boxed{...} pattern,
        or original text if no pattern is found

    Example:
        >>> trim_after_first_boxed("The answer is \\boxed{42} and more text")
        "The answer is \\boxed{42}"
        >>> trim_after_first_boxed("Complex: \\boxed{x^{2} + 1} extra")
        "Complex: \\boxed{x^{2} + 1}"
    """
    if "\\boxed{" not in text:
        return text

    start_idx = text.find("\\boxed{")
    if start_idx == -1:
        return text

    # Find matching closing brace, handling nested braces
    brace_count = 0
    i = start_idx + 7  # len('\\boxed{')
    while i < len(text):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            if brace_count == 0:
                # Found matching closing brace
                # Check if there's a closing $ immediately after the }
                end_idx = i + 1
                if end_idx < len(text) and text[end_idx] == "$":
                    end_idx += 1
                return text[:end_idx]
            brace_count -= 1
        i += 1

    # If no matching closing brace found, return original
    return text


def has_no_numbers(answer: str) -> bool:
    """
    Check if an answer contains no numbers.

    Mathematical answers should typically contain numbers. Answers without numbers
    are likely either variables only (like "x*y") or English words interpreted as
    variables (like "A*N*S*W*E*R*S"), both of which should be filtered out.

    Args:
        answer: The parsed answer string

    Returns:
        True if the answer contains no digits

    Example:
        >>> has_no_numbers("A*N*S*W*E*R*S")
        True
        >>> has_no_numbers("x*y")
        True
        >>> has_no_numbers("42")
        False
        >>> has_no_numbers("x^2 + 3")
        False
    """
    if answer is None:
        return True

    answer_str = str(answer).strip()

    # Empty strings have no numbers
    if not answer_str:
        return True

    # Check if the answer contains any digits
    return not any(char.isdigit() for char in answer_str)


def is_problem_too_short(problem_text: str, min_length: int = 30) -> bool:
    """
    Check if a problem text is too short to be a meaningful mathematical problem.

    Args:
        problem_text: The problem text to check
        min_length: Minimum number of characters required (default: 30)

    Returns:
        True if the problem is too short

    Example:
        >>> is_problem_too_short("What is 2+2?")
        True
        >>> is_problem_too_short("Find the derivative of f(x) = x^2 + 3x + 5 with respect to x.")
        False
    """
    if problem_text is None:
        return True

    return len(str(problem_text).strip()) < min_length


def contains_xml_tags(problem_text: str) -> bool:
    """
    Check if a problem text contains XML-like tags.

    This filter removes problems that contain XML tags within the problem
    content itself, which can occur with any model type.

    Args:
        problem_text: The problem text to check

    Returns:
        True if the problem contains XML tags

    Example:
        >>> contains_xml_tags("What is <problem>2+2</problem>?")
        True
        >>> contains_xml_tags("<think>Let me solve this</think>Find x")
        True
        >>> contains_xml_tags("If x < 5, what is x + 3?")
        False
    """
    if problem_text is None:
        return False

    text = str(problem_text).strip()

    # Simple and robust pattern to match XML-like tags:
    # - <tag>, </tag>, <tag/>, <tag attributes>
    # Focuses on well-formed tags that are most likely to appear in generated text
    xml_pattern = r"</?[a-zA-Z][a-zA-Z0-9_-]*(?:\s[^<>]*?)?/?>"

    matches = re.findall(xml_pattern, text)

    # Filter out mathematical inequalities and operators
    for match in matches:
        # Check if this looks like a real XML tag (has alphabetic tag name)
        tag_content = match.strip("<>/")
        if tag_content and re.match(r"^[a-zA-Z]", tag_content.split()[0]):
            # Additional check: avoid false positives with single character "tags"
            tag_name = tag_content.split()[0]
            if (
                len(tag_name) > 1
            ):  # Real XML tags are usually longer than 1 character
                return True

    return False


def compute_skill_diversity_scores(
    current_skills: list, selected_skills: list, candidate_indices: list
) -> np.ndarray:
    """
    Compute skill-based diversity scores for candidates.

    Args:
        current_skills: Skills for current iteration problems
        selected_skills: Skills for already selected problems
        candidate_indices: Indices of candidates to score

    Returns:
        Array of diversity scores based on skill overlap
    """
    diversity_scores = np.ones(len(candidate_indices))

    if not selected_skills:
        return diversity_scores

    for i, candidate_idx in enumerate(candidate_indices):
        candidate_skills = set(current_skills[candidate_idx])
        max_overlap = 0

        for selected_skill_set in selected_skills:
            overlap = len(
                candidate_skills.intersection(set(selected_skill_set))
            )
            max_overlap = max(max_overlap, overlap)

        # Normalize: overlap=0 gets score=1, overlap>=3 gets score=0
        if max_overlap >= 3:
            diversity_scores[i] = 0.0
        else:
            diversity_scores[i] = (len(candidate_skills) - max_overlap) / 3.0

    return diversity_scores


def top_k_selection(
    args,
    main_client,
    main_model,
    main_tokenizer,
    include_system=False,
):
    """
    Simplified iterative top-k selection using consistency filtering.

    Each iteration:
    1. Generates initial_batch_size * scale_factor problems using current example pool
    2. Scores all problems based on consistency
    3. Selects top initial_batch_size problems using vectorized operations
    4. Adds selected problems to example pool for next iteration
    5. Continues until n_target_problems total problems are collected

    Args:
        args: argparse.Namespace, command-line arguments
        main_client: OpenAI client for main model
        main_model: str, main model name
        main_tokenizer: tokenizer for main model
        include_system: Whether to include system prompt in generation
    """
    output_file = args.output_filename
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Initialize profiler
    profiler = TopKProfiler()
    profiler.start_total_timer()

    # Initialize tracking for formatting issues
    total_problems_attempted = 0
    total_formatting_issues = 0

    # Initialize with the same initial examples as main function
    icl_example_pool = [{"question": "What is 1+1?", "difficulty": 1}]
    all_selected_problems = []

    # Resume from existing file if it exists
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"Resuming from existing file: {output_file}")
        with open(output_file, "r") as f:
            try:
                all_selected_problems = [json.loads(line) for line in f]
                print(f"Loaded {len(all_selected_problems)} problems.")
                for p_data in all_selected_problems:
                    problem = sanitize_text(p_data["problem"])
                    icl_example_pool.append({"question": problem})
            except json.JSONDecodeError:
                print(
                    "Warning: Could not parse JSON from output file. Starting"
                    " fresh."
                )
                all_selected_problems = []

    if not all_selected_problems:
        # If file didn't exist, was empty, or failed to parse, truncate it
        with open(output_file, "w") as f:
            pass

    print("Starting iterative top-k selection with profiling:")
    print(f"- Target problems: {args.n_target_problems}")
    print(f"- Problems per iteration: {args.initial_batch_size}")
    print(f"- Scale factor: {args.scale_factor}")
    print("- Will continue until target is reached")

    iteration = 0
    while len(all_selected_problems) < args.n_target_problems:
        profiler.start_timer("iteration_total")
        remaining_problems = args.n_target_problems - len(
            all_selected_problems
        )
        if remaining_problems <= 0:
            break

        current_target = min(args.initial_batch_size, remaining_problems)
        n_to_generate = current_target * args.scale_factor

        print(f"\n--- Iteration {iteration + 1} ---")
        print(f"Current example pool size: {len(icl_example_pool)}")
        print(
            f"Generating {n_to_generate} problems to select top"
            f" {current_target}"
        )

        # Milestone 1: Problem Generation with Consistency Filtering
        profiler.start_timer("1_problem_generation")
        valid_candidate_problems = []
        total_problems_generated = 0
        generation_round = 0
        MAX_GENERATION_ROUNDS = 50  # Prevent infinite loops

        print(
            f"Generating problems until we have {n_to_generate} within"
            f" consistency range [{args.lower_ratio}, {args.higher_ratio}]..."
        )

        while (
            len(valid_candidate_problems) < n_to_generate
            and generation_round < MAX_GENERATION_ROUNDS
        ):
            generation_round += 1
            print(
                f"  Generation round {generation_round}: Have"
                f" {len(valid_candidate_problems)}/{n_to_generate} valid"
                " problems"
            )

            # Generate a batch of problems
            batch_candidate_problems = []
            consecutive_failures = 0
            MAX_CONSECUTIVE_FAILURES = 10

            while (
                len(batch_candidate_problems) < n_to_generate
                and consecutive_failures < MAX_CONSECUTIVE_FAILURES
            ):
                problems_before_batch = len(batch_candidate_problems)
                batch_size = n_to_generate - len(batch_candidate_problems)

                # Sample different examples from current pool for each problem prompt
                n_samples = min(len(icl_example_pool), args.n_icl)

                generate_problem_prompts = []
                for _ in range(batch_size):
                    # Sample different examples for each prompt
                    sampled_examples = (
                        random.sample(icl_example_pool, n_samples)
                        if n_samples > 0
                        else []
                    )
                    assert n_samples == 1, "n_samples should be 1"
                    generate_problem_prompts.append(
                        get_teacher_prompt(
                            question=sampled_examples[0]["question"],
                            include_system=include_system,
                        )
                    )
                prompts = [
                    main_tokenizer.apply_chat_template(
                        conversation=p["prompt"],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for p in generate_problem_prompts
                ]
                outputs = openai_request_with_retry(
                    lambda: main_client.completions.create(
                        model=main_model,
                        prompt=prompts,
                        max_tokens=args.max_tokens,
                        temperature=1.0,
                    )
                )
                raw_problem_outputs = [o.text for o in outputs.choices]

                # Track total problems attempted
                total_problems_attempted += len(raw_problem_outputs)

                parsed_outputs = parse_xml(raw_problem_outputs)

                # Count formatting issues
                formatting_issues_this_batch = sum(
                    1 for p_output in parsed_outputs if not p_output
                )
                total_formatting_issues += formatting_issues_this_batch

                # Extract problem text and skills from JSON
                for i, p_output in enumerate(parsed_outputs):
                    if p_output:
                        problem_text = sanitize_text(p_output["problem"])
                        # Filter out problems with XML tags for all models
                        if contains_xml_tags(problem_text):
                            continue
                        skills = p_output["concepts"]
                        batch_candidate_problems.append(
                            {
                                "problem": problem_text,
                                "concepts": skills,
                                "raw_problem_output": raw_problem_outputs[i],
                            }
                        )

                # Print formatting issues for this batch
                if formatting_issues_this_batch > 0:
                    print(
                        f"    Found {formatting_issues_this_batch} formatting"
                        f" issues out of {len(raw_problem_outputs)} problems"
                        " in this batch"
                    )

                # Check if we added any problems in this batch
                problems_added_this_batch = (
                    len(batch_candidate_problems) - problems_before_batch
                )
                if problems_added_this_batch == 0:
                    consecutive_failures += 1
                    print(
                        "    No valid problems added in this batch."
                        " Consecutive failures:"
                        f" {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}"
                    )
                else:
                    consecutive_failures = (
                        0  # Reset counter when we successfully add problems
                    )
                    print(
                        f"    Added {problems_added_this_batch} valid problems"
                        " in this batch."
                    )

            total_problems_generated += len(batch_candidate_problems)

            if len(batch_candidate_problems) == 0:
                print(
                    "    Warning: No problems generated in round"
                    f" {generation_round}"
                )
                continue

            print(
                f"    Generated {len(batch_candidate_problems)} problems in"
                f" round {generation_round}. Scoring them..."
            )

            # Generate solutions for this batch
            all_solution_prompts = []
            for p_data in batch_candidate_problems:
                # For base models, use include_system=False to get simpler prompt format
                solution_prompt = get_student_prompt(
                    p_data["problem"],
                    include_system=include_system,
                )["prompt"]

                solution_chat_prompt = main_tokenizer.apply_chat_template(
                    solution_prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                all_solution_prompts.append(solution_chat_prompt)

            # Single batch API call for all problems
            solution_outputs = openai_request_with_retry(
                lambda: main_client.completions.create(
                    model=main_model,
                    prompt=all_solution_prompts,
                    max_tokens=args.max_tokens,
                    temperature=1.0,
                    n=args.num_solution_generations,
                )
            )

            # Reorganize solutions by problem
            all_solutions = [
                choice.text for choice in solution_outputs.choices
            ]

            solutions_per_problem = [
                all_solutions[
                    i
                    * args.num_solution_generations : (i + 1)
                    * args.num_solution_generations
                ]
                for i in range(len(batch_candidate_problems))
            ]

            # Calculate consistency ratios for this batch
            batch_valid_problems = []
            for i, p_data in enumerate(batch_candidate_problems):
                solutions = solutions_per_problem[i]
                predictions = [
                    parse(
                        s,
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
                    for s in solutions
                ]
                parsed_answers = []
                for p in predictions:
                    try:
                        parsed_answers.append(
                            str(p[0]) if len(p) > 0 else None
                        )
                    except Exception as e:
                        # Handle cases where sympy expressions contain NaN, division by zero, None values, or other problematic values
                        parsed_answers.append(None)
                filtered_answers = [a for a in parsed_answers if a is not None]
                if not filtered_answers:
                    p_data["consistency_ratio"] = 0.0
                    p_data["most_popular_answer"] = None
                else:
                    c = Counter(filtered_answers)
                    p_data["consistency_ratio"] = c.most_common(1)[0][1] / len(
                        filtered_answers
                    )
                    p_data["most_popular_answer"] = c.most_common(1)[0][0]

                # Store the raw solutions for potential later use
                p_data["raw_student_solutions"] = solutions
                p_data["parsed_student_answers"] = parsed_answers

                # Check if this problem meets the consistency criteria,
                # has numbers in the answer, is not too short, and contains no XML tags
                if (
                    args.lower_ratio
                    <= p_data["consistency_ratio"]
                    <= args.higher_ratio
                    and not has_no_numbers(p_data.get("most_popular_answer"))
                    and not is_problem_too_short(p_data.get("problem"))
                    and not contains_xml_tags(p_data.get("problem"))
                ):
                    batch_valid_problems.append(p_data)

            # Add valid problems from this batch to our collection
            valid_candidate_problems.extend(batch_valid_problems)
            print(
                f"    Found {len(batch_valid_problems)} problems within"
                f" consistency range in round {generation_round}"
            )
            print(
                "    Total valid problems so far:"
                f" {len(valid_candidate_problems)}/{n_to_generate}"
            )

            # If we have more than needed, truncate to exactly n_to_generate
            if len(valid_candidate_problems) > n_to_generate:
                valid_candidate_problems = valid_candidate_problems[
                    :n_to_generate
                ]
                print(f"    Truncated to exactly {n_to_generate} problems")
                break

        # Use the valid problems as our candidate_problems for the rest of the pipeline
        candidate_problems = valid_candidate_problems

        profiler.increment_counter(
            "problems_generated", total_problems_generated
        )
        profiler.end_timer("1_problem_generation")

        if len(candidate_problems) < n_to_generate:
            print(
                f"Warning: Only found {len(candidate_problems)} problems"
                " within consistency range out of target"
                f" {n_to_generate} after {generation_round} generation rounds"
                f" and {total_problems_generated} total problems generated."
            )
        else:
            print(
                f"Successfully generated {len(candidate_problems)} problems"
                f" within consistency range after {generation_round} rounds"
                f" and {total_problems_generated} total problems."
            )

        # Skip the separate solution generation and consistency scoring since we already did it
        profiler.increment_counter(
            "solutions_generated",
            len(candidate_problems) * args.num_solution_generations,
        )

        # Select top k problems for this iteration
        selected_problems = []

        # Prepare unified skills list for skill-based diversity
        all_selected_skills = []
        if all_selected_problems:
            all_selected_skills = [
                p.get("concepts", []) for p in all_selected_problems
            ]

        # Track selected indices within current iteration
        selected_indices = set()
        available_indices = set(range(len(candidate_problems)))

        # Track consistency filtering statistics for current iteration only
        iteration_consistency_filtered = 0

        # Milestone 6: Iterative Selection Process
        profiler.start_timer("6_iterative_selection")

        # Simplified iterative selection using vectorized operations
        for selection_step in tqdm(
            range(current_target),
            desc=f"Selecting top-{current_target} problems",
        ):
            # Milestone 6a: Score Computation (per step)
            profiler.start_timer("6a_score_computation_per_step")

            # Print profiling time after each iteration
            if selection_step > 0:
                previous_duration = profiler.timings[
                    "6a_score_computation_per_step"
                ][-1]
                print(
                    f"  Selection step {selection_step}:"
                    f" {previous_duration:.2f}s"
                )
            if not available_indices:
                break

            # Filter problems based on answer-in-question check
            candidate_indices = []
            for idx in available_indices:
                p_data = candidate_problems[idx]

                # New filter: check if the most common answer is in the question as a whole word
                most_popular_answer = p_data.get("most_popular_answer")
                problem_text = p_data.get("problem", "")
                if most_popular_answer is not None:
                    # Escape any special regex characters in the answer string
                    answer_str = re.escape(str(most_popular_answer))
                    # Use word boundaries to ensure a whole word match
                    if re.search(r"\b" + answer_str + r"\b", problem_text):
                        continue  # Skip this problem because the answer is in the question

                candidate_indices.append(idx)

            if not candidate_indices:
                break

            # Compute consistency scores for all candidates
            consistency_scores = np.zeros(len(candidate_indices))
            for i, idx in enumerate(candidate_indices):
                consistency_ratio = candidate_problems[idx].get(
                    "consistency_ratio", 0.0
                )
                if args.lower_ratio <= consistency_ratio <= args.higher_ratio:
                    consistency_scores[i] = 1 - consistency_ratio
                else:
                    consistency_scores[i] = 0.0

            # Compute diversity scores using skill-based approach
            diversity_scores = np.ones(len(candidate_indices))

            if selected_indices or all_selected_skills:
                # Use skill-based diversity with unified skills list
                current_skills = [
                    candidate_problems[idx].get("concepts", [])
                    for idx in range(len(candidate_problems))
                ]

                # Combine previously selected skills with current iteration skills
                combined_selected_skills = all_selected_skills.copy()
                if selected_indices:
                    combined_selected_skills.extend(
                        [
                            candidate_problems[idx].get("concepts", [])
                            for idx in selected_indices
                        ]
                    )

                diversity_scores = compute_skill_diversity_scores(
                    current_skills,
                    combined_selected_skills,
                    candidate_indices,
                )

            # Iterative strategy: filter by consistency, then rank by diversity
            consistency_filtered_indices = [
                idx
                for i, idx in enumerate(candidate_indices)
                if consistency_scores[i] > 0
            ]

            # Count and track problems filtered out due to consistency
            problems_filtered_out = len(candidate_indices) - len(
                consistency_filtered_indices
            )
            iteration_consistency_filtered += problems_filtered_out

            if not consistency_filtered_indices:
                break  # No candidates passed consistency filter

            # Re-calculate diversity scores for the filtered set
            # Note: Diversity scores default to uniform (all ones) since
            # embedding-based diversity has been removed. Skill-based diversity
            # is handled separately above.
            final_diversity_scores = np.ones(len(consistency_filtered_indices))

            best_filtered_idx = np.argmax(final_diversity_scores)
            best_problem_idx = consistency_filtered_indices[best_filtered_idx]

            best_problem = candidate_problems[best_problem_idx]

            # Update tracking
            selected_indices.add(best_problem_idx)
            selected_problems.append(best_problem)
            available_indices.remove(best_problem_idx)

            profiler.increment_counter("selection_steps", 1)
            profiler.end_timer("6a_score_computation_per_step")

        profiler.increment_counter("problems_selected", len(selected_problems))
        profiler.end_timer("6_iterative_selection")

        # Add selected problems from this iteration to the overall collection
        all_selected_problems.extend(selected_problems)

        # Add selected problems to icl_example_pool for next iteration
        for p_data in selected_problems:
            problem = sanitize_text(p_data["problem"])
            icl_example_pool.append({"question": problem})

        profiler.end_timer("iteration_total")
        # Append selected problems to file
        with open(output_file, "a") as f:
            for p_data in selected_problems:
                # Create a clean copy without embeddings for JSON serialization
                clean_data = {
                    k: v
                    for k, v in p_data.items()
                    if k not in ["query_embedding", "key_embedding"]
                }
                f.write(json.dumps(clean_data) + "\n")

        # Print iteration profiling summary
        profiler.print_milestone_summary(iteration + 1)

        print(f"Iteration {iteration + 1} completed:")
        print(f"- Selected {len(selected_problems)} problems")
        print(f"- Total selected so far: {len(all_selected_problems)}")
        print(f"- Example pool size: {len(icl_example_pool)}")
        if iteration_consistency_filtered > 0:
            print(
                "- Problems filtered out due to consistency range:"
                f" {iteration_consistency_filtered}"
            )

        # Print consistency ratio distribution for current iteration
        if candidate_problems:
            consistency_ratios = [
                p.get("consistency_ratio", 0.0) for p in candidate_problems
            ]
            ratio_counts = {}
            for ratio in consistency_ratios:
                if ratio == 0.0:
                    bucket = "0.0"
                elif 0.0 < ratio <= 0.1:
                    bucket = "0.0-0.1"
                elif 0.1 < ratio <= 0.2:
                    bucket = "0.1-0.2"
                elif 0.2 < ratio <= 0.3:
                    bucket = "0.2-0.3"
                elif 0.3 < ratio <= 0.4:
                    bucket = "0.3-0.4"
                elif 0.4 < ratio <= 0.5:
                    bucket = "0.4-0.5"
                elif 0.5 < ratio <= 0.6:
                    bucket = "0.5-0.6"
                elif 0.6 < ratio <= 0.7:
                    bucket = "0.6-0.7"
                elif 0.7 < ratio <= 0.8:
                    bucket = "0.7-0.8"
                elif 0.8 < ratio <= 0.9:
                    bucket = "0.8-0.9"
                elif 0.9 < ratio <= 1.0:
                    bucket = "0.9-1.0"
                else:
                    bucket = "other"

                ratio_counts[bucket] = ratio_counts.get(bucket, 0) + 1

            print("- Consistency ratio distribution for current iteration:")
            for bucket in [
                "0.0",
                "0.0-0.1",
                "0.1-0.2",
                "0.2-0.3",
                "0.3-0.4",
                "0.4-0.5",
                "0.5-0.6",
                "0.6-0.7",
                "0.7-0.8",
                "0.8-0.9",
                "0.9-1.0",
                "other",
            ]:
                if bucket in ratio_counts:
                    print(f"  {bucket}: {ratio_counts[bucket]}")

        iteration += 1

    # Print final profiling summary
    profiler.print_final_summary()

    # Print formatting issues summary
    if total_problems_attempted > 0:
        formatting_issue_ratio = (
            total_formatting_issues / total_problems_attempted
        )
        print(f"\n--- Formatting Issues Summary ---")
        print(f"Total problems attempted: {total_problems_attempted}")
        print(f"Problems with formatting issues: {total_formatting_issues}")
        print(
            "Formatting issue ratio:"
            f" {formatting_issue_ratio:.4f} ({formatting_issue_ratio * 100:.2f}%)"
        )
    else:
        print(f"\nNo problems were attempted during generation.")

    print(f"\nIterative top-k selection completed!")
    print(f"Saved {len(all_selected_problems)} problems to {output_file}")


def openai_request_with_retry(
    api_call_func, max_retries=50, delay_seconds=10, initial_max_tokens=None
):
    """
    Wrapper function to retry OpenAI API calls on connection exceptions and context length errors.

    When context length errors occur, it will automatically halve the max_tokens parameter
    and retry until either success or a minimum token threshold is reached.

    Args:
        api_call_func: A callable that makes the OpenAI API request
        max_retries: Maximum number of retry attempts (default: 50)
        delay_seconds: Delay between retries in seconds (default: 10)
        initial_max_tokens: The initial max_tokens value to use for halving (optional)

    Returns:
        The result of the API call

    Raises:
        The last exception encountered if all retries are exhausted
    """
    min_tokens = 64  # Minimum token threshold to prevent infinite reduction
    current_max_tokens = initial_max_tokens

    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            return api_call_func()
        except openai.APIConnectionError as e:
            if attempt < max_retries:
                logger.warning(
                    "OpenAI connection error (attempt"
                    f" {attempt + 1}/{max_retries + 1}): {e}. Retrying in"
                    f" {delay_seconds} seconds..."
                )
                time.sleep(delay_seconds)
            else:
                logger.error(
                    "OpenAI connection failed after"
                    f" {max_retries + 1} attempts. Last error: {e}"
                )
                raise
        except openai.BadRequestError as e:
            # Check if this is a context length error
            error_message = str(e).lower()
            if any(
                keyword in error_message
                for keyword in [
                    "maximum context length",
                    "context length",
                    "token limit",
                    "too many tokens",
                    "context_length_exceeded",
                    "reduce the length",
                    "completion)",
                    "in the completion",
                ]
            ):
                logger.warning(
                    "Context length error detected (attempt"
                    f" {attempt + 1}/{max_retries + 1}): {e}"
                )

                # Use the provided initial_max_tokens or try to extract from closure as fallback
                if current_max_tokens is None:
                    # Try to extract from closure as fallback
                    if (
                        hasattr(api_call_func, "__closure__")
                        and api_call_func.__closure__
                    ):
                        closure_vars = {}
                        if api_call_func.__code__.co_freevars:
                            for i, var_name in enumerate(
                                api_call_func.__code__.co_freevars
                            ):
                                if i < len(api_call_func.__closure__):
                                    closure_vars[var_name] = (
                                        api_call_func.__closure__[
                                            i
                                        ].cell_contents
                                    )

                        # Look for max_tokens in closure
                        for var_name, var_value in closure_vars.items():
                            if var_name == "max_tokens" or (
                                hasattr(var_value, "max_tokens")
                            ):
                                current_max_tokens = getattr(
                                    var_value, "max_tokens", var_value
                                )
                                logger.info(
                                    "Found max_tokens in closure:"
                                    f" {current_max_tokens}"
                                )
                                break

                    # Final fallback
                    if current_max_tokens is None:
                        current_max_tokens = 8192  # Default fallback
                        logger.warning(
                            "Could not extract max_tokens, using default:"
                            f" {current_max_tokens}"
                        )

                # Halve the max_tokens
                new_max_tokens = max(current_max_tokens // 2, min_tokens)

                if new_max_tokens < min_tokens:
                    logger.error(
                        "Context length error: max_tokens reduced to minimum"
                        f" ({min_tokens}), but still failing. Cannot reduce"
                        f" further. Original error: {e}"
                    )
                    raise

                logger.warning(
                    f"Reducing max_tokens from {current_max_tokens} to"
                    f" {new_max_tokens} and retrying..."
                )

                # Create a new lambda function with reduced max_tokens
                api_call_func = _create_reduced_token_lambda(
                    api_call_func, new_max_tokens
                )
                current_max_tokens = (
                    new_max_tokens  # Update for next iteration
                )

            else:
                # Non-context length BadRequestError, don't retry
                logger.error(f"OpenAI API BadRequestError (non-context): {e}")
                raise
        except Exception as e:
            # For other non-connection errors, don't retry
            logger.error(f"OpenAI API error (non-connection): {e}")
            raise


def _create_reduced_token_lambda(original_lambda, new_max_tokens):
    """
    Helper function to create a new lambda with reduced max_tokens.
    This works by introspecting the original lambda and creating a new one with modified parameters.
    """
    try:
        # Get the closure variables from the original lambda
        closure_vars = {}
        if (
            original_lambda.__closure__
            and original_lambda.__code__.co_freevars
        ):
            for i, var_name in enumerate(original_lambda.__code__.co_freevars):
                if i < len(original_lambda.__closure__):
                    closure_vars[var_name] = original_lambda.__closure__[
                        i
                    ].cell_contents

        logger.debug(f"Closure variables found: {list(closure_vars.keys())}")

        def reduced_lambda():
            # Try to reconstruct the call with reduced max_tokens
            # Look for client in closure
            client = None
            model = None
            prompt = None
            temperature = None
            n = None
            seed = None

            # More robust parameter detection
            for var_name, var_value in closure_vars.items():
                if hasattr(
                    var_value, "completions"
                ):  # This is likely the client
                    client = var_value
                    logger.debug(f"Found client: {var_name}")
                elif isinstance(var_value, str):
                    if "/" in var_value and (
                        "model" in var_name.lower()
                        or len(var_value.split("/")) >= 2
                    ):
                        model = var_value
                        logger.debug(f"Found model: {var_name} = {var_value}")
                elif isinstance(var_value, (list, str)):
                    if "prompt" in var_name.lower() or isinstance(
                        var_value, list
                    ):
                        prompt = var_value
                        logger.debug(
                            f"Found prompt: {var_name} (type:"
                            f" {type(var_value)})"
                        )
                elif isinstance(var_value, (int, float)):
                    if "temperature" in var_name.lower():
                        temperature = var_value
                        logger.debug(
                            f"Found temperature: {var_name} = {var_value}"
                        )
                    elif (
                        var_name in ["n", "num_generations"]
                        or "generation" in var_name.lower()
                    ):
                        n = var_value
                        logger.debug(f"Found n: {var_name} = {var_value}")
                    elif "seed" in var_name.lower():
                        seed = var_value
                        logger.debug(f"Found seed: {var_name} = {var_value}")

            if client and hasattr(client, "completions"):
                # Build the API call parameters
                params = {"max_tokens": new_max_tokens}

                # Add other parameters if found
                if model:
                    params["model"] = model
                if prompt is not None:
                    params["prompt"] = prompt
                if temperature is not None:
                    params["temperature"] = temperature
                if n is not None:
                    params["n"] = n
                if seed is not None:
                    params["seed"] = seed

                logger.debug(
                    "Reconstructed API call with params:"
                    f" {list(params.keys())}"
                )
                return client.completions.create(**params)
            else:
                logger.warning(
                    "Could not find client in closure, falling back to"
                    " original lambda"
                )
                # Fallback: try to call original and hope for the best
                return original_lambda()

        return reduced_lambda

    except Exception as e:
        logger.warning(
            f"Failed to create reduced token lambda: {e}. Using original"
            " lambda."
        )
        return original_lambda


def create_openai_completion_call(
    client, model, prompt, max_tokens, temperature=None, n=None, seed=None
):
    """
    Helper function to create a structured OpenAI completion call that works well with the retry mechanism.

    Args:
        client: OpenAI client instance
        model: Model name string
        prompt: Prompt text or list of prompts
        max_tokens: Maximum tokens for completion
        temperature: Temperature parameter (optional)
        n: Number of completions (optional)
        seed: Random seed (optional)

    Returns:
        A lambda function that can be used with openai_request_with_retry
    """

    def completion_call():
        params = {"model": model, "prompt": prompt, "max_tokens": max_tokens}

        if temperature is not None:
            params["temperature"] = temperature
        if n is not None:
            params["n"] = n
        if seed is not None:
            params["seed"] = seed

        return client.completions.create(**params)

    return completion_call


def main(args):
    # Configuration
    main_port_number = args.main_port_number

    main_model = args.main_model
    main_tokenizer = AutoTokenizer.from_pretrained(main_model)
    main_client = openai.OpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{main_port_number}/v1",
    )

    # Call top_k_selection for problem generation
    top_k_selection(
        args,
        main_client,
        main_model,
        main_tokenizer,
        include_system=args.include_system,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate problems using teacher-student consistency filtering."
        )
    )
    parser.add_argument(
        "--main_port_number",
        type=int,
        default=8000,
        help="Port number for the main model API server.",
    )
    parser.add_argument(
        "--initial_batch_size",
        type=int,
        default=32,
        help="Initial batch size for generating problems.",
    )
    parser.add_argument(
        "--n_target_problems",
        type=int,
        default=256,
        help="Number of problems to generate.",
    )
    parser.add_argument(
        "--lower_ratio",
        type=float,
        default=0.5,
        help="Lower bound for consistency ratio filtering.",
    )
    parser.add_argument(
        "--higher_ratio",
        type=float,
        default=0.7,
        help="Higher bound for consistency ratio filtering.",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default=None,
        help="Name of the output file.",
    )
    parser.add_argument(
        "--difficulty",
        action="store_true",
        help="Whether to generate difficult problems.",
    )
    parser.add_argument(
        "--n_icl",
        type=int,
        default=1,
        help="Number of examples to use for in-context learning.",
    )
    # Arguments for top_k selection strategy
    parser.add_argument(
        "--scale_factor",
        type=int,
        default=5,
        help="Factor to scale n_target_problems for initial generation.",
    )
    parser.add_argument(
        "--num_solution_generations",
        type=int,
        default=8,
        help=(
            "Number of solution generations per problem for consistency"
            " checking."
        ),
    )
    parser.add_argument(
        "--main_model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Main model name for problem generation and solving.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="Maximum tokens for main model completions.",
    )
    parser.add_argument(
        "--include_system",
        action="store_true",
        help="Include system prompt in teacher prompt.",
    )

    args = parser.parse_args()
    main(args)
