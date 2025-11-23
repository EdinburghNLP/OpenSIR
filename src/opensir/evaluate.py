# queue add "micromamba run -n open-r1 python /dev/volume/code/public-open-r1/src/opensir/evaluate.py --model /dev/volume/code/public-open-r1/saves/Opensir_Llama-3.2-3B-Instruct/checkpoint-25  --output_fn /dev/volume/code/public-open-r1/results/Opensir_Llama-3.2-3B-Instruct_eval-outputs.jsonl --result_fn /dev/volume/code/public-open-r1/results/Opensir_Llama-3.2-3B-Instruct_eval-results.jsonl --n 16 --temp 0.6 --top_p 0.95" --n_gpus 1 --output_file /dev/volume/code/public-open-r1/logs/Opensir_Llama-3.2-3B-Instruct_eval.log

# CUDA_VISIBLE_DEVICES=3 python /dev/volume/code/public-open-r1/src/opensir/evaluate.py --model /dev/volume/code/public-open-r1/saves/Opensir_Llama-3.2-3B-Instruct/checkpoint-25  --output_fn /dev/volume/code/public-open-r1/results/Opensir_Llama-3.2-3B-Instruct_eval-outputs.jsonl --result_fn /dev/volume/code/public-open-r1/results/Opensir_Llama-3.2-3B-Instruct_eval-results.jsonl --n 16 --temp 0.6 --top_p 0.95
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from typing import Union, Literal, List, Dict, Any
from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from tabulate import tabulate
from tqdm import tqdm
import os

from strictfire import StrictFire
from math_verify import (
    LatexExtractionConfig,
    ExprExtractionConfig,
    parse,
    verify,
)
import re
import torch
import json
import wandb


def calculate_accuracy_by_task(
    save_outputs: List[Dict[str, Any]],
) -> Dict[str, float]:
    task_accuracy = {}

    for output in save_outputs:
        task = output["task"]
        if task not in task_accuracy:
            task_accuracy[task] = {"correct": 0, "total": 0}

        for is_correct in output["corrects"]:
            task_accuracy[task]["total"] += 1
            if is_correct:
                task_accuracy[task]["correct"] += 1

    # Calculate accuracy for each task
    for task in task_accuracy:
        task_accuracy[task] = (
            task_accuracy[task]["correct"] * 100 / task_accuracy[task]["total"]
            if task_accuracy[task]["total"] > 0
            else 0.0
        )
    return task_accuracy


def get_message(
    question: str, use_system: bool = True
) -> List[Dict[str, str]]:
    messages = []
    messages.append(
        {
            "role": "system",
            "content": (
                "You are a helpful AI Assistant, designed to provide"
                " well-reasoned and detailed responses. You FIRST think"
                " about the reasoning process step by step and then"
                " provide the user with the answer. The last line of your"
                " response should be 'Therefore, the final answer is:"
                " $\\boxed{ANSWER}$' (without quotes) where ANSWER is just"
                " the final number or expression that solves the problem."
            ),
        }
    )
    messages.append({"role": "user", "content": question})
    if not use_system and messages[0]["role"] == "system":
        messages = [
            {
                "role": "user",
                "content": (
                    messages[0]["content"] + "\n\n" + messages[1]["content"]
                ),
            }
        ]
    return messages


def main(
    model: str,
    output_fn: str,
    result_fn: str = None,
    lora_path: str = None,
    max_lora_rank: int = 8,
    max_tokens: int = 4096,
    temp: float = 0,
    top_p: float = 1.0,
    seed: int = 43,
    apply_chat_template: bool = True,
    n: int = 1,
    end_tokens: List[str] = [],
    n_limit: int = -1,
    overwrite: bool = False,
    tasks: List[
        Literal[
            "gsm8k",
            "math-500",
            "minerva_math",
            "olympiadbench",
            "college_math",
        ]
    ] = [
        "gsm8k",
        "math-500",
        "minerva_math",
        "olympiadbench",
        "college_math",
    ],
):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Modify output_fn to save in the results directory
    os.makedirs(os.path.dirname(output_fn), exist_ok=True)
    # If output_fn is None or empty, StrictFire would typically raise an error
    # if it's a required arg without a default. Assuming it's always provided.

    # Set default result_fn if not provided
    if result_fn is None:
        model_name_for_results = model.split("/")[-1]
        clean_model_name = re.sub(r"[^\w\.-]", "_", model_name_for_results)
        result_fn = os.path.join(
            results_dir, f"{clean_model_name}_eval-results.jsonl"
        )

    # Check if both files exist and skip if overwrite is False
    if (
        not overwrite
        and os.path.exists(output_fn)
        and os.path.exists(result_fn)
    ):
        print(f"Skipping evaluation - both output files already exist:")
        print(f"  Output file: {output_fn}")
        print(f"  Result file: {result_fn}")
        print("Use --overwrite to force re-evaluation.")
        return

    use_system = "gemma" not in model
    messages, meta_datas = [], []
    if "math-500" in tasks:
        print("Processing task: math-500")
        ds = load_dataset("HuggingFaceH4/MATH-500", "default", split="test")
        for i, row in enumerate(ds):
            meta_datas.append(
                {
                    "id": f"math-500_{i}",
                    "ori_id": row["unique_id"],
                    "problem": row["problem"],
                    "processed_answer": parse(
                        "$" + row["answer"] + "$",
                        extraction_config=[LatexExtractionConfig()],
                        fallback_mode="first_match",
                        extraction_mode="first_match",
                    ),
                    "answer": row["answer"],
                    "solution": row["solution"],
                }
            )
            messages.append(get_message(row["problem"], use_system=use_system))

    if "gsm8k" in tasks:
        print("Processing task: gsm8k")
        ds = load_dataset("openai/gsm8k", "main", split="test")
        for i, row in enumerate(ds):
            meta_datas.append(
                {
                    "id": f"gsm8k_{i}",
                    "problem": row["question"],
                    "processed_answer": parse(
                        "$" + row["answer"].split("####")[-1].strip() + "$",
                        extraction_config=[LatexExtractionConfig()],
                        fallback_mode="first_match",
                        extraction_mode="first_match",
                    ),
                    "answer": row["answer"].split("####")[-1].strip(),
                }
            )
            messages.append(
                get_message(row["question"], use_system=use_system)
            )

    # Olympiad Bench
    if "olympiadbench" in tasks:
        print("Processing task: olympiadbench")
        ds = load_dataset(
            "Hothan/OlympiadBench",
            "OE_TO_maths_en_COMP",
            trust_remote_code=True,
            split="train",
        )
        for i, row in enumerate(ds):
            meta_datas.append(
                {
                    "id": f"olympiad-bench_{i}",
                    "ori_id": row["id"],
                    "problem": row["question"],
                    "processed_answer": parse(
                        row["final_answer"][0],
                        extraction_config=[
                            LatexExtractionConfig(),
                            ExprExtractionConfig(),
                        ],
                        fallback_mode="first_match",
                        extraction_mode="first_match",
                    ),
                    "solution": row["solution"][0],
                }
            )
            messages.append(
                get_message(row["question"], use_system=use_system)
            )

    if "minerva_math" in tasks:
        print("Processing task: minerva_math")
        ds = load_dataset("math-ai/minervamath", "default", split="test")
        for i, row in enumerate(ds):
            meta_datas.append(
                {
                    "id": f"minerva-math_{i}",
                    "problem": row["question"],
                    "answer": row["answer"],
                    "processed_answer": parse(
                        "$" + row["answer"] + "$",
                        extraction_config=[LatexExtractionConfig()],
                        fallback_mode="first_match",
                        extraction_mode="first_match",
                    ),
                }
            )
            messages.append(
                get_message(row["question"], use_system=use_system)
            )

    if "college_math" in tasks:
        print("Processing task: college_math")
        ds = load_dataset("di-zhang-fdu/College_Math_Test", split="test")
        for i, row in enumerate(ds):
            meta_datas.append(
                {
                    "id": f"college-math_{i}",
                    "problem": row["question"],
                    "answer": row["answer"],
                    "processed_answer": parse(
                        row["answer"],
                        extraction_config=[
                            LatexExtractionConfig(),
                            ExprExtractionConfig(),
                        ],
                        fallback_mode="first_match",
                        extraction_mode="first_match",
                    ),
                }
            )
            messages.append(
                get_message(row["question"], use_system=use_system)
            )

    assert len(meta_datas) == len(messages)
    if n_limit > 0:
        messages = messages[:n_limit]  # Corrected: slice messages
        meta_datas = meta_datas[:n_limit]

    # LLM and tokenizer initialization was here in the original code,
    # and needs to be before 'apply_chat_template' block.
    # 'prompts' will be defined based on 'apply_chat_template' below.
    if lora_path is not None:
        llm = LLM(
            model=model,
            tensor_parallel_size=torch.cuda.device_count(),
            dtype="float16",
            enable_lora=True,
            max_lora_rank=max_lora_rank,
            seed=seed,
            gpu_memory_utilization=0.85,
            # max_model_len=32768,
        )
    else:
        llm = LLM(
            model=model,
            tensor_parallel_size=torch.cuda.device_count(),
            dtype="float16",
            seed=seed,
            gpu_memory_utilization=0.85,
            # max_model_len=32768,
        )
    tokenizer = llm.get_tokenizer()
    if apply_chat_template:
        prompts = [
            tokenizer.apply_chat_template(
                conversation=m,
                tokenize=False,
                add_generation_prompt=True,
            )
            for m in messages
        ]
    else:
        # If not applying chat template, assign messages to prompts.
        # Note: This will make 'prompts' a List[List[Dict[str, str]]],
        # which might not be what llm.generate expects if it's not mocked.
        # However, it fixes the UnboundLocalError for the test.
        prompts = messages

    print("First prompt:")
    if not prompts:  # Added safety check
        # This case should ideally be handled earlier, e.g. if messages is empty.
        # For now, raising an error if prompts ends up empty.
        raise ValueError("Prompts list is empty. Cannot print prompts[0].")
    print(prompts[0])
    sampling_params = SamplingParams(
        temperature=temp,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=end_tokens,
        n=n,
    )

    outputs = llm.generate(
        prompts=prompts,
        sampling_params=sampling_params,
        lora_request=(
            LoRARequest("lora", 1, lora_path) if lora_path else None
        ),
    )
    predictions = [
        [  # Start of inner list for multiple outputs per prompt
            parse(
                choice.text,  # Iterate over 'choice' in 'o.outputs'
                extraction_config=[
                    ExprExtractionConfig(),
                    LatexExtractionConfig(
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    ),
                ],
                extraction_mode="first_match",
                fallback_mode="first_match",
                parsing_timeout=1,
            )
            for choice in o.outputs  # Loop over each output for the current prompt 'o'
        ]
        for o in outputs
    ]
    references = []
    for i, meta_data_item in enumerate(meta_datas):
        # outputs[i] is the RequestOutput for the i-th problem.
        # outputs[i].outputs is the list of CompletionOutputs (generated sequences).
        num_generated_outputs_for_problem = len(outputs[i].outputs)
        reference_answer = meta_data_item["processed_answer"]
        references.append(
            [reference_answer] * num_generated_outputs_for_problem
        )

    flat_correctness_results = []
    # Store lengths of inner lists to help reshape later
    inner_list_lengths = []

    for i, pred_list in enumerate(tqdm(predictions, desc="Verifying answers")):
        ref_list = references[i]
        inner_list_lengths.append(len(pred_list))
        for j, pred_item in enumerate(pred_list):
            try:
                is_correct = verify(
                    gold=ref_list[j], target=pred_item, timeout_seconds=10
                )
                # Handle None return from verify as False
                flat_correctness_results.append(
                    False if is_correct is None else is_correct
                )
            except TimeoutError:
                print(
                    f"Timeout while processing: {pred_item} vs {ref_list[j]}"
                )
                flat_correctness_results.append(False)
            except Exception as e:
                print(
                    f"Error during verification for '{pred_item}' vs"
                    f" '{ref_list[j]}': {e}"
                )
                flat_correctness_results.append(False)

    # Reshape flat_correctness_results back to nested structure
    all_corrects_reshaped = []
    current_pos = 0
    for length in inner_list_lengths:
        all_corrects_reshaped.append(
            flat_correctness_results[current_pos : current_pos + length]
        )
        current_pos += length

    # Enrich meta_data with per-item details
    for i, meta_data_item in enumerate(meta_datas):
        meta_data_item["outputs"] = [
            choice.text for choice in outputs[i].outputs
        ]
        meta_data_item["parsed_outputs"] = predictions[i]
        meta_data_item["is_correct"] = all_corrects_reshaped[
            i
        ]  # all_corrects_reshaped is List[List[bool]]
        meta_data_item["prompt"] = prompts[i]  # Add the actual prompt used

    # Calculate Dataset-Level acc@n
    dataset_grouped_correctness = {}
    for i, item in enumerate(meta_datas):
        # Extract dataset name from item['id'] (e.g., 'gsm8k' from 'gsm8k_0')
        dataset_name = item["id"].split("_")[0]
        if dataset_name not in dataset_grouped_correctness:
            dataset_grouped_correctness[dataset_name] = []
        # item['is_correct'] is List[bool] for multiple generations of this single item
        dataset_grouped_correctness[dataset_name].append(item["is_correct"])

    dataset_metrics = {}
    for (
        dataset_name,
        all_item_correctness_lists,
    ) in dataset_grouped_correctness.items():
        if not all_item_correctness_lists:
            continue
        dataset_metrics[dataset_name] = {}

        # Calculate single average accuracy: total correct responses / total responses
        total_correct = 0
        total_responses = 0

        for item_flags_list in all_item_correctness_lists:
            if item_flags_list:
                total_correct += sum(item_flags_list)
                total_responses += len(item_flags_list)

        average_accuracy = (
            total_correct / total_responses if total_responses > 0 else 0.0
        )
        dataset_metrics[dataset_name]["average_accuracy"] = average_accuracy

    # Calculate overall metrics (macro average across all datasets)
    overall_metrics = {}
    if dataset_metrics:
        # Collect all unique metric keys first
        all_metric_keys_for_overall = set()
        for metrics in dataset_metrics.values():
            all_metric_keys_for_overall.update(metrics.keys())

        # Calculate mean for each metric across all datasets
        for metric_key in all_metric_keys_for_overall:
            values = []
            for ds_name in dataset_metrics:
                if metric_key in dataset_metrics[ds_name]:
                    values.append(dataset_metrics[ds_name][metric_key])
            if values:
                overall_metrics[metric_key] = sum(values) / len(values)

    # Print Markdown Table
    if dataset_metrics:
        sorted_dataset_names = sorted(dataset_metrics.keys())

        # Collect all unique metric keys (e.g., 'acc@1', 'acc@2') and sort them naturally
        # (e.g., acc@1, acc@2, acc@10 not acc@1, acc@10, acc@2)
        all_metric_keys = set()
        for metrics in dataset_metrics.values():
            all_metric_keys.update(metrics.keys())

        def sort_metric_keys(key):
            match = re.match(r"acc@(\d+)", key)
            return int(match.group(1)) if match else float("inf")

        sorted_metric_keys = sorted(
            list(all_metric_keys), key=sort_metric_keys
        )

        table_headers = ["Metric"] + sorted_dataset_names + ["Overall"]
        table_rows = []
        for m_key in sorted_metric_keys:
            row = [m_key] + [
                (
                    f"{dataset_metrics[ds_name].get(m_key, 'N/A'):.4f}"
                    if isinstance(dataset_metrics[ds_name].get(m_key), float)
                    else "N/A"
                )
                for ds_name in sorted_dataset_names
            ]
            # Add overall average for this metric
            overall_value = overall_metrics.get(m_key)
            if isinstance(overall_value, float):
                row.append(f"{overall_value:.4f}")
            else:
                row.append("N/A")
            table_rows.append(row)

        print("\n--- Evaluation Summary ---")
        print(
            tabulate(
                [table_headers] + table_rows,
                headers="firstrow",
                tablefmt="pipe",
            )
        )
        print("------------------------\n")

    # Append to Aggregated Results JSONL (now per-model, in results_dir, with overwrite)
    # result_fn is already set at the beginning of the function

    new_records_for_model = []
    for dataset_name_agg, metrics_agg in dataset_metrics.items():
        for metric_key_agg, metric_value_agg in metrics_agg.items():
            record = {
                "dataset": dataset_name_agg,
                "metric": metric_key_agg,
                "value": metric_value_agg,
                # "model_name" field is removed as it's in the filename
            }
            new_records_for_model.append(record)

    all_records_for_model = {}  # Using a dict: (dataset, metric) -> record
    os.makedirs(os.path.dirname(result_fn), exist_ok=True)
    if os.path.exists(result_fn):
        with open(result_fn, "r") as f:
            for line in f:
                try:
                    existing_record = json.loads(line)
                    # Ensure records have 'dataset' and 'metric' keys
                    if (
                        "dataset" in existing_record
                        and "metric" in existing_record
                    ):
                        all_records_for_model[
                            (
                                existing_record["dataset"],
                                existing_record["metric"],
                            )
                        ] = existing_record
                    else:
                        print(
                            "Warning: Record missing 'dataset' or 'metric' in"
                            f" {result_fn}: {line.strip()}"
                        )
                except json.JSONDecodeError:
                    print(
                        "Warning: Could not decode line in"
                        f" {result_fn}: {line.strip()}"
                    )

    # Add/update with new records
    for record_to_add in new_records_for_model:
        all_records_for_model[
            (record_to_add["dataset"], record_to_add["metric"])
        ] = record_to_add

    # Write all records back
    if all_records_for_model:
        with open(result_fn, "w") as f:  # "w" to overwrite the file
            for record_to_write in all_records_for_model.values():
                f.write(json.dumps(record_to_write) + "\n")
        print(f"Aggregated results saved to {result_fn}")
    elif os.path.exists(result_fn):  # If file exists but is now empty
        with open(result_fn, "w") as f:  # Clear the file
            f.write("")  # Write empty string to ensure it's an empty file
        print(
            f"Aggregated results file {result_fn} cleared as no"
            " records to save."
        )
    else:
        # File doesn't exist and no records to save, do nothing or print a specific message
        print(
            f"No aggregated results to save for {result_fn} (file"
            " not created)."
        )

    # Save enriched meta_data (per-item details) to output_fn
    # Ensure output_fn is not None before writing
    if output_fn:
        with open(output_fn, "w") as f:
            for item in meta_datas:
                # processed_answers
                for k in ["parsed_outputs", "processed_answer"]:
                    item[k] = [str(ans) for ans in item[k]]
                f.write(json.dumps(item) + "\n")
    else:
        print(
            "Warning: output_fn is not defined. Detailed per-item results not"
            " saved."
        )
    print(f"Detailed per-item results saved to {output_fn}")


if __name__ == "__main__":
    StrictFire(main)
