# OpenSIR: Open-Ended Self-Improving Reasoner
SONER


## Setup

```bash
pip install vllm==0.7.2 setuptools rootutils strictfire
pip install flash-attn --no-build-isolation
pip install flashinfer-python==0.2.2 -i https://flashinfer.ai/whl/cu124/torch2.5/
GIT_LFS_SKIP_SMUDGE=1 pip install -e ".[dev]"
```



## Train
We first generate a small set of questions that are within the solve rate range as the intial example pool.

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Llama-3.2-3B-Instruct --gpu-memory-utilization 0.60 --port 8000
python src/opensir/generate_initial_problems.py --main_model meta-llama/Llama-3.2-3B-Instruct --max_tokens 2048 --main_port_number 8000 --lower_ratio 0.5 \
    --scale_factor 5 \
    --higher_ratio 0.9  --n_target_problems 512 --initial_batch_size 32 --output_filename "src/opensir/data/initial_data_new.jsonl"
```

Then, we proceed to train the model with GRPO using self-play
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Linq-AI-Research/Linq-Embed-Mistral --task embed --gpu-memory-utilization 0.25 --port 8001

VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.6 \
  --max-model-len 8192 

CUDA_VISIBLE_DEVICES=1,2 NCCL_DEBUG=WARN HF_HUB_ENABLE_HF_TRANSFER=1 ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes 2 \
    src/opensir/train.py --config configs/opensir_train.yaml
```


## Evaluate
```bash
python evaluate.py --model saves/Opensir_Llama-3.2-3B-Instruct/checkpoint-200  --output_fn results/Opensir_Llama-3.2-3B-Instruct_eval-outputs.jsonl --result_fn results/Opensir_Llama-3.2-3B-Instruct_eval-results.jsonl --n 16 --temp 0.6 --top_p 0.95
```