
# Required dependencies:
# pip install unsloth torch pandas numpy datasets trl pydantic seaborn matplotlib scikit-learn groq pyarrow
# Note: Set GROQ_API_KEY environment variable before running

# =============================================================================
# GPU AUTO-SELECTION & FAILOVER (must run BEFORE any CUDA/torch imports)
# =============================================================================
import os
import sys
import subprocess as _sp_gpu

def _get_free_gpus(skip_indices=None):
    """Query nvidia-smi for GPU status. Returns list of dicts sorted by free memory (desc)."""
    skip_indices = set(skip_indices or [])
    try:
        result = _sp_gpu.run(
            ['nvidia-smi', '--query-gpu=index,memory.free,memory.total,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(',')]
            try:
                idx = int(parts[0])
                free_mb = int(parts[1])
                total_mb = int(parts[2])
                util_pct = int(parts[3])
            except (ValueError, IndexError):
                continue
            if idx in skip_indices:
                continue
            gpus.append({'index': idx, 'free_mb': free_mb, 'total_mb': total_mb, 'util_pct': util_pct})
        gpus.sort(key=lambda g: g['free_mb'], reverse=True)
        return gpus
    except Exception as e:
        print(f"[GPU] WARNING: Could not query GPUs via nvidia-smi: {e}")
        return []

# --- Pre-scan CLI for GPU-related args (before full argparse) ---
_gpu_explicit = None
_gpu_failover = '--gpu-failover' in sys.argv
_skip_gpus_str = None
OOM_EXIT_CODE = 77  # Special exit code to signal OOM for failover wrapper

for _i, _a in enumerate(sys.argv):
    if _a == '--gpu' and _i + 1 < len(sys.argv):
        _gpu_explicit = int(sys.argv[_i + 1])
    if _a.startswith('--skip-gpus='):
        _skip_gpus_str = _a.split('=', 1)[1]

_skip_set = set()
if _skip_gpus_str:
    _skip_set = {int(x) for x in _skip_gpus_str.split(',') if x.strip()}

if _gpu_explicit is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(_gpu_explicit)
    print(f"[GPU] Using GPU {_gpu_explicit} (set via --gpu)")
elif 'CUDA_VISIBLE_DEVICES' not in os.environ:
    _all_gpus = _get_free_gpus(skip_indices=_skip_set)
    # Consider a GPU "free" if utilization < 30% and > 15 GB free
    _candidates = [g for g in _all_gpus if g['util_pct'] < 30 and g['free_mb'] > 15000]
    if _candidates:
        _chosen = _candidates[0]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(_chosen['index'])
        print(f"[GPU] Auto-selected GPU {_chosen['index']} "
              f"({_chosen['free_mb']}MB free, {_chosen['util_pct']}% util)")
    elif _all_gpus:
        # Fallback: pick GPU with most free memory regardless of utilization
        _chosen = _all_gpus[0]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(_chosen['index'])
        print(f"[GPU] WARNING: No idle GPU found. Using GPU {_chosen['index']} "
              f"({_chosen['free_mb']}MB free, {_chosen['util_pct']}% util)")
    else:
        print("[GPU] WARNING: Could not detect any GPUs — using system default")
else:
    print(f"[GPU] Using CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} (pre-set)")

print(f"[GPU] Failover mode: {'ON' if _gpu_failover else 'OFF'}")

# =============================================================================
# Now safe to import CUDA-dependent libraries
# =============================================================================
from unsloth import FastLanguageModel
import torch
import subprocess
import argparse
from datetime import datetime

# =============================================================================
# CLI ARGUMENTS
# =============================================================================
parser = argparse.ArgumentParser(
    description='Train and/or Evaluate Qwen3-14B with SFT, GRPO, and multi-model comparison',
    formatter_class=argparse.RawTextHelpFormatter,
)

# --- Training mode (mutually exclusive) ---
train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--only-sft', action='store_true',
                         help='Run only SFT training')
train_group.add_argument('--only-grpo', action='store_true',
                         help='Run only GRPO training (requires SFT checkpoint)')
train_group.add_argument('--both-sft-grpo', action='store_true',
                         help='Run both SFT and GRPO training')

# --- Evaluation mode ---
parser.add_argument('--only-eval', action='store_true',
                    help='Skip all training, only run evaluation.\n'
                         'Requires existing GRPO LoRA checkpoint.')
parser.add_argument('--eval-local', action='store_true',
                    help='Include local models (Base + GRPO) in evaluation')
parser.add_argument('--eval-groq', action='store_true',
                    help='Include Groq-hosted models in evaluation')

# --- GPU options ---
parser.add_argument('--gpu', type=int, default=None,
                    help='Explicit GPU index to use (sets CUDA_VISIBLE_DEVICES)')
parser.add_argument('--gpu-failover', action='store_true',
                    help='If training crashes with OOM, automatically retry on the next free GPU')
parser.add_argument('--skip-gpus', type=str, default=None,
                    help='Comma-separated GPU indices to skip (e.g. "0,2")')

# --- Other options ---
parser.add_argument('--reward-type', type=str, default='hybrid',
                    choices=['hybrid', 'complexity', 'all'],
                    help='Reward function type. Use "all" to run hybrid and complexity\n'
                         'sequentially in a single process (default: hybrid)')
parser.add_argument('--skip-smoke-test', action='store_true',
                    help='Skip the pre-training smoke test for all models')
parser.add_argument('--smoke-train', action='store_true',
                    help='Quick pipeline verification: run only 5 SFT + 5 GRPO steps, then eval.\n'
                         'Use to verify the entire pipeline works before committing to full training.')
parser.add_argument('--eval-samples', type=int, default=500,
                    help='Number of evaluation samples (default: 500)')
parser.add_argument('--sft-steps', type=int, default=None,
                    help='Override SFT max_steps (default: 500, or 5 with --smoke-train)')
parser.add_argument('--grpo-steps', type=int, default=None,
                    help='Override GRPO max_steps (default: 1000, or 5 with --smoke-train)')

args = parser.parse_args()

# --- Derive effective mode ---
RUN_TRAINING = not args.only_eval
RUN_EVAL = args.only_eval or args.eval_local or args.eval_groq

# Default training: both SFT+GRPO when no flags given
if RUN_TRAINING and not args.only_sft and not args.only_grpo and not args.both_sft_grpo:
    args.both_sft_grpo = True

# If --only-eval is set but no eval sub-flags, run both local and groq
if args.only_eval and not args.eval_local and not args.eval_groq:
    args.eval_local = True
    args.eval_groq = True

# After training, automatically run evaluation unless user opted out
if RUN_TRAINING and not args.only_eval:
    if not args.eval_local and not args.eval_groq:
        args.eval_local = True
        args.eval_groq = True
        RUN_EVAL = True

if args.reward_type == 'all':
    REWARD_TYPES_LIST = ['hybrid', 'complexity']
else:
    REWARD_TYPES_LIST = [args.reward_type]
REWARD_TYPE = REWARD_TYPES_LIST[0]  # Will be updated in the loop

# --- Derive step counts ---
if args.smoke_train:
    SFT_MAX_STEPS = args.sft_steps if args.sft_steps is not None else 5
    GRPO_MAX_STEPS = args.grpo_steps if args.grpo_steps is not None else 5
    EVAL_SAMPLES = min(args.eval_samples, 50)  # Cap eval at 50 for smoke test
    print("[SMOKE TRAIN] Pipeline verification mode — minimal steps")
else:
    SFT_MAX_STEPS = args.sft_steps if args.sft_steps is not None else 500
    GRPO_MAX_STEPS = args.grpo_steps if args.grpo_steps is not None else 1000
    EVAL_SAMPLES = args.eval_samples

# =============================================================================
# GROQ MODEL REGISTRY
# =============================================================================
GROQ_EVAL_MODELS = [
    {"id": "kimi-k2",        "model": "moonshotai/kimi-k2-instruct-0905",                "display": "Kimi-K2",              "extra_kwargs": {}},
    {"id": "llama4-scout",   "model": "meta-llama/llama-4-scout-17b-16e-instruct",       "display": "Llama-4-Scout-17B",    "extra_kwargs": {}},
    {"id": "llama4-maverick","model": "meta-llama/llama-4-maverick-17b-128e-instruct",   "display": "Llama-4-Maverick-17B", "extra_kwargs": {}},
    {"id": "llama33-70b",    "model": "llama-3.3-70b-versatile",                         "display": "Llama-3.3-70B",        "extra_kwargs": {}},
    {"id": "gpt-oss-120b",   "model": "openai/gpt-oss-120b",                             "display": "GPT-OSS-120B",         "extra_kwargs": {"reasoning_effort": "medium"}},
]

print("=" * 60)
if RUN_TRAINING:
    print("MODE: TRAINING", end="")
    if args.only_sft:
        print(" (SFT only)")
    elif args.only_grpo:
        print(" (GRPO only)")
    else:
        print(" (SFT + GRPO)")
if RUN_EVAL:
    flags = []
    if args.eval_local:
        flags.append("local")
    if args.eval_groq:
        flags.append("groq")
    print(f"MODE: EVALUATION ({' + '.join(flags)})")
print(f"REWARD FUNCTION(S): {', '.join(r.upper() for r in REWARD_TYPES_LIST)}")
if args.smoke_train:
    print(f"SMOKE TRAIN: SFT={SFT_MAX_STEPS} steps, GRPO={GRPO_MAX_STEPS} steps, Eval={EVAL_SAMPLES} samples")
else:
    print(f"STEPS: SFT={SFT_MAX_STEPS}, GRPO={GRPO_MAX_STEPS}, Eval samples={EVAL_SAMPLES}")
print("=" * 60)
print()

# # Disable torch compilation to avoid Triton errors
# os.environ["TORCH_COMPILE_DISABLE"] = "1"
# os.environ["TORCHDYNAMO_DISABLE"] = "1"

# =============================================================================
# GPU FAILOVER: OOM HANDLER
# =============================================================================
def _handle_oom_failover():
    """Re-exec this script on the next available GPU after an OOM crash."""
    current_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '?')
    print(f"\n{'='*60}")
    print(f"[GPU FAILOVER] OOM detected on GPU {current_gpu} — searching for another GPU...")
    print(f"{'='*60}")

    # Build skip list: current GPU + any previously skipped
    skip_set = set(_skip_set)  # from pre-import phase
    try:
        skip_set.add(int(current_gpu))
    except ValueError:
        pass

    candidates = _get_free_gpus(skip_indices=skip_set)
    candidates = [g for g in candidates if g['free_mb'] > 15000]

    if not candidates:
        print("[GPU FAILOVER] No other free GPUs available. Exiting.")
        sys.exit(1)

    next_gpu = candidates[0]
    print(f"[GPU FAILOVER] Switching to GPU {next_gpu['index']} ({next_gpu['free_mb']}MB free)")

    # Build new argv with updated --gpu and --skip-gpus
    new_argv = [sys.executable]
    skip_arg = False
    for a in sys.argv:
        if skip_arg:
            skip_arg = False
            continue
        if a == '--gpu':
            skip_arg = True
            continue
        if a.startswith('--skip-gpus='):
            continue
        new_argv.append(a)

    new_argv.extend(['--gpu', str(next_gpu['index'])])
    skip_str = ','.join(str(x) for x in sorted(skip_set))
    new_argv.append(f'--skip-gpus={skip_str}')

    print(f"[GPU FAILOVER] Relaunching: {' '.join(new_argv)}")
    os.execv(sys.executable, new_argv)

# Install global OOM exception hook (only when --gpu-failover is active)
_original_excepthook = sys.excepthook

def _oom_excepthook(exc_type, exc_value, exc_tb):
    """Global exception hook: intercept CUDA OOM and trigger GPU failover."""
    is_oom = False
    if exc_type is RuntimeError and 'out of memory' in str(exc_value).lower():
        is_oom = True
    if exc_type is torch.cuda.OutOfMemoryError:
        is_oom = True
    # Also catch vllm/allocation errors that signal memory exhaustion
    if 'alloc' in str(exc_value).lower() and 'memory' in str(exc_value).lower():
        is_oom = True

    if is_oom and _gpu_failover:
        # Print the original traceback first
        import traceback
        traceback.print_exception(exc_type, exc_value, exc_tb)
        _handle_oom_failover()
    else:
        _original_excepthook(exc_type, exc_value, exc_tb)

if _gpu_failover:
    sys.excepthook = _oom_excepthook
    print("[GPU] OOM exception hook installed — will auto-failover on OOM")

# Clear CUDA cache before starting
torch.cuda.empty_cache()

max_seq_length = 4096 # Increased to 4096 for longer reasoning traces
lora_rank = 16 # Reduced from 32 to save memory

# GRPO Dataset Filtering Configuration
MIN_CRITERIA_FOR_GRPO = 3  # Only train on prompts with MORE than this many criteria
                            # Change this value for ablation studies
                            # Set to 0 to disable filtering

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-14B-Base",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit - diff_pari
    fast_inference = True, 
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.8, # Further reduced from 0.8 to 0.6
)
print("first part!")
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank*2, # *2 speeds up training
    use_gradient_checkpointing = "unsloth", # Reduces memory usage
    random_state = 3407,
)
print("Everythin is loaded yayyyyy!")

###################################################################################################

#GRPO Chat template4
reasoning_start = "<start_working_out>" # Acts as <think>
reasoning_end   = "<end_working_out>"   # Acts as </think>
solution_start  = "<SOLUTION>"
solution_end    = "</SOLUTION>"

system_prompt = \
f"""You are a helpful medical AI assistant specializing in heart-related health questions.
When responding to medical queries, think through the problem carefully and provide your reasoning.
Place your reasoning and analysis between {reasoning_start} and {reasoning_end}.
Then, provide your final response and recommendations between {solution_start}{solution_end}"""
system_prompt

chat_template = \
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + eos_token }}"\
        "{% set loop_messages = messages[1:] %}"\
    "{% else %}"\
        "{{ '{system_prompt}' + eos_token }}"\
        "{% set loop_messages = messages %}"\
    "{% endif %}"\
    "{% for message in loop_messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ message['content'] }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ message['content'] + eos_token }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
    "{% endif %}"

# Replace with out specific template:
chat_template = chat_template\
    .replace("'{system_prompt}'",   f"'{system_prompt}'")\
    .replace("'{reasoning_start}'", f"'{reasoning_start}'")
tokenizer.chat_template = chat_template

tokenizer.apply_chat_template([
    {"role" : "user", "content" : "What should I do if someone is experiencing chest pain?"},
    {"role" : "assistant", "content" : f"{reasoning_start}Chest pain can be a sign of a serious heart condition. I need to assess the urgency and recommend immediate action.{reasoning_end}{solution_start}Call emergency services immediately if experiencing chest pain, especially with shortness of breath, sweating, or pain radiating to the arm or jaw.{solution_end}"},
    {"role" : "user", "content" : "How can I prevent heart disease?"},
], tokenize = False, add_generation_prompt = True)

# =============================================================================
# GROQ API KEY LOADING (shared by GRPO judge + evaluation)
# =============================================================================
groq_api_key = os.environ.get('GROQ_API_KEY') or os.getenv('GROQ_API_KEY')
if not groq_api_key:
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    if key.strip() == 'GROQ_API_KEY':
                        groq_api_key = value.strip().strip('"').strip("'")
                        os.environ['GROQ_API_KEY'] = groq_api_key
                        break

if groq_api_key:
    os.environ['GROQ_API_KEY'] = groq_api_key
    print(f"GROQ_API_KEY loaded (length: {len(groq_api_key)})")
else:
    print("WARNING: GROQ_API_KEY not found. Groq-based features will fail.")

from groq import Groq
groq_client = Groq() if groq_api_key else None

# =============================================================================
# SMOKE TEST - verify all models work before long training runs
# =============================================================================
if not args.skip_smoke_test:
    print("\n" + "=" * 60)
    print("SMOKE TEST - verifying all models before proceeding")
    print("=" * 60)

    smoke_prompt = "What is normal resting heart rate? Answer in one sentence."
    smoke_failures = []

    # --- Local model smoke test ---
    print("\n[1/2] Testing local model (vllm fast_generate)...")
    try:
        from vllm import SamplingParams as _SmokeSP
        _smoke_sp = _SmokeSP(temperature=0.5, max_tokens=64, seed=42)
        _smoke_msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": smoke_prompt},
        ]
        _smoke_text = tokenizer.apply_chat_template(_smoke_msgs, tokenize=False, add_generation_prompt=True)
        _smoke_out = model.fast_generate([_smoke_text], sampling_params=_smoke_sp)[0].outputs[0].text
        print(f"  Local model: OK  ({_smoke_out[:80].strip()}...)")
    except Exception as e:
        msg = f"Local model FAILED: {e}"
        print(f"  {msg}")
        smoke_failures.append(msg)

    # --- Groq models smoke test ---
    print("[2/2] Testing Groq API models...")
    if groq_client:
        for gm in GROQ_EVAL_MODELS:
            try:
                extra = dict(gm["extra_kwargs"])
                resp = groq_client.chat.completions.create(
                    model=gm["model"],
                    messages=[{"role": "user", "content": smoke_prompt}],
                    temperature=0.3, max_completion_tokens=64, **extra,
                )
                snippet = (resp.choices[0].message.content or "")[:80].strip()
                print(f"  {gm['display']:30s}: OK  ({snippet}...)")
            except Exception as e:
                msg = f"{gm['display']} FAILED: {e}"
                print(f"  {gm['display']:30s}: FAIL  ({e})")
                smoke_failures.append(msg)
    else:
        print("  Groq client unavailable - skipping Groq smoke tests")
        smoke_failures.append("Groq client not initialised (no API key)")

    if smoke_failures:
        print("\n" + "-" * 60)
        print(f"SMOKE TEST: {len(smoke_failures)} failure(s):")
        for f in smoke_failures:
            print(f"  - {f}")
        print("-" * 60)
        if RUN_TRAINING:
            print("Proceeding despite failures (training will still run).")
    else:
        print("\nSMOKE TEST: ALL PASSED")
    print("=" * 60 + "\n")
else:
    print("Smoke test skipped (--skip-smoke-test)\n")

# =============================================================================
# DATA LOADING
# =============================================================================
from datasets import load_dataset
import pandas as pd
import numpy as np
import json

# Create output directory for RaR-Medicine results
os.makedirs("rar-dataset-results", exist_ok=True)

# =============================================================================
# TRAINING DATA: RaR-Medicine (heart-related only)
# =============================================================================
print("=" * 60)
print("LOADING TRAINING DATA: RaR-Medicine Dataset")
print("=" * 60)

# Load the RaR-Medicine dataset (with heart classification + synthetic reasoning)
rar_data_path = "rar-dataset-results/rar_medicine_with_synthetic_reasoning_2026-02-09-06-26-40.jsonl"
rar_data = []
with open(rar_data_path, 'r') as f:
    for line in f:
        rar_data.append(json.loads(line))

# Convert to pandas DataFrame
rar_dataset = pd.DataFrame(rar_data)

# Filter to only heart-related samples from training split
rar_train = rar_dataset[
    (rar_dataset['split'] == 'train') & 
    (rar_dataset['heart_related'] == 'YES')
].copy()

print(f"RaR-Medicine total samples: {len(rar_dataset)}")
print(f"RaR-Medicine heart-related training samples: {len(rar_train)}")
print()

# =============================================================================
# TEST DATA: HealthBench (for evaluation only - NOT for training)
# =============================================================================
print("=" * 60)
print("LOADING TEST DATA: HealthBench Dataset (evaluation only)")
print("=" * 60)

# Load HealthBench data for evaluation
healthbench_path = "dataset/healthbench_with_synthetic_reasoning_2025-10-12-21-47-39.jsonl"
healthbench_data = []
with open(healthbench_path, 'r') as f:
    for line in f:
        healthbench_data.append(json.loads(line))

# Load the rubric dataset from JSONL
rubric_path = "dataset/2025-05-07-06-14-12_oss_eval.jsonl"
rubric_data = []
with open(rubric_path, 'r') as f:
    for line in f:
        rubric_data.append(json.loads(line))

# Convert to pandas DataFrames
healthbench_dataset = pd.DataFrame(healthbench_data)
rubric_dataset = pd.DataFrame(rubric_data)

# Merge datasets on prompt_id to get rubrics for each prompt
healthbench_dataset = healthbench_dataset.merge(
    rubric_dataset[['prompt_id', 'rubrics']], 
    on='prompt_id', 
    how='left'
)

# Filter to heart-related only for evaluation
test_data = healthbench_dataset[healthbench_dataset['heart_related'] == 'YES'].copy()

print(f"HealthBench total samples: {len(healthbench_dataset)}")
print(f"HealthBench heart-related test samples: {len(test_data)}")
print()

# =============================================================================
# DATASET STATISTICS
# =============================================================================
print("=" * 60)
print("DATASET STATISTICS")
print("=" * 60)
print(f"TRAINING (RaR-Medicine):")
print(f"  Heart-related samples: {len(rar_train)}")
print()
print(f"TESTING (HealthBench - for evaluation only):")
print(f"  Heart-related samples: {len(test_data)}")
print()

# Shuffle RaR-Medicine heart-related samples
heart_related_all = rar_train.sample(frac=1, random_state=3407).reset_index(drop=True)

# Split heart-related data 50/50 between SFT and GRPO
heart_split_idx = len(heart_related_all) // 2

sft_dataset = heart_related_all.iloc[:heart_split_idx].copy()
grpo_dataset = heart_related_all.iloc[heart_split_idx:].copy()

# Print SFT dataset statistics
print("-" * 60)
print("SFT DATASET (for pre-finetuning) - RaR-Medicine")
print("-" * 60)
print(f"Total SFT samples: {len(sft_dataset)}")
print(f"  - Heart-related: {len(sft_dataset)} (100.0%)")
print(f"  - Non-heart-related: 0 (0.0%)")
print()

# Print GRPO dataset statistics
print("-" * 60)
print("GRPO DATASET (for GRPO training) - RaR-Medicine")
print("-" * 60)
print(f"Total GRPO samples: {len(grpo_dataset)}")
print(f"  - Heart-related: {len(grpo_dataset)} (100.0%)")
print(f"  - Non-heart-related: 0 (0.0%)")
print("=" * 60)
print()

# Use SFT dataset for pre-finetuning
dataset = sft_dataset

#format dataset to have the GRPO style formatting
def format_dataset(x):
    # Extract expected answer from 'completion' field
    expected_answer = x["completion"]
    
    # Extract the user prompt from the 'prompt' field
    # prompt is a list of message dicts with role and content
    user_message = ""
    for msg in x["prompt"]:
        if msg["role"] == "user":
            user_message = msg["content"]
            break
    problem = user_message

    # Extract thoughts from 'synthetic_reasoning' field (may be empty for RaR-Medicine)
    thoughts = x.get("synthetic_reasoning", "")
    
    # If no synthetic reasoning, generate a basic reasoning template from the answer
    if not thoughts or thoughts.strip() == "":
        # Use a structured reasoning template based on the question and answer
        thoughts = f"Analyzing this medical question carefully. The key clinical considerations are important for providing accurate medical guidance. Based on the medical knowledge and clinical guidelines, the appropriate response should address the patient's concerns comprehensively."
    
    # Strip newlines on left and right
    thoughts = thoughts.strip()
    expected_answer = expected_answer.strip()
    
    # Add our custom formatting
    final_prompt = \
        reasoning_start + thoughts + reasoning_end + \
        solution_start + expected_answer + solution_end
    return [
        {"role" : "system",    "content" : system_prompt},
        {"role" : "user",      "content" : problem},
        {"role" : "assistant", "content" : final_prompt},
    ]

dataset["Messages"] = dataset.apply(format_dataset, axis = 1)
#test to see if it worked
tokenizer.apply_chat_template(dataset["Messages"][0], tokenize = False)
# truncate the pre fine-tuning dataset
dataset["N"] = dataset["Messages"].apply(lambda x: len(tokenizer.apply_chat_template(x)))

print("-" * 60)
print("SFT DATASET FILTERING")
print("-" * 60)
print(f"Dataset size before filtering: {len(dataset)}")
print(f"Max sequence length: {max_seq_length}")
# Use 90% of max_seq_length as threshold (leave room for generation)
filtering_threshold = int(max_seq_length * 0.9)
print(f"Filtering threshold (90% of max_seq_length): {filtering_threshold}")
print(f"Token count statistics:")
print(f"  Min: {dataset['N'].min()}")
print(f"  Max: {dataset['N'].max()}")
print(f"  Mean: {dataset['N'].mean():.1f}")
print(f"  Median: {dataset['N'].median():.1f}")

dataset = dataset.loc[dataset["N"] <= filtering_threshold].copy()

print(f"Dataset size after filtering: {len(dataset)}")
if len(dataset) == 0:
    print("WARNING: No samples remain after filtering! This will cause training to fail.")
    print("Consider increasing max_seq_length or adjusting the filtering threshold.")
elif len(dataset) < 10:
    print(f"WARNING: Only {len(dataset)} samples remain. Consider increasing max_seq_length.")
print("-" * 60)
print()
#tokenize the messages and convert it to a Hugging Face compatible dataset format
from datasets import Dataset

dataset["text"] = tokenizer.apply_chat_template(dataset["Messages"].values.tolist(), tokenize = False)
dataset = Dataset.from_pandas(dataset)

# Verify dataset format before training
print("-" * 60)
print("VERIFYING DATASET FORMAT")
print("-" * 60)
print("Sample formatted text (first 500 chars):")
print(dataset["text"][0][:500])
print("\n...")
print(dataset["text"][0][-200:])
print("-" * 60)
print()

# =============================================================================
# EVAL-ONLY: load SFT checkpoint so the model is in the right state
# =============================================================================
if args.only_eval and args.eval_local:
    sft_checkpoint_path = "rar-dataset-results/sft_saved_lora"
    if os.path.exists(os.path.join(sft_checkpoint_path, "adapter_model.safetensors")):
        print("Loading SFT LoRA for evaluation…")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, sft_checkpoint_path)
        print("SFT LoRA loaded.\n")
    else:
        print("WARNING: SFT checkpoint not found – evaluating raw base model.\n")

#pre fine-tune the model so it follows our custom GRPO formatting
if RUN_TRAINING and (args.only_sft or args.both_sft_grpo):
    # Check if SFT LoRA checkpoint exists
    sft_checkpoint_path = "rar-dataset-results/sft_saved_lora"
    if args.smoke_train:
        sft_checkpoint_path = "rar-dataset-results/sft_saved_lora_smoke"
    sft_checkpoint_exists = os.path.exists(os.path.join(sft_checkpoint_path, "adapter_model.safetensors"))
    
    # For smoke training, always force re-training to verify pipeline
    if args.smoke_train:
        sft_checkpoint_exists = False
        print("[SMOKE TRAIN] Forcing SFT training (ignoring existing checkpoints)")
    
    if sft_checkpoint_exists:
        print("=" * 60)
        print("SFT CHECKPOINT FOUND - LOADING EXISTING MODEL")
        print("=" * 60)
        print(f"Loading SFT LoRA from: {sft_checkpoint_path}")
        print("Skipping SFT training...")
        print("=" * 60)
        print()
        
        # Load the SFT LoRA weights into the model
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, sft_checkpoint_path)
        print("SFT LoRA loaded successfully!")
        print("=" * 60)
        print()
    else:
        print("=" * 60)
        print("NO SFT CHECKPOINT FOUND - TRAINING FROM SCRATCH")
        print("=" * 60)
        print()
        
        from trl import SFTTrainer, SFTConfig
        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = dataset,
            args = SFTConfig(
                dataset_text_field = "text",
                per_device_train_batch_size = 4,
                gradient_accumulation_steps = 4, # Use GA to mimic batch size!
                warmup_steps = 5,
                num_train_epochs = 2, 
                learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
                logging_steps = 1,
                max_steps = SFT_MAX_STEPS,
                optim = "adamw_8bit",
                weight_decay = 0.001,
                lr_scheduler_type = "linear",
                seed = 3407,
                report_to = "wandb", # Use this for WandB etc
                # dataloader_num_workers = 0, # Reduce workers to save memory
            ),
        )
        
        # Manually mask instruction tokens so the model only learns from responses.
        # The custom chat template produces:
        #   {system}<eos>{user_text}<start_working_out>{reasoning}<end_working_out><SOLUTION>{answer}</SOLUTION><eos>
        # We mask everything BEFORE <start_working_out> (the response_start token sequence).
        print("-" * 60)
        print("MASKING INSTRUCTION TOKENS (train on responses only)")
        print("-" * 60)
        
        # BPE tokenizers merge '<' and '>' with surrounding characters, so we search
        # for the invariant core: 'start_working_out' (without angle brackets)
        core_marker_ids = tokenizer("start_working_out", add_special_tokens=False)["input_ids"]
        core_marker_len = len(core_marker_ids)
        print(f"Response core marker tokens: {core_marker_ids} ('start_working_out')")
        
        # Also get the EOS token ID to skip the first occurrence in system prompt
        eos_id = tokenizer.eos_token_id
        print(f"EOS token ID: {eos_id}")
        
        # Debug: verify on first sample
        sample_ids = list(trainer.train_dataset[0]["input_ids"])
        print(f"First sample has {len(sample_ids)} tokens")
        
        # Find all occurrences of the core marker
        occurrences = []
        for i in range(len(sample_ids) - core_marker_len + 1):
            if list(sample_ids[i:i + core_marker_len]) == list(core_marker_ids):
                occurrences.append(i)
        print(f"Core marker found at positions: {occurrences}")
        
        # Find the EOS token (separates system prompt from user message)
        eos_pos = -1
        for i, tok in enumerate(sample_ids):
            if tok == eos_id:
                eos_pos = i
                break
        print(f"First EOS at position: {eos_pos}")
        
        # The response marker is the one AFTER the EOS (i.e., the actual delimiter, not the one in system prompt)
        response_marker_pos = None
        for pos in occurrences:
            if pos > eos_pos:
                response_marker_pos = pos
                break
        if response_marker_pos is not None:
            print(f"Response marker (after EOS) at position: {response_marker_pos}")
            # The actual marker includes the preceding '<' variant and following '>' variant
            # We mask everything up to and including the marker + its surrounding brackets
            # The token BEFORE the core is the '<' variant, so mask_end = response_marker_pos + core_marker_len + 1
            mask_end = response_marker_pos + core_marker_len + 1  # +1 for the '>' variant token
            if mask_end > len(sample_ids):
                mask_end = response_marker_pos + core_marker_len
            print(f"Will mask tokens 0 to {mask_end} (out of {len(sample_ids)})")
        else:
            print("WARNING: Response marker not found after EOS!")
        
        def mask_instruction_labels(example):
            input_ids = list(example["input_ids"])
            labels = list(input_ids)  # Copy as labels
            
            # Find first EOS (end of system prompt)
            first_eos = -1
            for i, tok in enumerate(input_ids):
                if tok == eos_id:
                    first_eos = i
                    break
            
            # Find core marker AFTER the first EOS (the actual response delimiter)
            found = False
            for i in range(max(first_eos + 1, 0), len(input_ids) - core_marker_len + 1):
                if input_ids[i:i + core_marker_len] == list(core_marker_ids):
                    # Mask everything up to and including the marker + closing bracket
                    mask_end = i + core_marker_len + 1  # +1 for '>' variant
                    if mask_end > len(input_ids):
                        mask_end = i + core_marker_len
                    for j in range(mask_end):
                        labels[j] = -100
                    found = True
                    break
            
            if not found:
                # Fallback: mask nothing (train on full sequence)
                pass
            
            example["labels"] = labels
            return example
        
        trainer.train_dataset = trainer.train_dataset.map(mask_instruction_labels)
        
        # Verify masking
        sample_labels = trainer.train_dataset[0]["labels"]
        num_masked = sum(1 for l in sample_labels if l == -100)
        num_total = len(sample_labels)
        num_trained = num_total - num_masked
        print(f"Sample: {num_masked} masked, {num_trained} trained, {num_total} total tokens")
        print(f"Masked ratio: {num_masked/num_total*100:.1f}% instruction, {num_trained/num_total*100:.1f}% response")
        
        if num_trained == 0:
            print("WARNING: All labels masked! Check template alignment.")
        else:
            print("Masking looks correct - training on response tokens only.")
        print("-" * 60)
        print()
        
        # Test model BEFORE SFT training
        print("=" * 60)
        print("TESTING MODEL BEFORE SFT TRAINING")
        print("=" * 60)
        test_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What are the warning signs of a heart attack?"},
        ]
        test_text = tokenizer.apply_chat_template(
            test_messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        
        print("Prompt: What are the warning signs of a heart attack?")
        print("\nModel output BEFORE SFT:")
        print("-" * 60)
        
        from transformers import TextStreamer
        FastLanguageModel.for_inference(model)  # Enable inference mode
        _ = model.generate(
            **tokenizer(test_text, return_tensors="pt").to("cuda"),
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            streamer=TextStreamer(tokenizer, skip_prompt=True),
        )
        print("-" * 60)
        print()
        
        # Re-enable training mode
        FastLanguageModel.for_training(model)
        
        # Show memory stats before training
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print("-" * 60)
        print("GPU MEMORY STATS (Before Training)")
        print("-" * 60)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")
        print("-" * 60)
        print()
        
        #train the model
        print("=" * 60)
        print("Training the model with SFT...")
        print("=" * 60)
        trainer_stats = trainer.train()
        print("=" * 60)
        print("SFT training completed!")
        print("=" * 60)
        
        # Show memory and time stats after training
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        print("-" * 60)
        print("GPU MEMORY STATS (After Training)")
        print("-" * 60)
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
        print("-" * 60)
        print()
        
        # Save SFT LoRA checkpoint for future use
        print("=" * 60)
        print("SAVING SFT CHECKPOINT")
        print("=" * 60)
        model.save_pretrained(sft_checkpoint_path)
        tokenizer.save_pretrained(sft_checkpoint_path)
        print(f"SFT LoRA saved to: {sft_checkpoint_path}")
        print("Next run will load this checkpoint instead of retraining!")
        print("=" * 60)
        print()
    
    #it did follow the formatting! Great! Let's remove some items before the GRPO step
    del dataset
    torch.cuda.empty_cache()
    import gc
    gc.collect()
elif RUN_TRAINING:
    print("=" * 60)
    print("SKIPPING SFT training (--only-grpo or --only-eval specified)")
    print("=" * 60)

#Data preparation
# Use the GRPO dataset (second half of the heart-related data)
# grpo_dataset was already created earlier during the data split

if (args.only_grpo or args.both_sft_grpo) and RUN_TRAINING:
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not set. Cannot proceed with GRPO training.")

    def format_grpo_dataset(x):
        # Extract the user prompt from the 'prompt' field
        user_message = ""
        for msg in x["prompt"]:
            if msg["role"] == "user":
                user_message = msg["content"]
                break
        
        # Format the prompt as a string using chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        
        # Convert to text format that GRPO expects
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        return {
            "prompt": prompt_text,  # Now a string instead of list of dicts
            "answer": x["completion"].strip(),
            "rubrics": x["rubrics"],  # Include rubrics for LLM-as-judge evaluation
        }

    # Convert grpo_dataset to the format needed for GRPO training
    from datasets import Dataset
    dataset = Dataset.from_pandas(grpo_dataset)
    dataset = dataset.map(format_grpo_dataset)
    
    # Filter dataset by number of criteria (rubrics) for more challenging training
    if MIN_CRITERIA_FOR_GRPO > 0:
        print("=" * 60)
        print("FILTERING GRPO DATASET BY CRITERIA COUNT")
        print("=" * 60)
        print(f"Minimum criteria required: > {MIN_CRITERIA_FOR_GRPO}")
        print(f"Dataset size before filtering: {len(dataset)}")
        
        # Count criteria for each sample
        def count_criteria(example):
            rubrics = example.get('rubrics', [])
            if not rubrics or not isinstance(rubrics, list):
                return {'num_criteria': 0}
            # Count ALL rubrics - no length restrictions
            return {'num_criteria': len(rubrics)}
        
        dataset = dataset.map(count_criteria)
        
        # Show distribution before filtering
        criteria_counts = {}
        for num in dataset['num_criteria']:
            criteria_counts[num] = criteria_counts.get(num, 0) + 1
        
        print("\nCriteria count distribution (before filtering):")
        for count in sorted(criteria_counts.keys()):
            percentage = (criteria_counts[count] / len(dataset)) * 100
            print(f"  {count} criteria: {criteria_counts[count]} samples ({percentage:.1f}%)")
        
        # Filter to keep only samples with MORE than MIN_CRITERIA_FOR_GRPO
        original_size = len(dataset)
        dataset = dataset.filter(lambda x: x['num_criteria'] > MIN_CRITERIA_FOR_GRPO)
        filtered_count = original_size - len(dataset)
        
        print(f"\nDataset size after filtering: {len(dataset)}")
        print(f"Filtered out: {filtered_count} samples ({(filtered_count/original_size)*100:.1f}%)")
        
        if len(dataset) == 0:
            print("\nWARNING: No samples remain after filtering!")
            print(f"Consider reducing MIN_CRITERIA_FOR_GRPO (currently {MIN_CRITERIA_FOR_GRPO})")
            raise ValueError("No samples available for GRPO training after filtering")
        
        # Show distribution after filtering
        criteria_counts_after = {}
        for num in dataset['num_criteria']:
            criteria_counts_after[num] = criteria_counts_after.get(num, 0) + 1
        
        print("\nCriteria count distribution (after filtering):")
        for count in sorted(criteria_counts_after.keys()):
            percentage = (criteria_counts_after[count] / len(dataset)) * 100
            print(f"  {count} criteria: {criteria_counts_after[count]} samples ({percentage:.1f}%)")
        
        print("=" * 60)
        print()
    else:
        print(f"Criteria filtering disabled (MIN_CRITERIA_FOR_GRPO = {MIN_CRITERIA_FOR_GRPO})")
    
    # Verify GRPO dataset format
    print("-" * 60)
    print("VERIFYING GRPO DATASET FORMAT")
    print("-" * 60)
    print("Sample GRPO prompt (first 300 chars):")
    print(dataset["prompt"][0][:300])
    print("\n...")
    print("\nSample answer (first 200 chars):")
    print(dataset["answer"][0][:200])
    if MIN_CRITERIA_FOR_GRPO > 0:
        print(f"\nSample has {dataset['num_criteria'][0]} criteria")
    print("-" * 60)
    print()

    #create a regex format to match the reasoning sections and answers
    import re

    # Add optional EOS token matching
    solution_end_regex = r"</SOLUTION>[\s]{0,}" + \
        "(?:" + re.escape(tokenizer.eos_token) + ")?"

    match_format = re.compile(
        rf"{reasoning_end}.*?"\
        rf"{solution_start}(.+?){solution_end_regex}"\
        rf"[\s]{{0,}}$",
        flags = re.MULTILINE | re.DOTALL
    )
    match_format.findall(
        "Let me think!<end_working_out>"\
        f"<SOLUTION>\n2\n</SOLUTION>",
    )

    # Pydantic model for criterion evaluation
    from pydantic import BaseModel as _PydanticBase

    class CriterionEvaluation(_PydanticBase):
        present: str  # "yes" or "no"
        justification: str

    # Real-time logger class for detailed debugging
    class RealTimeLogger:
        """Logger that writes to file in real-time with instant flushing"""
        
        def __init__(self, filename):
            self.filename = filename
            self.file = open(filename, 'w', encoding='utf-8')
            
        def log(self, message):
            """Write message to log and flush immediately"""
            self.file.write(message + '\n')
            self.file.flush()
        
        def close(self):
            """Close the log file"""
            self.file.close()

    # Create logger with timestamp and reward type
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = f"/home/unsloth/Projects/rar-dataset-results/grpo_judging_{REWARD_TYPE}_{timestamp}.log"
    judge_logger = RealTimeLogger(log_filename)
    
    judge_logger.log("=" * 100)
    judge_logger.log(f"GRPO TRAINING - DETAILED JUDGING LOG ({REWARD_TYPE.upper()} REWARD)".center(100))
    judge_logger.log("=" * 100)
    judge_logger.log(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    judge_logger.log(f"Reward Function: {REWARD_TYPE.upper()}")
    judge_logger.log(f"Judge Model: Groq API - openai/gpt-oss-120b")
    judge_logger.log("=" * 100)
    judge_logger.log("")

    # ── Parquet reward log collector ──────────────────────────
    reward_log_rows = []   # list of dicts; flushed to parquet at the end
    parquet_path = f"/home/unsloth/Projects/rar-dataset-results/grpo_rewards_{REWARD_TYPE}_{timestamp}.parquet"

    # Global counter for printing
    global PRINTED_TIMES
    PRINTED_TIMES = 0
    global PRINT_EVERY_STEPS
    PRINT_EVERY_STEPS = 5
    
    # ============================================================================
    # REWARD FUNCTIONS - Two different approaches
    # ============================================================================
    
    import math
    
    def compute_hybrid_reward(positive_scored, max_positive_score, negative_scored, 
                              max_negative_score, num_criteria, all_positives_met, no_negatives_met):
        """
        Hybrid Reward Function.
        Combines continuous gradient with perfection bonus.
        
        Formula: (normalized_score × 15) + (5 if perfect) - (neg_ratio × 4.5)
        
        Benefits:
        - Provides continuous gradient for partial improvements
        - Maintains strong incentive for perfection
        - Explicitly penalizes negative criteria hits
        """
        MAX_REWARD = 20.0
        PERFECT_BONUS = 5.0
        BASE_PORTION = MAX_REWARD - PERFECT_BONUS  # 15.0
        
        # Calculate normalized score
        norm_score = positive_scored / max(max_positive_score, 1)
        neg_ratio = negative_scored / max(max_negative_score, 1) if max_negative_score > 0 else 0.0
        
        # Base reward from normalized score
        base = norm_score * BASE_PORTION
        
        # Penalty for negative criteria hits
        if neg_ratio > 0:
            penalty = neg_ratio * BASE_PORTION * 0.3  # 30% of base portion as max penalty
            base = max(0, base - penalty)
        
        # Perfect bonus
        if all_positives_met and no_negatives_met:
            return base + PERFECT_BONUS
        
        return base
    
    def compute_complexity_reward(positive_scored, max_positive_score, negative_scored, 
                                  max_negative_score, num_criteria, all_positives_met, no_negatives_met):
        """
        Complexity-Aware Reward Function.
        
        Formula: r = 20 × normalized_score^1.2 × (1 + 0.2 × log(1 + num_criteria) / log(1 + 25))
        
        Features:
        - Exponential emphasis on high scores (^1.2)
        - Bonus for handling more criteria (complexity scaling)
        - Penalizes negative criteria through score reduction
        """
        MAX_CRITERIA = 25
        
        # Calculate normalized score with negative penalty
        norm_score = positive_scored / max(max_positive_score, 1)
        neg_ratio = negative_scored / max(max_negative_score, 1) if max_negative_score > 0 else 0.0
        
        # Apply negative penalty to normalized score
        adjusted_score = max(0, norm_score - neg_ratio * 0.5)
        
        # Apply the formula: r = 20 × score^1.2 × (1 + 0.2 × log(1+n)/log(1+25))
        score_component = adjusted_score ** 1.2
        complexity_component = 1 + 0.2 * (math.log1p(num_criteria) / math.log1p(MAX_CRITERIA))
        
        reward = 20.0 * score_component * complexity_component
        
        # Clip to valid range
        return min(max(reward, 0.0), 25.0)  # Allow slight overshoot for complex tasks
    
    def compute_reward(positive_scored, max_positive_score, negative_scored, max_negative_score,
                       num_criteria, all_positives_met, no_negatives_met, reward_type):
        """
        Main reward computation function that dispatches to the appropriate method.
        """
        if reward_type == 'hybrid':
            return compute_hybrid_reward(
                positive_scored, max_positive_score, negative_scored, max_negative_score,
                num_criteria, all_positives_met, no_negatives_met
            )
        elif reward_type == 'complexity':
            return compute_complexity_reward(
                positive_scored, max_positive_score, negative_scored, max_negative_score,
                num_criteria, all_positives_met, no_negatives_met
            )
        else:
            raise ValueError(f"Unknown reward type: {reward_type}")


    # LLM-as-judge reward function using Groq API with GRPO-optimized rewards
    def llm_judge_reward(prompts, completions, answer, **kwargs):
        """
        Evaluates completions against rubric criteria using Groq API (gpt-oss-120b)
        Returns binary rewards: 20 if perfect, 0 otherwise
        """
        global PRINTED_TIMES
        global PRINT_EVERY_STEPS
        
        scores = []
        
        # Get the rubrics from kwargs (passed from the dataset)
        rubrics_list = kwargs.get('rubrics', [])
        
        judge_logger.log("\n" + "█" * 100)
        judge_logger.log(f"NEW BATCH EVALUATION - {len(completions)} completions".center(100))
        judge_logger.log("█" * 100 + "\n")
        
        for idx, completion in enumerate(completions):
            # Completions are already strings (the generated text), not dicts
            response_text = completion if isinstance(completion, str) else str(completion)
            
            # Extract question from prompt (now a string, not list of dicts)
            # The prompt is formatted text from chat template, extract the user message
            prompt_text = prompts[idx] if idx < len(prompts) else prompts[0]
            # Simple extraction: get text after last <|im_start|>user or similar marker
            # For simplicity, just use the full prompt as question context
            question = prompt_text
            
            judge_logger.log("\n" + "┏" + "━" * 98 + "┓")
            judge_logger.log("┃" + f"COMPLETION {idx + 1}/{len(completions)}".center(98) + "┃")
            judge_logger.log("┗" + "━" * 98 + "┛\n")
            
            judge_logger.log("─── PROMPT ───")
            judge_logger.log(question[:500] + ("..." if len(question) > 500 else ""))
            judge_logger.log("")
            
            judge_logger.log("─── MODEL COMPLETION ───")
            judge_logger.log(response_text[:1000] + ("..." if len(response_text) > 1000 else ""))
            judge_logger.log("")
            
            # Get rubrics for this specific prompt (all completions share same prompt)
            rubrics = rubrics_list[idx] if isinstance(rubrics_list, list) and idx < len(rubrics_list) else rubrics_list
            
            if not rubrics or not isinstance(rubrics, list):
                # If no rubrics available, give minimum score
                judge_logger.log("WARNING: No rubrics available - assigning score 0.0")
                judge_logger.log("")
                scores.append(0.0)
                continue
            
            # Track positive and negative rubric contributions separately
            positive_scored = 0.0
            negative_scored_abs = 0.0
            max_positive_score = 0.0
            max_negative_score_abs = 0.0
            
            judge_logger.log("─── RUBRIC EVALUATION ───")
            judge_logger.log(f"Total rubrics: {len(rubrics)}")
            judge_logger.log(f"Min criteria threshold for training: > {MIN_CRITERIA_FOR_GRPO}\n")
            
            # Evaluate each rubric criterion
            for rubric_idx, rubric in enumerate(rubrics):
                criterion = rubric.get('criterion', '')
                points = rubric.get('points', 0)
                
                # NO LENGTH RESTRICTIONS - evaluate all criteria
                # Any errors will be caught by the try-catch block below
                
                # Track positive and negative maxima
                if points > 0:
                    max_positive_score += float(points)
                elif points < 0:
                    max_negative_score_abs += float(abs(points))
                
                judge_logger.log(f"\n╔═══ CRITERION {rubric_idx + 1}/{len(rubrics)} (Worth {points} points) ═══")
                judge_logger.log(f"║ Criterion length: {len(criterion)} chars")
                judge_logger.log(f"║ Criterion: {criterion[:150]}{'...' if len(criterion) > 150 else ''}")
                
                # Create evaluation prompt for Groq
                # NO TRUNCATION - send full context to the judge
                eval_prompt = f"""Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.



[question]: {question}



[response]: {response_text}



Your judgement must be in the format and criteria specified below:



extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.



[correct_answer]: {criterion}



reasoning: Explain BRIEFLY (1-2 sentences) why the extracted_final_answer is correct or incorrect based on [correct_answer]. Focus ONLY on whether the answers match. Be concise.



correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.



confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put 100 if there is no confidence score available.

Respond with ONLY valid JSON in this exact format:
{{
  "present": "yes" or "no",
  "justification": "brief explanation"
}}"""
                
                # Retry mechanism with progressive fallbacks
                max_retries = 3
                evaluation_success = False
                
                for attempt in range(max_retries):
                    try:
                        judge_logger.log(f"║ Calling Groq API (attempt {attempt + 1}/{max_retries})...")
                        judge_logger.log(f"║ Prompt length: ~{len(eval_prompt)} chars")
                        
                        # Increase max tokens for longer criteria
                        max_tokens = 2048 if len(criterion) > 1000 else 1024
                        
                        # USING GROQ API instead of Ollama for faster judging
                        response = groq_client.chat.completions.create(
                            model="openai/gpt-oss-120b",
                            messages=[
                                {
                                    "role": "user",
                                    "content": eval_prompt,
                                }
                            ],
                            temperature=0.1,
                            max_completion_tokens=max_tokens,
                            response_format={"type": "json_object"},  # Force JSON response
                        )
                        
                        # Get response content
                        judge_response = response.choices[0].message.content
                        judge_logger.log(f"║ Judge Response: {judge_response[:200]}{'...' if len(judge_response) > 200 else ''}")
                        
                        # Parse response as JSON and validate with Pydantic
                        evaluation = CriterionEvaluation.model_validate_json(judge_response)
                        
                        verdict = evaluation.present.lower()
                        justification = evaluation.justification
                        
                        judge_logger.log(f"║ ✓ Verdict: {verdict}")
                        judge_logger.log(f"║ ✓ Justification: {justification[:150]}{'...' if len(justification) > 150 else ''}")
                        
                        # Accumulate scored points
                        points_awarded = 0
                        if verdict == 'yes':
                            if points > 0:
                                positive_scored += float(points)
                                points_awarded = points
                            elif points < 0:
                                negative_scored_abs += float(abs(points))
                                points_awarded = points
                        
                        judge_logger.log(f"║ ✓ Points Awarded: {points_awarded} / {points}")
                        judge_logger.log(f"╚═══════════════════════════════════════")
                        
                        evaluation_success = True
                        # If successful, break the retry loop
                        break
                        
                    except Exception as e:
                        error_msg = str(e)
                        error_type = type(e).__name__
                        judge_logger.log(f"║ ✗ Error on attempt {attempt + 1}: [{error_type}] {error_msg[:300]}")
                        
                        # On retry, try with slightly different parameters
                        if attempt < max_retries - 1:
                            judge_logger.log(f"║ Retrying with adjusted parameters...")
                        
                        # Log detailed error on last attempt
                        if attempt == max_retries - 1:
                            judge_logger.log(f"║ ✗ FAILED after {max_retries} attempts")
                            judge_logger.log(f"║ Criterion length: {len(criterion)}")
                            judge_logger.log(f"║ Error type: {error_type}")
                            judge_logger.log(f"║ Full error: {error_msg[:500]}")
                            judge_logger.log(f"╚═══════════════════════════════════════")
                            
                            # IMPORTANT: When evaluation fails, count it as criterion NOT MET
                            # This prevents giving free passes for failed evaluations
                            # For positive criteria: don't award points (default behavior)
                            # For negative criteria: assume they're not violated (safer)
                            
                            if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
                                print(f"Warning: Failed to evaluate criterion after {max_retries} attempts")
                                print(f"  Criterion: {criterion[:100]}...")
                                print(f"  Error: [{error_type}] {error_msg[:200]}")
                        continue
                
                if not evaluation_success:
                    judge_logger.log(f"║ NOTE: Criterion not evaluated (treated as not met)")
                    # Don't award any points for failed evaluations
                    # This is conservative but prevents gaming the system
            
            judge_logger.log("")
            
            # Compute normalized score for logging
            s_norm = positive_scored / max_positive_score if max_positive_score > 0 else 0.0
            all_pos_met = positive_scored >= max_positive_score if max_positive_score > 0 else True
            no_neg_met = negative_scored_abs == 0
            
            # Apply selected reward function
            grpo_reward = compute_reward(
                positive_scored=positive_scored,
                max_positive_score=max_positive_score,
                negative_scored=negative_scored_abs,
                max_negative_score=max_negative_score_abs,
                num_criteria=len(rubrics),
                all_positives_met=all_pos_met,
                no_negatives_met=no_neg_met,
                reward_type=REWARD_TYPE
            )
            
            judge_logger.log("─── FINAL RESULTS ───")
            judge_logger.log(f"Reward Function: {REWARD_TYPE.upper()}")
            judge_logger.log(f"Positive Scored: {positive_scored:.2f} / {max_positive_score:.2f}")
            judge_logger.log(f"Negative Met (BAD): {negative_scored_abs:.2f} / {max_negative_score_abs:.2f}")
            judge_logger.log(f"Normalized Score (positives only): {s_norm:.4f} ({s_norm*100:.2f}%)")
            judge_logger.log(f"Number of Criteria: {len(rubrics)}")
            judge_logger.log(f"All Positives Met: {all_pos_met}")
            judge_logger.log(f"No Negatives Met: {no_neg_met}")
            judge_logger.log(f"GRPO REWARD ({REWARD_TYPE}): {grpo_reward:.2f}")
            judge_logger.log("=" * 100 + "\n")
            
            scores.append(grpo_reward)

            # ── collect row for parquet ──
            reward_log_rows.append({
                "timestamp":           datetime.now().isoformat(),
                "reward_type":         REWARD_TYPE,
                "completion_idx":      idx,
                "prompt_snippet":      question[:300],
                "response_snippet":    response_text[:500],
                "num_criteria":        len(rubrics),
                "positive_scored":     positive_scored,
                "max_positive_score":  max_positive_score,
                "negative_scored":     negative_scored_abs,
                "max_negative_score":  max_negative_score_abs,
                "normalized_score":    s_norm,
                "all_positives_met":   all_pos_met,
                "no_negatives_met":    no_neg_met,
                "grpo_reward":         grpo_reward,
            })
            
            # Print every few steps for monitoring
            if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
                print('*' * 80)
                print(f"[{REWARD_TYPE.upper()} REWARD]")
                print(f"Question:\n{question[:200]}...")
                print(f"\nResponse:\n{response_text[:500]}...")
                print(f"\nRubric Contributions:")
                print(f"  + Positive met: {positive_scored:.2f} / {max_positive_score:.2f}")
                print(f"  - Negative met (BAD): {negative_scored_abs:.2f} / {max_negative_score_abs:.2f}")
                print(f"  # Criteria: {len(rubrics)}")
                print(f"  ✓ All positives met: {all_pos_met}")
                print(f"  ✓ No negatives met: {no_neg_met}")
                print(f"  => {REWARD_TYPE.upper()} Reward: {grpo_reward:.2f}")
                print('*' * 80)
            PRINTED_TIMES += 1
        
        judge_logger.log(f"\nBatch complete. Rewards: {scores}")
        judge_logger.log("=" * 100 + "\n\n")
        
        return scores
    #get the 90% prompt length so we don't accidenatllly truncate them
    # Since prompts are now strings (already formatted with chat template), just tokenize directly
    tokenized = dataset.map(
        lambda x: {"tokens" : tokenizer(x["prompt"], add_special_tokens=False)["input_ids"]},
        batched = False,  # Process one at a time since prompts are strings
    )
    print(tokenizer.decode(tokenized[0]["tokens"]))
    tokenized = tokenized.map(lambda x: {"L" : len(x["tokens"])}, batched=False)

    import numpy as np
    maximum_length = int(np.quantile(tokenized["L"], 0.9))
    print("Max Length = ", maximum_length)

    # Filter only samples smaller than 90% max length
    dataset = dataset.select(np.where(np.array(tokenized["L"]) <= maximum_length)[0])
    del tokenized

    #train the model
    print("Training the model!")
    max_prompt_length = maximum_length + 1 # + 1 just in case!
    max_completion_length = 1024  # Set to 1024 as requested

    from vllm import SamplingParams
    vllm_sampling_params = SamplingParams(
        min_p = 0.1,
        top_p = 1.0,
        top_k = -1,
        seed = 3407,
        stop = [tokenizer.eos_token],
        include_stop_str_in_output = True,
    )

    from trl import GRPOConfig, GRPOTrainer
    
    # Configure wandb with reward type in run name
    import wandb
    wandb_run_name = f"grpo_{REWARD_TYPE}_reward_{timestamp}"
    
    training_args = GRPOConfig(
        vllm_sampling_params = vllm_sampling_params,
        temperature = 1.0,
        learning_rate = 5e-6,
        weight_decay = 0.01,
        warmup_ratio = 0.1,
        lr_scheduler_type = "linear",
        optim = "adamw_8bit",
        logging_steps = 1,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1, 
        num_generations = 6,
        max_prompt_length = max_prompt_length,
        max_completion_length = max_completion_length,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps = GRPO_MAX_STEPS, 
        save_steps = 10,
        report_to = "wandb",
        run_name = wandb_run_name,  # Include reward type in wandb run name
        output_dir = f"rar-dataset-results/outputs_{REWARD_TYPE}",  # Separate output dir per reward type
        dataloader_pin_memory = False, # Disable pin_memory for WSL
        dataloader_num_workers = 0, # Reduce workers to save memory

        # For optional training + evaluation
        # fp16_full_eval = True,
        # per_device_eval_batch_size = 4,
        # eval_accumulation_steps = 1,
        # eval_strategy = "steps",
        # eval_steps = 1,
    )
    # For optional training + evaluation
    # new_dataset = dataset.train_test_split(test_size = 0.01)

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            llm_judge_reward,  # Use LLM-as-judge for evaluation
        ],
        args = training_args,
        train_dataset = dataset,

        # For optional training + evaluation
        # train_dataset = new_dataset["train"],
        # eval_dataset = new_dataset["test"],
    )
    print("=" * 60)
    print(f"Starting GRPO training with {REWARD_TYPE.upper()} reward...")
    print("=" * 60)
    trainer.train()
    print("=" * 60)
    print(f"GRPO training with {REWARD_TYPE.upper()} reward completed!")
    print("=" * 60)
    
    # Close the judge logger
    judge_logger.log("\n" + "=" * 100)
    judge_logger.log("GRPO TRAINING COMPLETE".center(100))
    judge_logger.log("=" * 100)
    judge_logger.log(f"Ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    judge_logger.log(f"Log saved to: {log_filename}")
    judge_logger.close()
    print(f"\nDetailed judging log saved to: {log_filename}")

    # ── Flush reward log to parquet ──────────────────────────
    if reward_log_rows:
        import pyarrow as pa
        import pyarrow.parquet as pq
        reward_table = pa.Table.from_pylist(reward_log_rows)
        pq.write_table(reward_table, parquet_path)
        print(f"Reward log saved to parquet: {parquet_path}  ({len(reward_log_rows)} rows)")
else:
    print("=" * 60)
    print("SKIPPING GRPO training (--only-sft specified)")
    print("=" * 60)

# =============================================================================
# POST-TRAINING: Save LoRA & quick sanity check
# =============================================================================
if args.smoke_train:
    grpo_lora_path = f"rar-dataset-results/grpo_saved_lora_{REWARD_TYPE}_smoke"
else:
    grpo_lora_path = f"rar-dataset-results/grpo_saved_lora_{REWARD_TYPE}"

if RUN_TRAINING and (args.only_grpo or args.both_sft_grpo):
    print("Quick sanity: base model (no LoRA)…")
    from vllm import SamplingParams
    _sp = SamplingParams(temperature=1.0, top_k=50, max_tokens=1024)
    _out = model.fast_generate(
        ["What are the warning signs of a heart attack?"],
        sampling_params=_sp, lora_request=None,
    )[0].outputs[0].text
    print(_out[:300])

    print(f"\nSaving GRPO LoRA to {grpo_lora_path}…")
    model.save_lora(grpo_lora_path)

    from safetensors import safe_open
    zero_count = 0
    total_count = 0
    with safe_open(f"{grpo_lora_path}/adapter_model.safetensors", framework="pt") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            total_count += 1
            if (t == 0).sum().item() == t.numel():
                zero_count += 1
                print(f"  WARNING: Tensor {key} is all zeros (may be normal for lora_B in some layers)")
    if zero_count == total_count:
        raise RuntimeError("ALL LoRA tensors are zeros — training likely failed!")
    elif zero_count > 0:
        print(f"  {zero_count}/{total_count} tensors are all-zero (this is normal for some lora_B layers)")
    print("LoRA verified — at least some weights are non-zero.")

    _msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": "What are the warning signs of a heart attack?"},
    ]
    _text = tokenizer.apply_chat_template(_msgs, add_generation_prompt=True, tokenize=False)
    _sp2 = SamplingParams(temperature=1.0, top_k=50, max_tokens=2048)
    _out2 = model.fast_generate(
        _text, sampling_params=_sp2,
        lora_request=model.load_lora(grpo_lora_path),
    )[0].outputs[0].text
    print(f"GRPO output: {_out2[:300]}")
    print()

elif RUN_TRAINING and args.only_sft:
    print("=" * 60)
    print("TESTING MODEL AFTER SFT TRAINING")
    print("=" * 60)
    FastLanguageModel.for_inference(model)
    for prompt in ["What are the warning signs of a heart attack?",
                   "How can I prevent heart disease?",
                   "What should I do if I have chest pain?"]:
        _msgs = [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": prompt}]
        _text = tokenizer.apply_chat_template(_msgs, add_generation_prompt=True, tokenize=False)
        from transformers import TextStreamer
        print(f"\nPrompt: {prompt}")
        _ = model.generate(
            **tokenizer(_text, return_tensors="pt").to("cuda"),
            max_new_tokens=512, temperature=0.7, top_p=0.8, top_k=20,
            streamer=TextStreamer(tokenizer, skip_prompt=True),
        )
    print("=" * 60 + "\n")

###################################################################################################
# EVALUATION METRICS AND VISUALIZATION
###################################################################################################

if RUN_EVAL:
    print("\n" + "=" * 80)
    print("STARTING COMPREHENSIVE MODEL EVALUATION")
    print("=" * 80 + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
    from tqdm import tqdm
    from vllm import SamplingParams

    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 8)
    eval_ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # ── Prepare evaluation dataset ────────────────────────────
    eval_heart = test_data[test_data['heart_related'] == 'YES']
    eval_heart = eval_heart.sample(n=min(EVAL_SAMPLES, len(eval_heart)), random_state=42)
    eval_dataset = eval_heart.reset_index(drop=True)
    print(f"Evaluation dataset: {len(eval_dataset)} heart-related samples (requested: {EVAL_SAMPLES})\n")

    # ── Helper: generate from local model ─────────────────────
    def generate_local(prompt_text, use_lora=None):
        msgs = [{"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt_text}]
        text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        sp = SamplingParams(temperature=0.7, top_k=50, max_tokens=1024)
        return model.fast_generate([text], sampling_params=sp,
                                   lora_request=use_lora)[0].outputs[0].text

    # ── Helper: generate from Groq model ──────────────────────
    def generate_groq(prompt_text, model_id, extra_kwargs=None):
        extra = dict(extra_kwargs or {})
        resp = groq_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user",   "content": prompt_text}],
            temperature=0.7, max_completion_tokens=1024, **extra,
        )
        return resp.choices[0].message.content or ""

    # ── Helper: simplified rubric scorer ──────────────────────
    def evaluate_completion(completion_text, rubrics):
        if not rubrics or not isinstance(rubrics, list):
            return 0
        total_score, max_possible, min_possible = 0, 0, 0
        for rubric in rubrics[:5]:
            criterion = rubric.get('criterion', '')
            points = rubric.get('points', 0)
            if len(criterion) > 2000:
                continue
            if points > 0:
                max_possible += points
                if len(completion_text) > 100 and any(
                    w in completion_text.lower()
                    for w in ['should', 'recommend', 'important', 'seek', 'care']
                ):
                    total_score += points * 0.5
            elif points < 0:
                min_possible += points
                if any(w in completion_text.lower() for w in ['unsafe', 'harmful']):
                    total_score += points
        score_range = max_possible - min_possible
        normalized = (total_score - min_possible) / score_range if score_range > 0 else 0.5
        return 1 if normalized > 0.5 else 0

    # ── Build model registry for this run ─────────────────────
    model_registry = []   # list of dicts: {key, display, type}

    if args.eval_local:
        model_registry.append({"key": "base_model",  "display": "Qwen3-14B Base",
                                "type": "local", "lora": None})
        # Check GRPO checkpoint exists
        grpo_ckpt_ok = os.path.exists(os.path.join(grpo_lora_path, "adapter_model.safetensors"))
        if grpo_ckpt_ok:
            model_registry.append({"key": "grpo_model", "display": f"Qwen3-14B GRPO ({REWARD_TYPE.upper()})",
                                    "type": "local", "lora": grpo_lora_path})
        else:
            print(f"WARNING: GRPO checkpoint not found at {grpo_lora_path} - skipping GRPO eval")

    if args.eval_groq and groq_client:
        for gm in GROQ_EVAL_MODELS:
            model_registry.append({
                "key":     gm["id"],
                "display": gm["display"],
                "type":    "groq",
                "model":   gm["model"],
                "extra":   gm["extra_kwargs"],
            })

    print(f"Evaluating {len(model_registry)} models:")
    for m in model_registry:
        print(f"  [{m['type']:5s}] {m['display']}")
    print()

    # ── Run evaluation loop ───────────────────────────────────
    results = {m["key"]: [] for m in model_registry}
    results["ground_truth"] = []

    _grpo_lora_req = None
    if args.eval_local and os.path.exists(os.path.join(grpo_lora_path, "adapter_model.safetensors")):
        _grpo_lora_req = model.load_lora(grpo_lora_path)

    for idx, row in tqdm(eval_dataset.iterrows(), total=len(eval_dataset), desc="Evaluating"):
        user_prompt = ""
        for msg in row['prompt']:
            if msg['role'] == 'user':
                user_prompt = msg['content']
                break
        rubrics = row.get('rubrics', [])

        # ground truth
        if 'binary_labels' in row and isinstance(row.get('binary_labels'), list) and len(row['binary_labels']) > 0:
            gt = 1 if sum(row['binary_labels']) / len(row['binary_labels']) > 0.5 else 0
        else:
            gt = evaluate_completion(row['completion'], rubrics)
        results["ground_truth"].append(gt)

        for m in model_registry:
            try:
                if m["type"] == "local":
                    lora = _grpo_lora_req if m.get("lora") else None
                    comp = generate_local(user_prompt, use_lora=lora)
                else:
                    comp = generate_groq(user_prompt, m["model"], m.get("extra"))
                score = evaluate_completion(comp, rubrics)
            except Exception as e:
                print(f"  Error with {m['display']}: {e}")
                score = 0
            results[m["key"]].append(score)

    print("\nEvaluation complete!")

    # ── Compute metrics ───────────────────────────────────────
    print("\n" + "=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80 + "\n")

    metrics_data = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': []}
    y_true = results['ground_truth']

    for m in model_registry:
        y_pred = results[m["key"]]
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        metrics_data['Model'].append(m["display"])
        metrics_data['Accuracy'].append(acc)
        metrics_data['Precision'].append(prec)
        metrics_data['Recall'].append(rec)
        metrics_data['F1-Score'].append(f1)
        print(f"{m['display']}:")
        print(f"  Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}\n")

    metrics_df = pd.DataFrame(metrics_data)

    # ── Save metrics to CSV + parquet ─────────────────────────
    csv_path  = f'rar-dataset-results/evaluation_metrics_{REWARD_TYPE}_{eval_ts}.csv'
    pqt_path  = f'rar-dataset-results/evaluation_metrics_{REWARD_TYPE}_{eval_ts}.parquet'
    metrics_df.to_csv(csv_path, index=False)
    metrics_df.to_parquet(pqt_path, index=False)
    print(f"Saved metrics: {csv_path}")
    print(f"Saved metrics: {pqt_path}")

    # ── Save raw results to parquet ───────────────────────────
    raw_results_df = pd.DataFrame(results)
    raw_pqt = f'rar-dataset-results/evaluation_raw_{REWARD_TYPE}_{eval_ts}.parquet'
    raw_results_df.to_parquet(raw_pqt, index=False)
    print(f"Saved raw results: {raw_pqt}")

    # ==================================================================
    # VISUALIZATIONS
    # ==================================================================
    print("\nCreating visualizations…")
    n_models = len(model_registry)
    palette = sns.color_palette("husl", n_models)

    # 1. Grouped bar chart ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(14, 3 * n_models), 8))
    melted = metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    sns.barplot(data=melted, x='Metric', y='Score', hue='Model', ax=ax, palette=palette)
    ax.set_title(f'Model Comparison ({REWARD_TYPE.upper()} Reward)', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.legend(title='Model', fontsize=12, loc='upper left', bbox_to_anchor=(1, 1))
    for c in ax.containers:
        ax.bar_label(c, fmt='%.3f', padding=3, fontsize=12)
    plt.tight_layout()
    plt.savefig(f'rar-dataset-results/model_comparison_metrics_{REWARD_TYPE}_{eval_ts}.jpg', dpi=300, bbox_inches='tight')
    plt.savefig(f'rar-dataset-results/model_comparison_metrics_{REWARD_TYPE}_{eval_ts}.pdf', bbox_inches='tight')
    plt.close()

    # 2. Confusion matrices ────────────────────────────────────
    ncols = min(n_models, 4)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    for i, m in enumerate(model_registry):
        y_pred = results[m["key"]]
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Poor', 'Good'], yticklabels=['Poor', 'Good'])
        axes[i].set_title(m["display"], fontsize=10, fontweight='bold')
        axes[i].set_xlabel('Predicted'); axes[i].set_ylabel('Actual')
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(f'Confusion Matrices ({REWARD_TYPE.upper()})', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'rar-dataset-results/confusion_matrices_{REWARD_TYPE}_{eval_ts}.jpg', dpi=300, bbox_inches='tight')
    plt.savefig(f'rar-dataset-results/confusion_matrices_{REWARD_TYPE}_{eval_ts}.pdf', bbox_inches='tight')
    plt.close()

    # 3. Radar chart ───────────────────────────────────────────
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    for i, m in enumerate(model_registry):
        vals = [metrics_data['Accuracy'][i], metrics_data['Precision'][i],
                metrics_data['Recall'][i],   metrics_data['F1-Score'][i]]
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', linewidth=2, label=m["display"], color=palette[i])
        ax.fill(angles, vals, alpha=0.10, color=palette[i])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title(f'Performance Radar ({REWARD_TYPE.upper()})', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(f'rar-dataset-results/performance_radar_{REWARD_TYPE}_{eval_ts}.jpg', dpi=300, bbox_inches='tight')
    plt.savefig(f'rar-dataset-results/performance_radar_{REWARD_TYPE}_{eval_ts}.pdf', bbox_inches='tight')
    plt.close()

    # 4. Improvement over base (only if base_model present) ────
    if 'base_model' in [m['key'] for m in model_registry] and n_models > 1:
        base_idx = next(i for i, m in enumerate(model_registry) if m['key'] == 'base_model')
        base_vals = [metrics_data[c][base_idx] for c in categories]

        fig, ax = plt.subplots(figsize=(max(12, 2.5 * (n_models - 1)), 6))
        x = np.arange(len(categories))
        width = 0.8 / max(n_models - 1, 1)
        bar_idx = 0
        for i, m in enumerate(model_registry):
            if m['key'] == 'base_model':
                continue
            improvements = [(metrics_data[c][i] - base_vals[ci]) * 100
                            for ci, c in enumerate(categories)]
            bars = ax.bar(x + bar_idx * width - 0.4 + width / 2, improvements,
                          width, label=m["display"], color=palette[i])
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., h,
                        f'{h:+.1f}%', ha='center',
                        va='bottom' if h >= 0 else 'top', fontsize=7)
            bar_idx += 1
        ax.set_xticks(x); ax.set_xticklabels(categories)
        ax.set_ylabel('Improvement over Base (%)')
        ax.set_title(f'Improvement vs Qwen3-14B Base ({REWARD_TYPE.upper()})', fontsize=14, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'rar-dataset-results/improvement_comparison_{REWARD_TYPE}_{eval_ts}.jpg', dpi=300, bbox_inches='tight')
        plt.savefig(f'rar-dataset-results/improvement_comparison_{REWARD_TYPE}_{eval_ts}.pdf', bbox_inches='tight')
        plt.close()

    # 5. Horizontal bar chart – F1 scores ranked ───────────────
    fig, ax = plt.subplots(figsize=(10, max(4, 0.8 * n_models)))
    sorted_idx = sorted(range(n_models), key=lambda i: metrics_data['F1-Score'][i])
    names  = [metrics_data['Model'][i] for i in sorted_idx]
    f1vals = [metrics_data['F1-Score'][i] for i in sorted_idx]
    colors_sorted = [palette[i] for i in sorted_idx]
    bars = ax.barh(names, f1vals, color=colors_sorted)
    ax.set_xlabel('F1-Score')
    ax.set_title(f'F1-Score Ranking ({REWARD_TYPE.upper()})', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    for bar, v in zip(bars, f1vals):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{v:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'rar-dataset-results/f1_ranking_{REWARD_TYPE}_{eval_ts}.jpg', dpi=300, bbox_inches='tight')
    plt.savefig(f'rar-dataset-results/f1_ranking_{REWARD_TYPE}_{eval_ts}.pdf', bbox_inches='tight')
    plt.close()

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"EVALUATION COMPLETE - {REWARD_TYPE.upper()} REWARD")
    print("=" * 80)
    print(f"\nModels evaluated: {n_models}")
    print(f"  Local: {sum(1 for m in model_registry if m['type']=='local')}")
    print(f"  Groq:  {sum(1 for m in model_registry if m['type']=='groq')}")
    print(f"\nArtifacts saved to rar-dataset-results/  (timestamp: {eval_ts})")
    print(f"  - evaluation_metrics_*.csv / .parquet")
    print(f"  - evaluation_raw_*.parquet")
    print(f"  - model_comparison_metrics_*.jpg/.pdf")
    print(f"  - confusion_matrices_*.jpg/.pdf")
    print(f"  - performance_radar_*.jpg/.pdf")
    print(f"  - improvement_comparison_*.jpg/.pdf")
    print(f"  - f1_ranking_*.jpg/.pdf")
    print("=" * 80 + "\n")

else:
    print("\nEvaluation skipped (no --eval-local or --eval-groq flags).\n")

# =============================================================================
# MULTI-REWARD LOOP: run additional reward types if --reward-type all
# =============================================================================
# The first reward type has already been run above. If there are additional
# reward types (from --reward-type all), re-run GRPO training + save + eval
# for each. We reuse the already-defined functions, dataset, and model.
# SFT is NOT repeated (it's reward-type agnostic).

if len(REWARD_TYPES_LIST) > 1 and (args.only_grpo or args.both_sft_grpo) and RUN_TRAINING:
    for _extra_reward_idx, _extra_rt in enumerate(REWARD_TYPES_LIST[1:], start=2):
        REWARD_TYPE = _extra_rt
        print("\n" + "#" * 80)
        print(f"  REWARD TYPE {_extra_reward_idx}/{len(REWARD_TYPES_LIST)}: {REWARD_TYPE.upper()}")
        print("#" * 80 + "\n")

        # Re-create logger and parquet collector for this reward type
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_filename = f"/home/unsloth/Projects/rar-dataset-results/grpo_judging_{REWARD_TYPE}_{timestamp}.log"
        judge_logger = RealTimeLogger(log_filename)
        judge_logger.log("=" * 100)
        judge_logger.log(f"GRPO TRAINING - DETAILED JUDGING LOG ({REWARD_TYPE.upper()} REWARD)".center(100))
        judge_logger.log("=" * 100)
        judge_logger.log(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        judge_logger.log(f"Reward Function: {REWARD_TYPE.upper()}")
        judge_logger.log(f"Judge Model: Groq API - openai/gpt-oss-120b")
        judge_logger.log("=" * 100)
        judge_logger.log("")
        reward_log_rows = []
        parquet_path = f"/home/unsloth/Projects/rar-dataset-results/grpo_rewards_{REWARD_TYPE}_{timestamp}.parquet"
        PRINTED_TIMES = 0

        # Create new GRPO trainer with this reward type's config
        import wandb
        wandb_run_name = f"grpo_{REWARD_TYPE}_reward_{timestamp}"
        from trl import GRPOConfig, GRPOTrainer
        from vllm import SamplingParams

        vllm_sampling_params = SamplingParams(
            min_p=0.1, top_p=1.0, top_k=-1, seed=3407,
            stop=[tokenizer.eos_token], include_stop_str_in_output=True,
        )
        training_args = GRPOConfig(
            vllm_sampling_params=vllm_sampling_params,
            temperature=1.0, learning_rate=5e-6, weight_decay=0.01,
            warmup_ratio=0.1, lr_scheduler_type="linear", optim="adamw_8bit",
            logging_steps=1, per_device_train_batch_size=1,
            gradient_accumulation_steps=1, num_generations=6,
            max_prompt_length=max_prompt_length, max_completion_length=max_completion_length,
            max_steps=GRPO_MAX_STEPS, save_steps=10,
            report_to="wandb", run_name=wandb_run_name,
            output_dir=f"rar-dataset-results/outputs_{REWARD_TYPE}",
            dataloader_pin_memory=False, dataloader_num_workers=0,
        )
        trainer = GRPOTrainer(
            model=model, processing_class=tokenizer,
            reward_funcs=[llm_judge_reward],
            args=training_args, train_dataset=dataset,
        )
        print("=" * 60)
        print(f"Starting GRPO training with {REWARD_TYPE.upper()} reward...")
        print("=" * 60)
        trainer.train()
        print("=" * 60)
        print(f"GRPO training with {REWARD_TYPE.upper()} reward completed!")
        print("=" * 60)

        # Close logger and flush parquet
        judge_logger.log("\n" + "=" * 100)
        judge_logger.log("GRPO TRAINING COMPLETE".center(100))
        judge_logger.log("=" * 100)
        judge_logger.log(f"Ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        judge_logger.log(f"Log saved to: {log_filename}")
        judge_logger.close()
        print(f"\nDetailed judging log saved to: {log_filename}")
        if reward_log_rows:
            import pyarrow as pa
            import pyarrow.parquet as pq
            reward_table = pa.Table.from_pylist(reward_log_rows)
            pq.write_table(reward_table, parquet_path)
            print(f"Reward log saved to parquet: {parquet_path}  ({len(reward_log_rows)} rows)")

        # Save LoRA for this reward type
        if args.smoke_train:
            grpo_lora_path = f"rar-dataset-results/grpo_saved_lora_{REWARD_TYPE}_smoke"
        else:
            grpo_lora_path = f"rar-dataset-results/grpo_saved_lora_{REWARD_TYPE}"

        print(f"\nSaving GRPO LoRA to {grpo_lora_path}…")
        model.save_lora(grpo_lora_path)
        from safetensors import safe_open
        zero_count = 0
        total_count = 0
        with safe_open(f"{grpo_lora_path}/adapter_model.safetensors", framework="pt") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                total_count += 1
                if (t == 0).sum().item() == t.numel():
                    zero_count += 1
                    print(f"  WARNING: Tensor {key} is all zeros (may be normal for lora_B in some layers)")
        if zero_count == total_count:
            print(f"WARNING: ALL LoRA tensors are zeros for {REWARD_TYPE} — training may have failed!")
        elif zero_count > 0:
            print(f"  {zero_count}/{total_count} tensors are all-zero (this is normal for some lora_B layers)")
        print("LoRA verified — at least some weights are non-zero.")

        # Run evaluation for this reward type
        if RUN_EVAL:
            print("\n" + "=" * 80)
            print(f"STARTING EVALUATION FOR {REWARD_TYPE.upper()} REWARD")
            print("=" * 80 + "\n")

            eval_ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            eval_heart = test_data[test_data['heart_related'] == 'YES']
            eval_heart = eval_heart.sample(n=min(EVAL_SAMPLES, len(eval_heart)), random_state=42)
            eval_dataset = eval_heart.reset_index(drop=True)
            print(f"Evaluation dataset: {len(eval_dataset)} heart-related samples (requested: {EVAL_SAMPLES})\n")

            model_registry = []
            if args.eval_local:
                model_registry.append({"key": "base_model", "display": "Qwen3-14B Base",
                                       "type": "local", "lora": None})
                grpo_ckpt_ok = os.path.exists(os.path.join(grpo_lora_path, "adapter_model.safetensors"))
                if grpo_ckpt_ok:
                    model_registry.append({"key": "grpo_model",
                                           "display": f"Qwen3-14B GRPO ({REWARD_TYPE.upper()})",
                                           "type": "local", "lora": grpo_lora_path})
            if args.eval_groq and groq_client:
                for gm in GROQ_EVAL_MODELS:
                    model_registry.append({"key": gm["id"], "display": gm["display"],
                                           "type": "groq", "model": gm["model"],
                                           "extra": gm["extra_kwargs"]})

            print(f"Evaluating {len(model_registry)} models:")
            for m in model_registry:
                print(f"  [{m['type']:5s}] {m['display']}")
            print()

            all_results = {m["key"]: [] for m in model_registry}
            from tqdm import tqdm
            for sample_idx in tqdm(range(len(eval_dataset)), desc="Evaluating"):
                row = eval_dataset.iloc[sample_idx]
                prompt_text = row['question'] if 'question' in row else row.get('prompt', '')
                rubrics = row.get('rubrics', [])
                for m in model_registry:
                    try:
                        if m["type"] == "local":
                            response = generate_local(prompt_text, use_lora=model.load_lora(m["lora"]) if m["lora"] else None)
                        else:
                            response = generate_groq(prompt_text, m["model"], m.get("extra"))
                        score = evaluate_completion(response, rubrics)
                    except Exception as e:
                        print(f"  Error ({m['display']}): {e}")
                        score = 0
                        response = ""
                    all_results[m["key"]].append({"score": score, "response": response})

            print("\nEvaluation complete!")

            # Compute and display metrics
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
            import numpy as np

            metrics_data = {"Model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1-Score": []}
            print("\n" + "=" * 80)
            print("EVALUATION METRICS")
            print("=" * 80)
            for m in model_registry:
                scores = [r["score"] for r in all_results[m["key"]]]
                ones = [1] * len(scores)
                acc = accuracy_score(ones, scores) if scores else 0
                prec = precision_score(ones, scores, zero_division=0) if scores else 0
                rec = recall_score(ones, scores, zero_division=0) if scores else 0
                f1 = f1_score(ones, scores, zero_division=0) if scores else 0
                metrics_data["Model"].append(m["display"])
                metrics_data["Accuracy"].append(acc)
                metrics_data["Precision"].append(prec)
                metrics_data["Recall"].append(rec)
                metrics_data["F1-Score"].append(f1)
                print(f"\n{m['display']}:")
                print(f"  Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")

            # Save metrics
            import pandas as pd
            metrics_df = pd.DataFrame(metrics_data)
            csv_path = f'rar-dataset-results/evaluation_metrics_{REWARD_TYPE}_{eval_ts}.csv'
            pqt_path = f'rar-dataset-results/evaluation_metrics_{REWARD_TYPE}_{eval_ts}.parquet'
            metrics_df.to_csv(csv_path, index=False)
            metrics_df.to_parquet(pqt_path, index=False)
            print(f"\nSaved metrics: {csv_path}")
            print(f"Saved metrics: {pqt_path}")

            # Save raw results
            raw_rows = []
            for m in model_registry:
                for i, r in enumerate(all_results[m["key"]]):
                    raw_rows.append({"model": m["display"], "sample_idx": i,
                                     "score": r["score"], "response_snippet": r["response"][:500]})
            raw_df = pd.DataFrame(raw_rows)
            raw_pqt = f'rar-dataset-results/evaluation_raw_{REWARD_TYPE}_{eval_ts}.parquet'
            raw_df.to_parquet(raw_pqt, index=False)
            print(f"Saved raw results: {raw_pqt}")

            print(f"\nCreating visualizations for {REWARD_TYPE.upper()}…")
            n_models = len(model_registry)
            palette = sns.color_palette("husl", n_models)
            sns.set_style("whitegrid")

            # 1. Grouped bar chart
            fig, ax = plt.subplots(figsize=(14, 8))
            categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            x = np.arange(len(categories))
            width = 0.8 / n_models
            for i, m_name in enumerate(metrics_data['Model']):
                vals = [metrics_data[c][i] for c in categories]
                ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=m_name, color=palette[i])
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.set_ylabel('Score')
            ax.set_title(f'Model Comparison ({REWARD_TYPE.upper()} Reward)', fontsize=16, fontweight='bold')
            ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))
            ax.set_ylim(0, 1)
            plt.tight_layout()
            plt.savefig(f'rar-dataset-results/model_comparison_metrics_{REWARD_TYPE}_{eval_ts}.jpg', dpi=300, bbox_inches='tight')
            plt.savefig(f'rar-dataset-results/model_comparison_metrics_{REWARD_TYPE}_{eval_ts}.pdf', bbox_inches='tight')
            plt.close()

            # 2. F1-Score ranking
            fig, ax = plt.subplots(figsize=(10, max(4, 0.8 * n_models)))
            sorted_idx = sorted(range(n_models), key=lambda i: metrics_data['F1-Score'][i])
            names = [metrics_data['Model'][i] for i in sorted_idx]
            f1vals = [metrics_data['F1-Score'][i] for i in sorted_idx]
            colors_sorted = [palette[i] for i in sorted_idx]
            bars = ax.barh(names, f1vals, color=colors_sorted)
            ax.set_xlabel('F1-Score')
            ax.set_title(f'F1-Score Ranking ({REWARD_TYPE.upper()})', fontsize=14, fontweight='bold')
            ax.set_xlim(0, 1)
            for bar, v in zip(bars, f1vals):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                        f'{v:.3f}', va='center', fontsize=9)
            plt.tight_layout()
            plt.savefig(f'rar-dataset-results/f1_ranking_{REWARD_TYPE}_{eval_ts}.jpg', dpi=300, bbox_inches='tight')
            plt.savefig(f'rar-dataset-results/f1_ranking_{REWARD_TYPE}_{eval_ts}.pdf', bbox_inches='tight')
            plt.close()

            print(f"\n{'='*80}")
            print(f"EVALUATION COMPLETE - {REWARD_TYPE.upper()} REWARD")
            print(f"{'='*80}\n")

        # Clear memory between reward types
        del trainer
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print(f"\n[MULTI-REWARD] {REWARD_TYPE.upper()} done. Memory cleared.\n")

elif len(REWARD_TYPES_LIST) > 1 and args.only_eval:
    # Multi-reward eval-only mode
    for _extra_reward_idx, _extra_rt in enumerate(REWARD_TYPES_LIST[1:], start=2):
        REWARD_TYPE = _extra_rt
        print("\n" + "#" * 80)
        print(f"  EVAL REWARD TYPE {_extra_reward_idx}/{len(REWARD_TYPES_LIST)}: {REWARD_TYPE.upper()}")
        print("#" * 80 + "\n")

        if args.smoke_train:
            grpo_lora_path = f"rar-dataset-results/grpo_saved_lora_{REWARD_TYPE}_smoke"
        else:
            grpo_lora_path = f"rar-dataset-results/grpo_saved_lora_{REWARD_TYPE}"

        if not os.path.exists(os.path.join(grpo_lora_path, "adapter_model.safetensors")):
            print(f"WARNING: GRPO checkpoint not found at {grpo_lora_path} - skipping {REWARD_TYPE}")
            continue

        eval_ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        eval_heart = test_data[test_data['heart_related'] == 'YES']
        eval_heart = eval_heart.sample(n=min(EVAL_SAMPLES, len(eval_heart)), random_state=42)
        eval_dataset = eval_heart.reset_index(drop=True)
        print(f"Evaluation dataset: {len(eval_dataset)} samples\n")

        model_registry = []
        if args.eval_local:
            model_registry.append({"key": "base_model", "display": "Qwen3-14B Base",
                                   "type": "local", "lora": None})
            model_registry.append({"key": "grpo_model",
                                   "display": f"Qwen3-14B GRPO ({REWARD_TYPE.upper()})",
                                   "type": "local", "lora": grpo_lora_path})
        if args.eval_groq and groq_client:
            for gm in GROQ_EVAL_MODELS:
                model_registry.append({"key": gm["id"], "display": gm["display"],
                                       "type": "groq", "model": gm["model"],
                                       "extra": gm["extra_kwargs"]})

        all_results = {m["key"]: [] for m in model_registry}
        from tqdm import tqdm
        for sample_idx in tqdm(range(len(eval_dataset)), desc=f"Evaluating ({REWARD_TYPE})"):
            row = eval_dataset.iloc[sample_idx]
            prompt_text = row['question'] if 'question' in row else row.get('prompt', '')
            rubrics = row.get('rubrics', [])
            for m in model_registry:
                try:
                    if m["type"] == "local":
                        response = generate_local(prompt_text, use_lora=model.load_lora(m["lora"]) if m["lora"] else None)
                    else:
                        response = generate_groq(prompt_text, m["model"], m.get("extra"))
                    score = evaluate_completion(response, rubrics)
                except Exception as e:
                    score = 0
                    response = ""
                all_results[m["key"]].append({"score": score, "response": response})

        # Save metrics (simplified for eval-only)
        import pandas as pd
        metrics_data = {"Model": [], "Accuracy": [], "F1-Score": []}
        for m in model_registry:
            scores = [r["score"] for r in all_results[m["key"]]]
            ones = [1] * len(scores)
            acc = accuracy_score(ones, scores) if scores else 0
            f1 = f1_score(ones, scores, zero_division=0) if scores else 0
            metrics_data["Model"].append(m["display"])
            metrics_data["Accuracy"].append(acc)
            metrics_data["F1-Score"].append(f1)
            print(f"{m['display']}: Acc={acc:.4f} F1={f1:.4f}")
        pd.DataFrame(metrics_data).to_csv(
            f'rar-dataset-results/evaluation_metrics_{REWARD_TYPE}_{eval_ts}.csv', index=False)
        pd.DataFrame(metrics_data).to_parquet(
            f'rar-dataset-results/evaluation_metrics_{REWARD_TYPE}_{eval_ts}.parquet', index=False)
        print(f"\nEVALUATION COMPLETE - {REWARD_TYPE.upper()}\n")

if len(REWARD_TYPES_LIST) > 1:
    print("\n" + "=" * 80)
    print("ALL REWARD TYPES COMPLETED")
    print("=" * 80)
    print(f"Reward types processed: {', '.join(r.upper() for r in REWARD_TYPES_LIST)}")
    print("=" * 80 + "\n")