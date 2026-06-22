# LiveCodeBench
Official repository for the paper "LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code"

<p align="center">
    <a href="https://livecodebench.github.io/">🏠 Home Page</a> •
    <a href="https://huggingface.co/livecodebench/">💻 Data </a> •
    <a href="https://livecodebench.github.io/leaderboard.html">🏆 Leaderboard</a> •
    <a href="https://huggingface.co/spaces/livecodebench/code_generation_samples">🔍 Explorer</a> 
</p>

## Introduction
LiveCodeBench provides holistic and contamination-free evaluation of coding capabilities of LLMs.  Particularly, LiveCodeBench continuously collects new problems over time from contests across three competition platforms -- LeetCode, AtCoder, and CodeForces. Next, LiveCodeBench also focuses on a broader range of code-related capabilities, such as self-repair, code execution, and test output prediction, beyond just code generation. Currently, LiveCodeBench hosts four hundred high-quality coding problems that were published between May 2023 and March 2024.


## Installation
You can clone the repository using the following command:

```bash
git clone https://github.com/LiveCodeBench/LiveCodeBench.git
cd LiveCodeBench
```

We recommend using [uv](https://github.com/astral-sh/uv)
for managing dependencies, which can be installed a [number of ways](https://github.com/astral-sh/uv?tab=readme-ov-file#installation).

Verify that `uv` is installed on your system by running:

```bash
uv --version
```

Once `uv` has been installed, use it to create a virtual environment for
LiveCodeBench and install its dependencies with the following commands:

```bash
uv venv --python 3.11
source .venv/bin/activate

uv pip install -e .
```

## Data
We provide a benchmark for different code capability scenarios
- [Code Generation](https://huggingface.co/datasets/livecodebench/code_generation_lite)
- [Code Execution](https://huggingface.co/datasets/livecodebench/execution)
- [Test Output Prediction](https://huggingface.co/datasets/livecodebench/test_generation)

## Inference and Evaluation

### TL;DR — local run with vLLM

Generate with a local model and grade the results against the test cases in one command:

```bash
python -m lcb_runner.runner.main \
    --model Qwen/Qwen2.5-7B-Instruct-1M \
    --scenario codegeneration \
    --max_model_len 32768 \
    --evaluate
```

Then read the overall `pass@1` from `output/Qwen2.5-7B-Instruct-1M/codegeneration_10_0.2_eval.json`. The rest of this section explains each stage, every relevant flag, and the other scenarios. See [How a run works](#how-a-run-works-overview) for the high-level flow and [Local test-case evaluation](#local-test-case-evaluation) for an important security note about executing model-generated code.

### Dataset Versions
Since LiveCodeBench is a continuously updated benchmark, we provide different versions of the dataset. Particularly, we provide the following versions of the dataset:
- `release_v1`: The initial release of the dataset with problems released between May 2023 and Mar 2024 containing 400 problems.
- `release_v2`: The updated release of the dataset with problems released between May 2023 and May 2024 containing 511 problems.
- `release_v3`: The updated release of the dataset with problems released between May 2023 and Jul 2024 containing 612 problems.
- `release_v4`: The updated release of the dataset with problems released between May 2023 and Sep 2024 containing 713 problems.
- `release_v5`: The updated release of the dataset with problems released between May 2023 and Jan 2025 containing 880 problems.
- `release_v6`: The updated release of the dataset with problems released between May 2023 and Apr 2025 containing 1055 problems.

You can use the `--release_version` flag to specify the dataset version you wish to use. Particularly, you can use the following command to run the evaluation on the `release_v2` dataset. Release version defaults to `release_latest`. Additionally, we have introduced fine-grained release versions such as `v1`, `v2`, `v1_v3`, `v4_v5` for specific versions of the dataset.

```bash
python -m lcb_runner.runner.main --model {model_name} --scenario codegeneration --evaluate --release_version release_v2
```

### How a run works (overview)

Every run goes through a single entrypoint, `lcb_runner.runner.main`, which performs up to three stages in order:

1. **Build prompts** — the chosen dataset is loaded and each problem is turned into a model-specific prompt (chat template applied for local models, message list for API models). This is driven by `--scenario` and the model's `LMStyle`.
2. **Generate** — prompts are sent to the model. For local models this uses **vLLM** (`VLLMRunner`); for API models it uses the matching API client. Outputs are written to `output/<model_repr>/<scenario>_<n>_<temperature>[ _cot].json`.
3. **Evaluate** (only with `--evaluate`) — the code/answers are extracted from the generations and graded. For code scenarios this means **executing the generated code against the problem's test cases locally** and computing `pass@k`. Results are written next to the generations as `..._eval.json` (aggregate metrics) and `..._eval_all.json` (per-problem details).

`<model_repr>` is the short display name from [`lcb_runner/lm_styles.py`](./lcb_runner/lm_styles.py) (e.g. `Qwen2.5-7B-Instruct-1M`), not the full repo id. The same naming scheme is reused across all scenarios.

### Code Generation (local inference with vLLM)

For local open-weight models we use [`vllm`](https://github.com/vllm-project/vllm) for generation. Provide the `model_name` exactly as registered in [`lcb_runner/lm_styles.py`](./lcb_runner/lm_styles.py) (which is usually the Hugging Face repo id). The minimal command to generate **and** evaluate is:

```bash
python -m lcb_runner.runner.main \
    --model Qwen/Qwen2.5-7B-Instruct-1M \
    --scenario codegeneration \
    --max_model_len 32768 \
    --evaluate
```

This will (1) load the model in vLLM, (2) sample `n=10` completions per problem at `temperature=0.2`, (3) extract the Python code from each completion, and (4) run that code against every test case to compute `pass@1`/`pass@5`. The first run downloads the model weights from Hugging Face into your `HF_HOME` cache; subsequent runs reuse them.

#### GPU and context-length configuration

- **`--tensor_parallel_size`** — number of GPUs to shard the model across. Defaults to **all visible GPUs** (`torch.cuda.device_count()`). Restrict visibility with `CUDA_VISIBLE_DEVICES=0,1 ...` or set the flag explicitly. A model must fit in the aggregate memory of the chosen GPUs.
- **`--max_model_len`** — caps the served context window. **Set this for long-context models.** Several registered models report context windows of 256k–1M tokens (the `*-1M`, `*-512k`, `*-262k` variants); by default vLLM allocates a KV cache for the *full* window and will OOM immediately. A value like `32768` is plenty for LiveCodeBench prompts.
- **`--gpu_memory_utilization`** — fraction of GPU memory vLLM may use for weights + KV cache (default `0.9`). Lower it (e.g. `0.8`) if you hit OOM during cache allocation or share the GPU with other processes.
- **`--dtype`** — compute dtype passed to vLLM (default `bfloat16`). A few checkpoints are published in `float32`; the default `bfloat16` downcast is fine and expected for inference.

```bash
# Example: a 14B model across 2 GPUs with a capped context window
CUDA_VISIBLE_DEVICES=0,1 python -m lcb_runner.runner.main \
    --model Qwen/Qwen2.5-14B-Instruct-1M \
    --scenario codegeneration \
    --tensor_parallel_size 2 \
    --max_model_len 32768 \
    --gpu_memory_utilization 0.9 \
    --evaluate
```

#### Models that need extra flags

`zai-org/glm-4-9b-chat-1m` and `internlm/internlm2_5-7b-chat-1m` ship custom model code, so pass **`--trust_remote_code`** for vLLM to load them:

```bash
python -m lcb_runner.runner.main \
    --model zai-org/glm-4-9b-chat-1m \
    --scenario codegeneration \
    --trust_remote_code \
    --max_model_len 32768 \
    --evaluate
```

Some `meta-llama/*` and all `OctoLong/*` repositories are **gated** on Hugging Face. Accept the license on the model page and authenticate before running, e.g. `export HF_TOKEN=hf_...` (or `huggingface-cli login`). Alternatively, point at a directory of already-downloaded weights with `--local_model_path /path/to/model` (the `--model` value is still used to select the prompt style and the output folder name).

#### Sampling and run-control flags

- **`--n`** — completions sampled per problem (default `10`). Use `--n 1` for a quick smoke test.
- **`--temperature` / `--top_p`** — sampling controls (defaults `0.2` / `0.95`). The output path encodes `n` and `temperature`, so different settings won't overwrite each other.
- **`--max_tokens`** — max tokens generated per completion (default `2000`).
- **`--debug`** — runs on only the first 15 problems; useful to validate an end-to-end run cheaply.
- **`--use_cache`** — caches raw generations under `cache/<model_repr>/...` keyed by prompt, so re-runs skip already-generated prompts.
- **`--continue_existing`** / **`--continue_existing_with_eval`** — reuse a prior run's generations (and, with `_with_eval`, its evaluations) and only fill in what's missing. Handy when extending an old run to a newer `--release_version`.
- **`--release_version`** — selects the dataset version (see [Dataset Versions](#dataset-versions)); defaults to `release_latest`.

#### Local instruction-tuned models supported out of the box

Many instruction-tuned models are registered with the `GenericInstruct` style, which renders each prompt using the model's **own Hugging Face chat template** (shipped with the tokenizer). This is what makes it trivial to add new local models without hand-writing a per-family prompt — see [Adding Support for New Models](#adding-support-for-new-models). The following are registered and verified:

`OctoLong/OctoLong-{0.6B,1.7B,4B,8B,14B}-Instruct`, `meta-llama/Llama-3.2-{1B,3B}-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`, `01-ai/Yi-Coder-{1.5B,9B}-Chat`, `ibm-granite/granite-3.1-{2b,8b}-instruct`, `Qwen/Qwen3-4B-Instruct-2507`, `Qwen/Qwen2.5-{7B,14B}-Instruct-1M`, `arcee-ai/AFM-4.5B`, `aws-prototyping/MegaBeam-Mistral-7B-512k`, `internlm/internlm2_5-7b-chat-1m`, `nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct`, `princeton-nlp/Llama-3-8B-ProLong-512k-Instruct`, `gradientai/Llama-3-8B-Instruct-262k`, `mistralai/Ministral-8B-Instruct-2410`, `zai-org/glm-4-9b-chat-1m`.

> For closed API models, no GPU is needed; instead set the provider's API key (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) and use `--multiprocess N` to parallelize requests within your rate limits. The vLLM-only flags above are ignored for API models.

### Local test-case evaluation

Adding `--evaluate` to any code scenario grades the generations by **executing the extracted code against the problem's test cases on your machine**. The grading logic lives in [`lcb_runner/evaluation/`](./lcb_runner/evaluation/) (a hardened fork of the [`apps`](https://github.com/hendrycks/apps/blob/main/eval/testing_util.py) checker). Two problem shapes are handled automatically:

- **stdin/stdout problems** — the program is run and its stdout is compared against the expected output.
- **functional problems** (LeetCode-style with starter code) — the target function is called with each test input and its return value is compared (the function name comes from the problem metadata).

Evaluation is independent of how the generations were produced, so you can evaluate generations from any model/source the same way (see [Custom Evaluation](#custom-evaluation)).

> ⚠️ **Security:** evaluation executes model-generated code in-process using `exec()`, guarded by a per-test timeout (`signal.alarm`) and a `reliability_guard` that disables obviously destructive builtins (e.g. `os.remove`, `shutil` ops). This is **not** a strong sandbox. Run evaluations on untrusted generations inside a container/VM you are willing to throw away.

Evaluation-specific flags:

- **`--num_process_evaluate`** — number of worker processes used to grade problems in parallel (default `12`). Lower it if you see flaky timeouts or run out of memory/CPU.
- **`--timeout`** — per-test-case wall-clock timeout in seconds (default `6`). Increase it for heavier problems if you observe spurious time-limit failures.

```bash
python -m lcb_runner.runner.main \
    --model Qwen/Qwen2.5-7B-Instruct-1M \
    --scenario codegeneration \
    --max_model_len 32768 \
    --evaluate \
    --num_process_evaluate 8 \
    --timeout 10
```

Note that time limits can cause slight (`< 0.5` point) variation in `pass@1`/`pass@5`. If you observe a *significant* variation, lower `--num_process_evaluate` or raise `--timeout`.

#### Output files

After a `codegeneration` run with `--evaluate`, look in `output/<model_repr>/`:

| File | Contents |
| --- | --- |
| `codegeneration_<n>_<temp>.json` | raw generations + extracted code per problem |
| `codegeneration_<n>_<temp>_eval.json` | aggregate metrics, including overall `pass@1` |
| `codegeneration_<n>_<temp>_eval_all.json` | per-problem details: `code_list`, `graded_list` (pass/fail per sample), `pass@1`, and failure `metadata` |

#### Scores over time windows / difficulty

To recompute `pass@k` over a date range, platform, or difficulty split from an existing `_eval_all.json`, use [`compute_scores.py`](./lcb_runner/evaluation/compute_scores.py). It can locate the file from `--model/--scenario/--n/--temperature`, or you can pass it directly with `--eval_all_file`. It prints overall and per-difficulty `pass@k` for `k ∈ {1,5,10,25,...}`:

```bash
# By model identity (resolves the eval_all file automatically):
python -m lcb_runner.evaluation.compute_scores --model Qwen/Qwen2.5-7B-Instruct-1M --scenario codegeneration --n 10

# Or point straight at the file, e.g. to counter contamination by filtering early problems:
python -m lcb_runner.evaluation.compute_scores --eval_all_file {saved_eval_all_file} --start_date 2023-09-01
```

`--start_date`/`--end_date` use `YYYY-MM-DD` and filter by problem release date; `--platform` filters by `leetcode`/`codeforces`/`atcoder`.

**NOTE: We have pruned a large number of test cases from the original benchmark and created `code_generation_lite` which is set as the default benchmark offering similar performance estimation much faster. If you wish to use the original benchmark, please use the `--not_fast` flag. We are in the process of updating the leaderboard scores with this updated setting.** 

**NOTE: V2 Update: to run the update LiveCodeBench please use `--release_version release_v2`. In addition, if you have existing results from `release_v1` you can add `--continue_existing` or better `--continue_existing_with_eval` flags to reuse the old completions or evaluations respectively.**


### Self Repair
Self-repair feeds a model its own failing code plus the error/feedback and asks it to fix it, so **it requires a prior `codegeneration` run with `--evaluate`** for the same model: it reads `output/<model_repr>/codegeneration_<codegen_n>_<temperature>_eval_all.json`. Provide `--codegen_n` to point at that file (the `n` used during code generation) and `--temperature` to match it. Only `--n 1` is supported for the repair step.

```bash
python -m lcb_runner.runner.main --model {model_name} --scenario selfrepair --codegen_n {num_codes_codegen} --n 1 --evaluate # only n=1 supported
```

In case you have results on a smaller subset or version of the benchmark, you can use `--continue_existing` and `--continue_existing_with_eval` flags to reuse the old computations. Particularly, you can run the following command to continue from existing generated solutions.

```bash
python -m lcb_runner.runner.main --model {model_name} --scenario selfrepair --evaluate --continue_existing
```

Note that this will only reuse the generated samples and rerun evaluations. To reuse the old evaluations, you can add the `--continue_existing_with_eval` flag.

All local-inference flags from [Code Generation](#code-generation-local-inference-with-vllm) (`--max_model_len`, `--tensor_parallel_size`, `--trust_remote_code`, etc.) apply to every scenario below as well.

### Test Output Prediction
The model is given a function and a test input and must predict the asserted output; grading checks the predicted assertion. Run with:

```bash
python -m lcb_runner.runner.main --model {model_name} --scenario testoutputprediction --evaluate
```

### Code Execution
The model is given a function and an input and must predict the execution result. Run with:

```bash
python -m lcb_runner.runner.main --model {model_name} --scenario codeexecution --evaluate
```

Additionally, we support a chain-of-thought (CoT) setting that lets the model reason step-by-step before answering (its output path is suffixed with `_cot`):

```bash
python -m lcb_runner.runner.main --model {model_name} --scenario codeexecution --cot_code_execution --evaluate
```

## Custom Evaluation
If you generated solutions elsewhere (any model, any framework) you can run **just the local test-case evaluation** on them with [`lcb_runner/runner/custom_evaluator.py`](./lcb_runner/runner/custom_evaluator.py) — no GPU or model loading required. Pass the same `--scenario`/`--release_version` you want to grade against, plus the evaluation flags (`--num_process_evaluate`, `--timeout`) described in [Local test-case evaluation](#local-test-case-evaluation).

```bash
python -m lcb_runner.runner.custom_evaluator --custom_output_file {path_to_custom_outputs} --scenario codegeneration
```

The file must be a JSON **list with one entry per benchmark problem**. The recommended form is a list of dicts keyed by problem id (entries are matched/sorted by id, so order does not matter):

```json
[
    {"question_id": "id1", "code_list": ["code1", "code2"]},
    {"question_id": "id2", "code_list": ["code1", "code2"]}
]
```

The required keys depend on the scenario:
- `codegeneration` / `selfrepair`: `question_id` + `code_list` (already-extracted code, one string per sample).
- `testoutputprediction`: `question_id` + `test_id` + `pred_list`.
- `codeexecution`: `id` + `pred_list`.

Alternatively, the file may be a plain `list[list[str]]` (the per-problem answer lists) **in benchmark order** — i.e. sorted by `question_id` (and `test_id` for test output prediction, or numeric `id` for code execution). Results are written next to your input file as `<name>_<scenario>_output.json` and `..._eval(_all).json`, unless you set `--custom_output_save_name`.


## Adding Support for New Models

To add support for new models, we have implemented an extensible framework to add new models and customize prompts appropriately. 

**Quick path (recommended for most instruction-tuned local models):** If the model ships a Hugging Face `chat_template` (most modern instruction-tuned models do), simply add it to the `LanguageModelList` array in [./lcb_runner/lm_styles.py](./lcb_runner/lm_styles.py) with `LMStyle.GenericInstruct`. No prompt code needs to change — the runner applies the model's own chat template at inference time (see [./lcb_runner/prompts/utils.py](./lcb_runner/prompts/utils.py)).

```python
# ./lcb_runner/lm_styles.py
LanguageModel(
    "my-org/My-Instruct-Model",
    "My-Instruct-Model",
    LMStyle.GenericInstruct,
    datetime(2025, 1, 1),
    link="https://huggingface.co/my-org/My-Instruct-Model",
),
```

**Custom path (for a bespoke prompt format):** 

Step 1: Add a new model to the [./lcb_runner/lm_styles.py](./lcb_runner/lm_styles.py) file. Particularly, extend the `LMStyle` class to add a new model family and extend the model to the `LanguageModelList` array.

Step 2: Since we use instruction tuned models, we allow configuring the instruction for each model. Modify the [./lcb_runner/prompts/generation.py](./lcb_runner/prompts/generation.py) file to add a new prompt for the model in the `format_prompt_generation` function. 
For example, the prompt for `DeepSeekCodeInstruct` family of models looks as follows

```python
# ./lcb_runner/prompts/generation.py
if LanguageModelStyle == LMStyle.DeepSeekCodeInstruct:
    prompt = f"{PromptConstants.SYSTEM_MESSAGE_DEEPSEEK}\n\n"
    prompt += f"{get_deepseekcode_question_template_answer(question)}"
    return prompt
```

## Submit Models to Leaderboard
We are currently only accepting submissions for only the code generation scenario. To submit models you can create a pull request on our [submissions](https://github.com/LiveCodeBench/submissions). Particularly, you can copy your model generations folder from `output` to the `submissions` folder and create a pull request. We will review the submission and add the model to the leaderboard accordingly. 

## ERRATA
We maintain a list of known issues and updates in the [ERRATA.md](./ERRATA.md) file. Particularly, we document issues regarding erroneous tests and problems not amenable to autograding. We are constantly using this feedback to improve our problem selection heuristics as we update LiveCodeBench.

## Results
LiveCodeBench can be used to evaluate performance of LLMs on different time-windows (using problem release date to filter the models). 
Thus we can detect and prevent potential contamination in the evaluation process and evaluate LLMs on _new_ problems.

<div style="text-align: center;">
    <img src="./assets/images/contamination1.png" alt="Code Generation Live Evaluation" class="teaser-image"
    width="40%" />
    <img src="./assets/images/contamination2.png" alt="Test Output Prediction Live Evaluation" class="teaser-image"
    width="40%" />
</div>

Next, we evaluate models on different code capabilities and find that relative performances of models do change over tasks (left). 
Thus, it highlights the need for holistic evaluation of LLMs for code.

<div style="text-align: center;">
    <img src="./assets/images/tasks_radar.png" alt="Holistic Tasks Evaluation" class="teaser-image"
    width="36.1%" />
    <img src="./assets/images/lcb_vs_he.png" alt="Comparing LCB vs HumanEval" class="teaser-image"
    width="46%" />
</div>

We also find evidence of possible overfitting on HumanEval (right). 
Particularly, models that perform well on HumanEval do not necessarily perform well on LiveCodeBench. 
In the scatterplot above, we find the models get clustered into two groups, shaded in red and green. 
The red group contains models that perform well on HumanEval but poorly on LiveCodeBench, while the green group contains models that perform well on both.

For more details, please refer to our website at [livecodebench.github.io](https://livecodebench.github.io).

## Citation

```bibtex
@article{jain2024livecodebench,
  author    = {Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, Ion Stoica},
  title     = {LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code},
  year      = {2024},
  journal   = {arXiv preprint},
}
```
