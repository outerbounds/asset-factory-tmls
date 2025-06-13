# Photo Tagger – Vision-Language Model Factory

A reproducible pipeline for large-scale photo tagging.  It lets you:

1. Ingest data from the web (Unsplash via `UpdatePhotos`).
2. Benchmark state-of-the-art VLMs against your domain (`PromptModels`).
3. Optionally fine-tune a local model with LoRA (`FineTuneVLM`).
4. Compare vendor and custom models in a unified dashboard (`EvaluationFlow`).
5. Iterate on data, model and prompts with CI-friendly tests.

Fine-tuning is **optional**.  You can run either path:

```
(quick baseline)  UpdatePhotos → PromptModels → EvaluationFlow
(full loop)       UpdatePhotos → PrepareVLMData → FineTuneVLM → PromptModels → EvaluationFlow
```

---

## Quick Start (local CLI)

```bash
# 1. Dependencies (CUDA 11.8+, Python ≥3.10)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # ~3 min

# 2. Secrets (export or use Outerbounds Integrations feature - see UI tab)
export client_id=UNSPLASH_KEY     # data ingest
export openai_api=OPENAI_KEY      # vendor models
export togetherai_api=TOGETHER_KEY

# 3. Smoke test – baseline evaluation only
python updatephotos.py --environment=fast-bakery run  # fetch a batch
python promptmodels.py --environment=fast-bakery run  # evaluate GPT-4, Llama-3-70B, Qwen-3B
python evalsflow.py --environment=fast-bakery run     # build dashboard card (stdout shows URL)
```

Optional fine-tune branch (single-GPU, LoRA-4bit):

```bash
python prepare_vlm_data.py  --environment=fast-bakery run --max-samples 2000  # curate dataset
python finetune_vlm.py --environment=fast-bakery run --epochs 3 --batch-size 4  # ~25 min
python promptcustommodels.py --environment=fast-bakery run  # evaluate fine-tuned model
python evalsflow.py  --environment=fast-bakery run                      # refresh dashboard
```

All of the above can be executed on Outerbounds with `--environment=fast-bakery argo-workflows create` – see inline decorators for resource hints.

---

## Flows & Scripts

| File | Role |
|------|------|
| `updatephotos.py` | Fetch raw photos + tags from Unsplash |
| `prepare_vlm_data.py` | Incremental, validated dataset builder |
| `finetune_vlm.py` | LoRA fine-tuning of Qwen2.5-VL-3B (4-bit NF4) |
| `promptmodels.py` | Evaluate vendor VLM APIs (OpenAI, Together) |
| `promptcustommodels.py` | Evaluate local fine-tuned models |
| `evalsflow.py` | Aggregate metrics → phototagger dashboard |
| `prompter.py` | Unified inference wrapper (API or local) |
| `diagnose_data.py` | Deep data-quality diagnostics |
| `check_data.py` | Lightweight dataset sanity check |
| `test_pipeline.py` | End-to-end CI smoke test |

---

## Configuration knobs

* Dataset curation: `--min-tags`, `--max-samples`, `--incremental`, `--force-rebuild`
* Fine-tuning: `--epochs`, `--batch-size`, `--learning-rate`, `--lora-rank`
* GPU resources: adjust `@kubernetes(gpu=…, memory=…)` decorators
* Prompt template: edit `prompt.txt` and re-run evaluation

---

## Data & Model Format

Training examples (instruction-following JSON):

```json
{
  "messages": [
    {"role": "user", "content": [
      {"type": "text", "text": "Tag this image with 9 relevant tags in JSON"},
      {"type": "image", "image": "https://…"}
    ]},
    {"role": "assistant", "content": [
      {"type": "text", "text": "{\"tags\":[\"mountain\",…]}"}
    ]}
  ],
  "ground_truth_tags": ["mountain", "landscape", …]
}
```

Fine-tuned models are stored as a LoRA adapter (≈45 MB) + quantised base model.  `prompter.py` attaches the adapter automatically when provider=`finetuned`.

---

## Testing & CI

```bash
./run_tests.sh  # runs data check, fine-tune stub, evaluation stub
```

Integrate with GitHub Actions: set secrets, then add a job that calls `./run_tests.sh` in a CPU container (it skips GPU-only paths by design).

---

## License

See `LICENSE` (Apache-2.0).

## Evaluation dashboard & metrics

The `EvaluationFlow` builds the HTML card that shows vendor-vs-finetuned results.  Each run outputs a JSON payload (`self.evals`) consumed by Metaflow Cards and the web front-end.

Metric definitions (see `evalsflow.py`):

* **Model** – provider + model name (row).
* **Avg Latency (ms) | Offline** – mean inference wall-clock latency across all photos that were evaluated _offline_ (i.e. aggregated over multiple batches).
* **Avg Recall | Offline** – mean tag-recall percentage across all batches.  Recall is computed per photo as:
  ```text
  recall% = (# ground-truth tags that the model predicted) / (total # ground-truth tags) * 100
  ```
  so 22 means on average the model reproduced 22 % of the gold tags.
* **Batch columns** – one column per evaluation batch (e.g. `zvkxk (06-03)`).  The label shows the short batch id and the collection date.  The value shows `avg-recall (n)` where _n_ is the # photos in that batch.
* **Heat-map colouring** – cells are shaded from red (high) → yellow → light based on deciles of the recall distribution (`COLORS` palette in `evalsflow.py`).  Latency cells are left uncoloured.

Rows are sorted by **Avg Recall | Offline** so the best model bubbles to the top.

Behind the scenes `EvaluationFlow` ingests two data sources:
1. JSON log events emitted via `EvalsLogger` inside the `PromptModels` / `PromptCustomModels` flows.
2. The latest `PromptCustomModels` run artifacts (to guarantee finetuned results are included even if log scraping misses them).

The resulting `headers` + `rows` dict is rendered automatically by the Phototagger card template when you open the flow run in the Outerbounds UI.