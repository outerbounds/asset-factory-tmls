#!/bin/bash

echo "🚀 VLM Factory End-to-End Testing"
echo "=================================="

# Set up environment
export PYTHONPATH=".:$PYTHONPATH"

# Create lightweight stub packages to satisfy pylint without heavy installs
mkdir -p stubs/{torch,transformers,peft,datasets}
for pkg in torch transformers peft datasets; do
  echo "# Stub $pkg module for CI" > "stubs/$pkg/__init__.py"
done
export PYTHONPATH="$(pwd)/stubs:$PYTHONPATH"

echo "📋 Step 1: Check Current Data Pipeline (Incremental Mode)"
echo "--------------------------------------------------------"
python prepare_vlm_data.py run --max-samples 100 --min-tags 3 --data-ns global
python check_data.py

echo ""
echo "🔧 Step 2: Run Fine-tuning (Quick Test)"
echo "--------------------------------------"
python finetune_vlm.py --environment=fast-bakery run --epochs 1 --batch-size 2 --learning-rate 1e-4 --skip-training True

echo ""
echo "🧪 Step 3: Test Integration"
echo "---------------------------"
python test_pipeline.py

echo ""
echo "📊 Step 4: Manual Verification Commands"
echo "---------------------------------------"
echo "To verify results manually:"
echo ""
echo "# Check fine-tuned model exists:"
echo "python -c \"from metaflow import Flow; print(Flow('FineTuneVLM').latest_successful_run.data.registered_model_key)\""
echo ""
echo "# Test prompter directly:"
echo "python -c \"from prompter import Prompter; print('Prompter loaded')\""
echo ""
echo "# Run evaluation with fine-tuned model:"
echo "python promptmodels.py run"
echo ""
echo "# Check evaluation results:"
echo "python -c \"from metaflow import Flow; pm = Flow('PromptModels').latest_successful_run; print(f'Models tested: {len(pm.data.models_eval[\\\"photos\\\"][0][\\\"models\\\"])}')\""

echo ""
echo "✅ Testing sequence complete!"
echo "Check outputs above for any errors." 