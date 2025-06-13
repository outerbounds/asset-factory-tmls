#!/usr/bin/env python3

from metaflow import Flow
import json

def analyze_dataset():
    try:
        prepare_flow = Flow('PrepareVLMData')
        latest_run = prepare_flow.latest_successful_run
        dataset = latest_run.data.training_dataset
        
        print(f"📊 Dataset Analysis - Run: {latest_run.pathspec}")
        print(f"Created: {dataset['metadata']['created_at']}")
        print(f"Training: {len(dataset['train'])} examples")
        print(f"Validation: {len(dataset['validation'])} examples")
        
        # Analyze tags
        all_tags = set()
        for example in dataset['train'] + dataset['validation']:
            all_tags.update(example['ground_truth_tags'])
        
        print(f"Unique tags: {len(all_tags)}")
        print(f"Top tags: {sorted(list(all_tags))[:10]}")
        
        # Check data format
        sample = dataset['train'][0]
        print(f"\nSample format check:")
        print(f"  Photo ID: {sample['photo_id']}")
        print(f"  Ground truth tags: {len(sample['ground_truth_tags'])}")
        print(f"  Messages format: {len(sample['messages'])} messages")
        
        # Verify prompt consistency
        user_prompt = sample['messages'][0]['content'][0]['text']
        with open('prompt.txt', 'r') as f:
            production_prompt = f.read().strip()
        
        if user_prompt == production_prompt:
            print("✅ Prompt consistency verified")
        else:
            print("⚠️  Prompt mismatch detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset analysis failed: {e}")
        return False

if __name__ == "__main__":
    analyze_dataset() 