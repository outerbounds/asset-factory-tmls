#!/usr/bin/env python3

from metaflow import FlowSpec, step, Parameter, card, retry, current, project, pypi, kubernetes, model, environment
from metaflow.profilers import gpu_profile
import os
from datetime import datetime
import tempfile
import json
import shutil

@project(name="photo_tagger")
class FineTuneVLM(FlowSpec):
    
    epochs = Parameter("epochs", default=3)
    batch_size = Parameter("batch-size", default=4)
    learning_rate = Parameter("learning-rate", default=1e-4)
    lora_rank = Parameter("lora-rank", default=16)
    skip_training = Parameter("skip-training", default=False, help="Skip heavy training and register dummy model (for CI)")
    
    @step
    def start(self):
        from metaflow import Flow
        
        prepare_flow = Flow('PrepareVLMData')
        dataset_run = prepare_flow.latest_successful_run
        dataset = dataset_run.data.training_dataset
        
        self.training_data = dataset["train"]
        self.validation_data = dataset["validation"]
        self.dataset_metadata = dataset["metadata"]
        
        print(f"Dataset: {len(self.training_data)} train, {len(self.validation_data)} validation")
        
        self.next(self.train_model)
    
    @environment(vars={"TOKENIZERS_PARALLELISM": "false"})
    @model
    @kubernetes(gpu=1, memory=32_000)
    @gpu_profile(interval=1)
    @pypi(packages={
        "transformers": "4.49.0",
        "torch": "2.4.1", 
        "peft": "0.13.2",
        "datasets": "3.1.0",
        "accelerate": "1.1.0",
        "pillow": "10.0.0",
        "torchvision": "0.19.1"
    })
    @step
    def train_model(self):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, TrainingArguments, Trainer, DataCollatorForLanguageModeling
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
        
        if self.skip_training:
            # Quick stub for CI: create a dummy model folder with metadata only
            print("⚡ Skip-training flag enabled – registering dummy model artifact.")
            dummy_dir = tempfile.mkdtemp()
            metadata_path = os.path.join(dummy_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump({"stub": True}, f)

            # Save fake model via Metaflow model artifact
            self.model_id = current.model.save(dummy_dir, metadata={"stub": True})

            from assets import Asset
            asset = Asset(project='photo_tagger', branch='main')
            model_key = f"finetuned_vlm_stub_{current.run_id}"
            asset.register_model_asset(
                name=model_key,
                description="Stub model for CI pipeline",
                kind="pytorch_lora",
                blobs=[dummy_dir]
            )

            self.registered_model_key = model_key
            self.training_metrics = {"train_loss": 0.0, "eval_loss": 0.0, "train_runtime": 0, "train_samples_per_second": 0}
            self.model_metadata = {"stub": True, "base_model": "stub", "training_config": {"lora_rank": self.lora_rank}}

            # Clean up temp dir after registration
            shutil.rmtree(dummy_dir, ignore_errors=True)

            # Skip the heavy rest of the function and go to next step
            self.next(self.end)
            return
        
        base_model = "Qwen/Qwen2.5-VL-3B-Instruct"
        
        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        
        # Load model with conservative configuration to avoid initialization issues
        try:
            # First try loading config separately to debug any issues
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
            print(f"Config loaded successfully: {type(config)}")
            
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_model,
                config=config,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        except Exception as e:
            print(f"Error loading Qwen2.5-VL model: {e}")
            print("Falling back to a different model approach...")
            raise e
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.lora_rank,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        
        model = get_peft_model(model, peft_config)
        
        def prepare_dataset(data):
            def tokenize_function(examples):
                texts = []
                for messages in examples["messages"]:
                    text = ""
                    for msg in messages:
                        role = msg["role"]
                        content = msg["content"]
                        if isinstance(content, list):
                            text_content = next((c["text"] for c in content if c["type"] == "text"), "")
                            text += f"<|{role}|>\n{text_content}\n"
                        else:
                            text += f"<|{role}|>\n{content}\n"
                    text += "<|end|>"
                    texts.append(text)
                
                tokenized = processor.tokenizer(
                    texts,
                    truncation=True,
                    padding="longest",  # let the collator handle final padding
                    max_length=512,
                    return_tensors=None,
                )
                return tokenized
            
            dataset = Dataset.from_list(data)
            return dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
        
        train_dataset = prepare_dataset(self.training_data)
        eval_dataset = prepare_dataset(self.validation_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            training_args = TrainingArguments(
                output_dir=temp_dir,
                learning_rate=self.learning_rate,
                num_train_epochs=self.epochs,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                gradient_accumulation_steps=2,
                warmup_steps=10,
                logging_steps=5,
                eval_steps=20,
                save_steps=40,
                evaluation_strategy="steps", # pylint: disable=deprecated-argument
                save_strategy="steps",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                report_to=[],
                remove_unused_columns=False,
                dataloader_pin_memory=False,
            )
            
            data_collator = DataCollatorForLanguageModeling(tokenizer=processor.tokenizer, mlm=False)
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                tokenizer=processor.tokenizer
            )
            
            train_result = trainer.train()
            eval_result = trainer.evaluate()
            
            self.training_metrics = {
                "train_loss": train_result.training_loss,
                "eval_loss": eval_result["eval_loss"],
                "train_runtime": train_result.metrics["train_runtime"],
                "train_samples_per_second": train_result.metrics["train_samples_per_second"]
            }
            
            # Save model locally first
            model_save_path = os.path.join(temp_dir, "final_model")
            model.save_pretrained(model_save_path)
            processor.save_pretrained(model_save_path)
            
            # Create model metadata
            model_metadata = {
                "model_type": "lora_finetuned_vlm",
                "base_model": base_model,
                "training_config": {
                    "learning_rate": self.learning_rate,
                    "num_epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "lora_rank": self.lora_rank
                },
                "training_metrics": self.training_metrics,
                "dataset_metadata": self.dataset_metadata,
                "created_at": datetime.now().isoformat(),
                "run_id": current.run_id
            }
            
            # Use @model decorator functionality to save model with metadata
            self.model_id = current.model.save(model_save_path, metadata=model_metadata)
            
            # Register the model in the asset registry
            from assets import Asset
            asset = Asset(project='photo_tagger', branch='main')
            
            model_key = f"finetuned_vlm_{current.run_id}"
            asset.register_model_asset(
                name=model_key,
                description="LoRA fine-tuned VLM model trained on photo tagging data",
                kind="pytorch_lora",
                blobs=[model_save_path]
            )
            self.registered_model_key = model_key
            self.model_metadata = model_metadata
        
        self.next(self.end)
    
    @card(type="blank")
    @step
    def end(self):
        print(f"Fine-tuning completed!")
        print(f"Training loss: {self.training_metrics['train_loss']:.4f}")
        print(f"Eval loss: {self.training_metrics['eval_loss']:.4f}")
        print(f"Training runtime: {self.training_metrics['train_runtime']:.2f}s")
        print(f"Model registered: {self.registered_model_key}")
        print(f"Model saved with ID: {self.model_id}")
        print(f"Base model: {self.model_metadata['base_model']}")
        print(f"LoRA rank: {self.model_metadata['training_config']['lora_rank']}")
        
        self.final_results = {
            "model_key": self.registered_model_key,
            "model_id": self.model_id,
            "training_metrics": self.training_metrics,
            "model_metadata": self.model_metadata
        }

if __name__ == "__main__":
    FineTuneVLM() 