from metaflow import (
    FlowSpec,
    step,
    current,
    project,
    Flow,
    card,
    retry,
    Parameter,
    namespace
)
import json
import random
from datetime import datetime, timezone
from dateutil.parser import isoparse
from assets import Asset


@project(name="photo_tagger")
class PrepareVLMData(FlowSpec):
    
    min_tags = Parameter("min-tags", default=5, help="Minimum number of tags per photo")
    max_samples = Parameter("max-samples", default=10000, help="Maximum samples to collect")
    val_split = Parameter("val-split", default=0.2, help="Validation split ratio")
    lookback_runs = Parameter("lookback-runs", default=50, help="Number of UpdatePhotos runs to examine")
    namespace = Parameter("data-ns", default="", help="Namespace to filter runs by")
    force_full_rebuild = Parameter("force-rebuild", default=False, help="Force full recomputation instead of incremental")
    incremental = Parameter("incremental", default=True, help="Only process new runs since last execution")

    @card(type="phototagger")
    @step
    def start(self):
        if self.namespace != current.namespace and self.namespace != "":
            if self.namespace == "global" or self.namespace == "none" or self.namespace == "null":
                print(f"Resetting namespace to global")
                namespace(None)
            else:
                print(f"Setting namespace to {self.namespace}")
                namespace(self.namespace)
        
        print(f"Collecting photo data for VLM fine-tuning")
        print(f"Parameters: min_tags={self.min_tags}, max_samples={self.max_samples}")
        print(f"Mode: {'Full rebuild' if self.force_full_rebuild else 'Incremental'}")
        
        # Get existing data if doing incremental update
        existing_photos = {}
        last_processed_time = None
        
        if self.incremental and not self.force_full_rebuild:
            existing_photos, last_processed_time = self._load_existing_dataset()
            print(f"Loaded {len(existing_photos)} existing photos")
            if last_processed_time:
                print(f"Last processed: {last_processed_time}")
        
        # Collect new photos
        new_photos = self._collect_photos(existing_photos, last_processed_time)
        
        # Combine existing and new photos
        self.raw_photos = {**existing_photos, **new_photos}
        
        print(f"Total photos: {len(self.raw_photos)} ({len(existing_photos)} existing + {len(new_photos)} new)")
        
        self.next(self.format_training_data)
    
    def _load_existing_dataset(self):
        """Load existing photos and processing state from previous run"""
        try:
            # Get the latest successful run
            flow = Flow("PrepareVLMData")
            latest_run = flow.latest_successful_run
            
            if latest_run and hasattr(latest_run.data, 'raw_photos'):
                existing_photos = latest_run.data.raw_photos
                metadata = getattr(latest_run.data, 'processing_metadata', {})
                last_processed_time = metadata.get('last_processed_time')
                
                print(f"Found previous run: {latest_run.pathspec}")
                return existing_photos, last_processed_time
            else:
                print("No previous successful run found")
                return {}, None
                
        except Exception as e:
            print(f"Could not load existing dataset: {e}")
            return {}, None
    
    def _collect_photos(self, existing_photos, last_processed_time):
        """Efficiently collect only new photos since last processing"""
        new_photos = {}
        runs_processed = 0
        skipped_old_runs = 0
        
        try:
            # Iterate through UpdatePhotos runs (newest first)
            update_runs = list(Flow("UpdatePhotos"))
            
            for run in update_runs:
                if runs_processed >= self.lookback_runs:
                    break
                
                # Check if run has photo data
                if not (hasattr(run, 'data') and hasattr(run.data, 'photos')):
                    continue
                
                # Skip old runs if doing incremental update
                if last_processed_time and hasattr(run, 'created_at'):
                    run_time = run.created_at
                    if isinstance(run_time, str):
                        run_time = isoparse(run_time)
                    if isinstance(last_processed_time, str):
                        last_processed_time = isoparse(last_processed_time)
                    
                    if run_time <= last_processed_time:
                        skipped_old_runs += 1
                        continue
                
                # Process this run
                run_photos = run.data.photos
                run_new_photos = 0
                run_total_photos = len(run_photos)
                
                for photo_id, photo_data in run_photos.items():
                    # Skip if we already have this photo (from existing or current collection)
                    if photo_id in existing_photos or photo_id in new_photos:
                        continue
                    
                    # Apply quality filters
                    if self._is_valid_photo(photo_data):
                        new_photos[photo_id] = photo_data
                        run_new_photos += 1
                    
                    # Stop if we've hit the max samples limit
                    total_photos = len(existing_photos) + len(new_photos)
                    if total_photos >= self.max_samples:
                        break
                
                if run_total_photos > 0:
                    print(f"Run {run.pathspec}: {run_total_photos} photos, {run_new_photos} new valid")
                
                runs_processed += 1
                
                # Stop if we've hit the max samples limit
                total_photos = len(existing_photos) + len(new_photos)
                if total_photos >= self.max_samples:
                    print(f"Reached max samples limit: {self.max_samples}")
                    break
            
            if skipped_old_runs > 0:
                print(f"Skipped {skipped_old_runs} old runs (already processed)")
            
            # Store processing metadata
            self.processing_metadata = {
                'last_processed_time': datetime.now(timezone.utc).isoformat(),
                'runs_processed': runs_processed,
                'skipped_old_runs': skipped_old_runs,
                'incremental_mode': self.incremental and not self.force_full_rebuild
            }
            
        except Exception as e:
            print(f"Error accessing UpdatePhotos runs: {e}")
            print("Creating sample data for testing...")
            new_photos = self._create_sample_data()
            self.processing_metadata = {
                'last_processed_time': datetime.now(timezone.utc).isoformat(),
                'sample_data_used': True
            }
        
        return new_photos
    
    def _is_valid_photo(self, photo_data):
        required_fields = ['id', 'image_url', 'ground_truth_tags']
        if not all(field in photo_data for field in required_fields):
            return False
            
        if len(photo_data.get('ground_truth_tags', [])) < self.min_tags:
            return False
            
        image_url = photo_data.get('image_url', '')
        if not image_url or not image_url.startswith(('http://', 'https://')):
            return False
            
        tags = photo_data.get('ground_truth_tags', [])
        if any(len(tag.strip()) < 2 for tag in tags):
            return False
                
        return True
    
    def _create_sample_data(self):
        
        sample_photos = {
            "sample_landscape_1": {
                "id": "sample_landscape_1",
                "image_url": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4",
                "ground_truth_tags": ["mountain", "landscape", "nature", "sky", "clouds", "peak", "scenic", "outdoor", "terrain"],
                "label": "Mountain landscape with cloudy sky",
                "timestamp": "2024-01-15T10:30:00Z"
            },
            "sample_ocean_1": {
                "id": "sample_ocean_1",
                "image_url": "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b",
                "ground_truth_tags": ["ocean", "water", "blue", "waves", "beach", "shore", "marine", "aquatic", "coastal"],
                "label": "Ocean waves at the beach",
                "timestamp": "2024-01-16T14:20:00Z"
            },
            "sample_city_1": {
                "id": "sample_city_1", 
                "image_url": "https://images.unsplash.com/photo-1449824913935-59a10b8d2000",
                "ground_truth_tags": ["city", "building", "urban", "architecture", "street", "modern", "metropolitan", "structure", "downtown"],
                "label": "Modern city skyline",
                "timestamp": "2024-01-17T09:45:00Z"
            }
        }
        
        photos = {}
        for i in range(min(100, self.max_samples)):
            base_key = list(sample_photos.keys())[i % len(sample_photos)]
            base_photo = sample_photos[base_key].copy()
            base_photo["id"] = f"{base_key}_{i}"
            photos[base_photo["id"]] = base_photo
            
        return photos

    @step
    def format_training_data(self):
        print(f"Formatting {len(self.raw_photos)} photos for instruction tuning")
        
        photo_items = list(self.raw_photos.values())
        random.shuffle(photo_items)
        
        split_idx = int((1 - self.val_split) * len(photo_items))
        train_photos = photo_items[:split_idx]
        val_photos = photo_items[split_idx:]
        
        train_examples = [self._format_instruction_example(photo) for photo in train_photos]
        val_examples = [self._format_instruction_example(photo) for photo in val_photos]
        self.training_dataset = {
            "train": train_examples,
            "validation": val_examples,
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "train_size": len(train_examples),
                "validation_size": len(val_examples),
                "total_size": len(photo_items),
                "validation_split": self.val_split,
                "min_tags_filter": self.min_tags,
                "format_version": "vlm_instruction_v2.1_incremental",
                "task": "photo_tagging",
                "tags_per_photo": 9,
                "processing": self.processing_metadata
            }
        }
        
        print(f"Formatted {len(train_examples)} training examples")
        print(f"Formatted {len(val_examples)} validation examples")
        
        self.next(self.validate_dataset)
    
    def _format_instruction_example(self, photo):
        tags = photo["ground_truth_tags"][:9]
        while len(tags) < 9:
            tags.append("image")
        
        try:
            with open('prompt.txt', 'r') as f:
                production_prompt = f.read().strip()
        except FileNotFoundError:
            production_prompt = "You are an image tagging assistant. Return a JSON object with a list of 9 relevant tags for the image.\n\nRespond with this format:\n\n{ \"tags\": [\"tag1\", \"tag2\", \"tag3\", \"tag4\", \"tag5\", \"tag6\", \"tag7\", \"tag8\", \"tag9\"] }"
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": production_prompt
                    },
                    {
                        "type": "image",
                        "image": photo["image_url"]
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({"tags": tags})
                    }
                ]
            }
        ]
        
        return {
            "messages": conversation,
            "photo_id": photo["id"],
            "image_url": photo["image_url"],
            "ground_truth_tags": tags,
            "original_label": photo.get("label", "")
        }

    @step 
    def validate_dataset(self):
        print(f"Validating dataset quality...")
        
        validation_results = {
            "total_examples": len(self.training_dataset["train"]) + len(self.training_dataset["validation"]),
            "train_examples": len(self.training_dataset["train"]),
            "validation_examples": len(self.training_dataset["validation"]),
            "validation_errors": [],
            "quality_metrics": {}
        }
        
        for i, example in enumerate(self.training_dataset["train"][:10]):
            errors = self._validate_example(example, f"train[{i}]")
            validation_results["validation_errors"].extend(errors)
        
        all_tags = []
        for example in self.training_dataset["train"]:
            all_tags.extend(example["ground_truth_tags"])
            
        unique_tags = set(all_tags)
        validation_results["quality_metrics"] = {
            "unique_tags": len(unique_tags),
            "avg_tags_per_example": len(all_tags) / len(self.training_dataset["train"]),
            "most_common_tags": list(sorted(unique_tags))[:20]
        }
        
        self.validation_results = validation_results
        
        if validation_results["validation_errors"]:
            print(f"Found {len(validation_results['validation_errors'])} validation issues")
            for error in validation_results["validation_errors"][:5]:
                print(f"  - {error}")
        else:
            print(f"Dataset validation passed!")
            
        print(f"Quality metrics:")
        print(f"  - Unique tags: {validation_results['quality_metrics']['unique_tags']}")
        print(f"  - Avg tags per example: {validation_results['quality_metrics']['avg_tags_per_example']:.1f}")
        
        self.next(self.register_dataset)
    
    def _validate_example(self, example, example_id):
        errors = []
        
        required_fields = ["messages", "photo_id", "image_url", "ground_truth_tags"]
        for field in required_fields:
            if field not in example:
                errors.append(f"{example_id}: Missing field '{field}'")
        
        if "messages" in example:
            messages = example["messages"]
            if len(messages) != 2:
                errors.append(f"{example_id}: Expected 2 messages, got {len(messages)}")
            
            expected_roles = ["user", "assistant"]
            for i, msg in enumerate(messages):
                if msg.get("role") != expected_roles[i]:
                    errors.append(f"{example_id}: Message {i} has wrong role")
        
        if "ground_truth_tags" in example:
            tags = example["ground_truth_tags"]
            if len(tags) != 9:
                errors.append(f"{example_id}: Expected 9 tags, got {len(tags)}")
        
        return errors

    @retry
    @step
    def register_dataset(self):
        print(f"Registering VLM training dataset as asset...")
        
        try:
            asset = Asset(project='photo_tagger', branch='main')
            
            asset.register_data_asset(
                name="vlm_training_data",
                description=f"VLM fine-tuning dataset for photo tagging - {self.training_dataset['metadata']['train_size']} train, {self.training_dataset['metadata']['validation_size']} val examples",
                kind="artifact",
                blobs=["training_dataset", "validation_results"]
            )
            
            self.asset_registered = True
            print(f"Dataset registered successfully!")
            
        except Exception as e:
            print(f"Failed to register asset: {e}")
            self.asset_registered = False
        
        self.next(self.end)

    @card(type="phototagger")
    @step
    def end(self):
        processing_info = self.training_dataset["metadata"]["processing"]
        
        summary = {
            "status": "success" if self.asset_registered else "partial_success",
            "total_examples": self.training_dataset["metadata"]["total_size"],
            "train_examples": self.training_dataset["metadata"]["train_size"],
            "validation_examples": self.training_dataset["metadata"]["validation_size"],
            "unique_tags": self.validation_results["quality_metrics"]["unique_tags"],
            "dataset_ready_for_training": self.asset_registered,
            "processing_mode": "incremental" if processing_info.get("incremental_mode") else "full_rebuild",
            "runs_processed": processing_info.get("runs_processed", 0),
            "skipped_old_runs": processing_info.get("skipped_old_runs", 0)
        }
        
        print(f"\nVLM Data Curation Complete!")
        print(f"Dataset Summary:")
        print(f"  - Total examples: {summary['total_examples']}")
        print(f"  - Training examples: {summary['train_examples']}")
        print(f"  - Validation examples: {summary['validation_examples']}")
        print(f"  - Unique tags: {summary['unique_tags']}")
        print(f"  - Asset registered: {summary['dataset_ready_for_training']}")
        
        print(f"\nProcessing Efficiency:")
        print(f"  - Mode: {summary['processing_mode']}")
        print(f"  - Runs processed: {summary['runs_processed']}")
        if summary['skipped_old_runs'] > 0:
            print(f"  - Runs skipped (already processed): {summary['skipped_old_runs']}")
            efficiency = summary['skipped_old_runs'] / (summary['runs_processed'] + summary['skipped_old_runs']) * 100
            print(f"  - Efficiency gain: {efficiency:.1f}% fewer runs processed")
        
        if summary['dataset_ready_for_training']:
            print(f"\nReady for VLM fine-tuning!")
            print(f"Next step: Run VLM fine-tuning workflow")
        else:
            print(f"\nManual review needed before fine-tuning")


if __name__ == "__main__":
    PrepareVLMData() 