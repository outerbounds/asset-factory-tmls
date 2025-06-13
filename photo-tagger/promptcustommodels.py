from metaflow import (
    card,
    FlowSpec,
    step,
    current,
    pypi,
    project,
    Flow,
    retry,
    secrets,
    Config,
    trigger_on_finish,
    Parameter,
    Run,
    namespace,
    kubernetes,
    environment,
)
import os, time, json
from datetime import datetime, timezone
from dateutil.parser import isoparse
from metaflow.cards import Markdown, Image
from metaflow.profilers import gpu_profile

import requests
from highlight import highlight


@trigger_on_finish(flow="FineTuneVLM")
@project(name="photo_tagger")
class PromptCustomModels(FlowSpec):

    data_namespace = Parameter(
        'data-ns',
        default='',
        help="The namespace to pull data assets and models from."
    )
    prompt = Config("prompt", default="prompt.txt", parser=lambda x: {"prompt": x.strip()})
    obhost = Config("obhost", default="obhost.txt", parser=lambda x: {'obhost': x.strip()})

    @step
    def start(self):
        from metaflow import Flow

        # Get the latest UpdatePhotos run for data
        print("Fetching latest successful UpdatePhotos run for data.")
        if self.data_namespace:
            namespace(self.data_namespace)
            print(f"Switched to namespace: {self.data_namespace}")
        
        update_photos_run = Flow('UpdatePhotos').latest_successful_run
        self.unseen = update_photos_run.data.unseen
        self.batch_id = update_photos_run.pathspec
        print(f"Using data from UpdatePhotos run: {self.batch_id}")
        
        # Get the fine-tuned model from the triggering run or latest successful run
        if hasattr(current, 'trigger') and current.trigger:
            # If triggered by FineTuneVLM, use that model
            self.model_id = current.trigger.run.data.model_id
            self.ft_run_id = current.trigger.run.pathspec
            print(f"Flow triggered by {self.ft_run_id}, using its model.")
        else:
            # If run manually, get the latest FineTuneVLM run
            print("Flow run manually, fetching latest successful FineTuneVLM run.")
            ft_flow = Flow('FineTuneVLM')
            latest_ft = ft_flow.latest_successful_run
            if latest_ft and latest_ft.successful:
                self.model_id = latest_ft.data.model_id
                self.ft_run_id = latest_ft.pathspec
                print(f"Using model from run: {self.ft_run_id}")
            else:
                raise ValueError("No successful FineTuneVLM run found")
        
        # Custom models to evaluate
        self.models = [("finetuned", self.model_id)]
        
        print(f"Total custom models to evaluate: {len(self.models)}")
        self.next(self.prompt_model, foreach="models")

    @environment(vars={"TOKENIZERS_PARALLELISM": "false"})
    @gpu_profile(interval=1)
    @pypi(packages={
        "openai": "1.82.1",
        "torch": "2.4.1", 
        "transformers": "4.49.0", 
        "peft": "0.13.2",
        "pillow": "10.4.0",
        "torchvision": "0.19.1",
        "accelerate": "1.1.1",
        "datasets": "3.1.0",
        "bitsandbytes": "0.43.0"
    })
    @kubernetes(gpu=1, memory=32000)
    @card(type="phototagger")
    @step
    def prompt_model(self):
        from prompter import Prompter
        
        self.provider, self.model = self.input

        responses = []
        
        # The Prompter class knows how to handle the Model object for 'finetuned' provider
        prompter = Prompter(self.provider, self.model, obhost=self.obhost['obhost'])
        
        for photo in self.unseen.values():
            resp, tags, validity, latency = prompter.prompt(
                self.prompt["prompt"], image_url=photo["image_url"]
            )
            responses.append(
                {
                    "id": photo["id"],
                    "image_url": photo["image_url"],
                    "latency": latency,
                    "tags": tags,
                    "validity": validity,
                    "response": resp,
                }
            )
        self.model_responses = {
            "provider": self.provider,
            "model": self.model,
            "responses": responses,
        }

        self.next(self.join)

    @highlight
    @card(type="phototagger")
    @step
    def join(self, inputs):
        from evals_logger import EvalsLogger
        logger = EvalsLogger(project='photo_tagger', branch='main')

        # Get the latest vendor model results from PromptModels
        print("Fetching latest vendor model results from PromptModels flow...")
        try:
            vendor_flow = Flow('PromptModels')
            latest_vendor_run = vendor_flow.latest_successful_run
            if latest_vendor_run:
                vendor_models_eval = latest_vendor_run.data.models_eval
                print(f"Found vendor results from run: {latest_vendor_run.pathspec}")
            else:
                print("No vendor results found, creating empty baseline")
                vendor_models_eval = {"photos": []}
        except Exception as e:
            print(f"Could not fetch vendor results: {e}")
            vendor_models_eval = {"photos": []}

        # Process custom model results
        models_eval = {}
        batch_id = inputs[0].batch_id
        
        # Initialize with photos from current run
        for photo in inputs[0].unseen.values():
            models_eval[photo["id"]] = photo
            models_eval[photo["id"]]["models"] = []

        # Add vendor model results if available
        if vendor_models_eval["photos"]:
            for vendor_photo in vendor_models_eval["photos"]:
                photo_id = vendor_photo["id"]
                if photo_id in models_eval:
                    models_eval[photo_id]["models"].extend(vendor_photo["models"])

        # Add custom model results
        for inp in inputs:
            responses = inp.model_responses["responses"]
            for photo in responses:
                ground_truth = frozenset(models_eval[photo["id"]]["ground_truth_tags"])
                num = len(ground_truth)
                highlighted = [
                    {"label": t, "highlight": t in ground_truth} for t in photo["tags"]
                ]
                recall = int(100 * sum(1 for t in highlighted if t["highlight"]) / num)
                if inp.provider == "finetuned":
                    if isinstance(inp.model, dict):
                        model_name = inp.model.get('model_uuid', 'finetuned')
                    else:
                        model_name = str(inp.model).split('_')[-1]
                else:
                    model_name = inp.model
                res = {
                    "model": model_name,
                    "provider": "Finetuned",
                    "recall": recall,
                    "tags": highlighted,
                }
                models_eval[photo["id"]]["models"].append(res)
                logger.log(
                    {
                        "model": inp.model,
                        "provider": inp.provider,
                        "photo_id": photo["id"],
                        "latency": photo["latency"],
                        "batch_id": batch_id,
                        "recall": recall,
                    }
                )

        self.models_eval = {"photos": list(models_eval.values())}

        # Beautiful unified visualization with all models
        total_models = sum(len(photo["models"]) for photo in self.models_eval["photos"]) // len(self.models_eval["photos"])
        
        self.highlight.add_label("unified_eval")
        self.highlight.add_label("live")
        self.highlight.set_title(f"🚀 Unified Model Evaluation: {total_models} Models")
        
        # Show model performance summary
        model_performance = {}
        for photo in self.models_eval["photos"]:
            for model_result in photo["models"]:
                model_key = f"{model_result['model']} ({model_result['provider']})"
                if model_key not in model_performance:
                    model_performance[model_key] = []
                model_performance[model_key].append(model_result['recall'])
        
        # Compute and surface aggregate statistics for each model
        import statistics

        for model_key, recalls in model_performance.items():
            avg_recall = statistics.mean(recalls)
            median_recall = statistics.median(recalls)
            best_recall = max(recalls)

            # Add a nicely formatted line to the highlight card
            self.highlight.add_line(
                f"{model_key}: avg {avg_recall:.1f}% | median {median_recall:.1f}% | best {best_recall}%",
                caption=f"{len(recalls)} photos"
            )

            # Log the aggregate stats once per model
            logger.log({
                "model": model_key,
                "avg_recall": avg_recall,
                "median_recall": median_recall,
                "best_recall": best_recall,
                "batch_id": batch_id,
            })

        self.highlight.add_column("Vendor APIs", "4 models")
        self.highlight.add_column("Fine-tuned", "1 model")
        self.highlight.add_column("Total", f"{total_models} models")

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    PromptCustomModels() 