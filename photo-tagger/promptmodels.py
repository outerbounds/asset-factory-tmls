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
)
import os, time, json
from datetime import datetime, timezone
from dateutil.parser import isoparse
from metaflow.cards import Markdown, Image

import requests
from highlight import highlight


@trigger_on_finish(flow="UpdatePhotos")
@project(name="photo_tagger")
class PromptModels(FlowSpec):

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

        if hasattr(current, 'trigger') and current.trigger:
            # If triggered, get data from the triggering run
            self.unseen = current.trigger.run.data.unseen
            self.batch_id = current.trigger.run.pathspec
            print(f"Flow triggered by {self.batch_id}, using its data.")
        else:
            # If run manually, get the latest UpdatePhotos run directly
            print("Flow run manually, fetching latest successful UpdatePhotos run.")
            if self.data_namespace:
                namespace(self.data_namespace)
                print(f"Switched to namespace: {self.data_namespace}")
            
            run = Flow('UpdatePhotos').latest_successful_run
            self.unseen = run.data.unseen
            self.batch_id = run.pathspec
            print(f"Using data from run: {self.batch_id}")
        
        # Vendor APIs only
        self.models = [
            ("togetherai", "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"),
            ("togetherai", "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo"),
            ("openai", "o4-mini-2025-04-16"),
            ("openai", "gpt-4.1-2025-04-14"),
        ]
        
        print(f"Total vendor models to evaluate: {len(self.models)}")
        self.next(self.prompt_model, foreach="models")

    @retry
    @pypi(packages={"openai": "1.82.1"})
    @card(type="phototagger")
    @secrets(sources=["outerbounds.openai", "outerbounds.togetherai"])
    @step
    def prompt_model(self):
        from prompter import Prompter
        
        self.provider, self.model = self.input

        responses = []
        
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

        models_eval = {}
        batch_id = inputs[0].batch_id
        for photo in inputs[0].unseen.values():
            models_eval[photo["id"]] = photo
            models_eval[photo["id"]]["models"] = []

        for inp in inputs:
            responses = inp.model_responses["responses"]
            for photo in responses:
                ground_truth = frozenset(models_eval[photo["id"]]["ground_truth_tags"])
                num = len(ground_truth)
                highlighted = [
                    {"label": t, "highlight": t in ground_truth} for t in photo["tags"]
                ]
                recall = int(100 * sum(1 for t in highlighted if t["highlight"]) / num)
                model_name = inp.model
                res = {
                    "model": model_name,
                    "provider": inp.provider.capitalize(),
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

        self.highlight.add_label("product_xp")
        self.highlight.add_label("live")
        self.highlight.set_title("A/B Experiment: Blue vs. Green")
        for i, x in enumerate(["Blue variant", "Green variant", "Control"]):
            self.highlight.add_line(x, caption=f"Slot {i}")
        self.highlight.add_column("8%", "click rate")
        self.highlight.add_column("34%", "completion rate")

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    PromptModels()
