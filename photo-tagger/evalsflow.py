from metaflow import (
    FlowSpec,
    step,
    project,
    current,
    S3,
    card,
    Parameter,
    resources,
    conda_base,
    schedule,
    Run,
    retry,
    namespace,
)
from datetime import datetime, timedelta, timezone
from collections import defaultdict, Counter
import gzip
import time
import json
import re

DATE = re.compile(r"year=(\d{4})/month=(\d{2})/day=(\d{2})")

OPACITY = 'BF' # 75%
COLORS = [
    "#EF4A3C",
    "#F77440",
    "#F3993E",
    "#F6C053",
    "#F6E072",
    "#D0E181",
    "#B0DE9F",
    "#C4E9D0",
    "#E0F1F2",
    "#F4FAFA",
]


def timed_prefixes(root, namespace, num):
    today = datetime.now(timezone.utc)
    pattern = "{root}/namespace={namespace}/tag=evals/year={year}/month={month:02d}/day={day:02d}/"
    prefixes = []
    for i in range(num):
        day = today - timedelta(days=i)
        yield pattern.format(
            root=root, namespace=namespace, year=day.year, month=day.month, day=day.day
        )


def parse_date(url):
    m = DATE.search(url)
    return "-".join(m.groups())


def read_lines(namespace, num):
    from metaflow.metaflow_config import DATASTORE_SYSROOT_S3

    s3_bucket_name = DATASTORE_SYSROOT_S3.split("/")[2]
    root = "s3://" + s3_bucket_name + "/usm/logs"
    with S3() as s3:
        files = s3.list_recursive(list(timed_prefixes(root, namespace, num)))
        for obj in s3.get_many([f.url for f in files]):
            date = parse_date(obj.url)
            with gzip.open(obj.path, mode="rt") as f:
                for line in f:
                    yield date, json.loads(line)


@schedule(hourly=True)
@project(name="photo_tagger")
class EvaluationFlow(FlowSpec):
    @step
    def start(self):
        self.next(self.evaluate)

    @retry
    @card(type="phototagger")
    @step
    def evaluate(self):
        models = defaultdict(Counter)
        totals = defaultdict(Counter)
        batches = {}
        no_cell_color = {"latency"}

        # Helper to accumulate stats
        def _add_metric(row_id, batch_key, recall, latency=None):
            models[row_id][batch_key] += recall
            totals[row_id][batch_key] += 1
            models[row_id]["total"] += recall
            totals[row_id]["total"] += 1
            if latency is not None:
                models[row_id]["latency"] += latency
                totals[row_id]["latency"] += 1

        for date, obj in read_lines("jobs-default", 10):
            try:
                msg = obj["content"]
                batch_id = msg["batch_id"]
                provider = msg["provider"]
                model = msg["model"]
                recall = msg["recall"]
                latency = msg["latency"]
            except KeyError:
                continue

            row_id = f"{provider} {model}"
            batch_key = f"col_{batch_id.split('/')[-1].split('-')[-1]}"
            batches[batch_key] = batch_id
            _add_metric(row_id, batch_key, recall, latency)

        # ------------------------------------------------------------------
        # ALSO ingest metrics directly from the latest PromptCustomModels run.
        # This guarantees fine-tuned results appear even if log scraping fails.
        # ------------------------------------------------------------------
        try:
            from metaflow import Flow
            pcm_run = Flow("PromptCustomModels").latest_successful_run
            if pcm_run:
                pcm_batch_id = pcm_run.data.models_eval["photos"][0].get("batch_id", pcm_run.pathspec)
                pcm_batch_key = f"col_{pcm_batch_id.split('/')[-1].split('-')[-1]}"
                batches[pcm_batch_key] = pcm_batch_id

                for photo in pcm_run.data.models_eval["photos"]:
                    for res in photo["models"]:
                        provider = res.get("provider", "Finetuned")
                        model = res["model"]
                        recall = res["recall"]
                        row_id = f"{provider.lower()} {model}"
                        _add_metric(row_id, pcm_batch_key, recall)
        except Exception as _:
            # Safe-guard: if anything goes wrong we just skip this augmentation.
            pass

        rows = defaultdict(dict)
        for row_id, row in models.items():
            r = rows[row_id]

            r["_highlight"] = {"cell": {}}
            for key in row:
                avg = row[key] / totals[row_id][key]
                r[key] = f"{int(avg)} ({totals[row_id][key]})"
                if key not in no_cell_color:
                    r["_highlight"]["cell"][key] = COLORS[min(int(round(avg / 10)), 9)] + OPACITY
            r["model"] = row_id

        row_list = [
            r | {"_id": i}
            for i, r in enumerate(
                sorted(rows.values(), key=lambda x: int(x["total"].split(' ')[0]), reverse=True)
            )
        ]

        cols = [
            {"key": "model", "label": "Model"},
            {"key": "latency", "label": "Avg Latency (ms) | Offline"},
            {"key": "total", "label": "Avg Recall | Offline"},
        ]

        namespace(None)
        col_iter = (
            (Run(pathspec).created_at, batch_key)
            for batch_id, pathspec in batches.items()
        )
        cols.extend(
            {"key": batch_key, "label": f'{batch_key[4:]} ({d.strftime("%m-%d")})'}
            for d, batch_key in sorted(col_iter)
        )
        self.evals = {"headers": cols, "rows": row_list}
        print(self.evals)
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    EvaluationFlow()
