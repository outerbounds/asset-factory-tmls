import streamlit as st
import base64
import time
import threading

# from assets import Asset
# asset = Asset(project='demosequence', branch='main', entity_ref={"entity_kind": "task", "entity_id": f"app"})
# print(len(asset.list_model_assets()))

from evals_logger import EvalsLogger
from prompter import Prompter
#import model_auth

logger = EvalsLogger(project='photo_tagger', branch='main')

MODELS = [
    ("togetherai", "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"),
    ("togetherai", "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo"),
    ("openai", "o4-mini-2025-04-16"),
    ("openai", "gpt-4.1-2025-04-14"),
]
PROMPT = open("prompt.txt").read()

K = 4  # Number of background tasks
st.title("Model tester 📸 🧠")

st.markdown("##### Model assets\n\n%s" % '\n'.join(" - %s/%s" % x for x in MODELS))

# Input fields
user_tags = frozenset(
    t.strip().lower()
    for t in st.text_input("Enter a command-separated list of tags:").split(",")
)
photo = st.camera_input("Take a photo")


# Dummy background task
def background_task(encoded_photo, worker_id, output_store):
    provider, model = MODELS[worker_id]
    prompter = Prompter(provider, model)
    resp, tags, validity, latency = prompter.prompt(PROMPT, encoded_image=encoded_photo)
    if validity == "✅":
        output_store[worker_id] = (tags, latency)
    else:
        output_store[worker_id] = (resp, latency)


# Pill rendering with highlight
def render_pills(strings, highlight=None):
    base_style = """
        display: inline-block;
        padding: 6px 12px;
        margin: 4px 4px 4px 0;
        border-radius: 999px;
        font-size: 0.9em;
        background-color: #ebe8e5;
        color: #333;
    """
    highlight_style = """
        display: inline-block;
        padding: 6px 12px;
        margin: 4px 4px 4px 0;
        border-radius: 999px;
        font-size: 0.9em;
        background-color: #b4d3f2;
        color: #333;
    """
    return "".join(
        [
            f"<span style='{highlight_style if s.lower() in highlight else base_style}'>{s}</span>"
            for s in strings
        ]
    )


def render_result(i):
    provider, model = MODELS[i]
    resp, latency = output_store[i]
    if isinstance(resp, list):
        st.markdown(f"**Model {model}** responded in {latency}ms:")
        st.markdown(
            render_pills(resp, highlight=user_tags), unsafe_allow_html=True
        )
    else:
        st.markdown(f"**Model {model}** returned an invalid response in {latency}ms:")
        st.text_area("Response", resp, height=100, key=str(uuid4.uuid()))


if photo is not None:
    # Encode photo
    image_bytes = photo.getvalue()
    encoded_photo = base64.b64encode(image_bytes).decode("utf-8")

    # Shared state
    output_store = [None] * K
    threads = []
    result_blocks = [st.empty() for _ in range(K)]

    # Status banner while running
    status_placeholder = st.empty()
    status_placeholder.info("🔄 Prompting models...")

    # Start threads
    for i in range(K):
        provider, model = MODELS[i]
        with result_blocks[i].container():
            st.write(f"**Model {model}** ⏳ Prompting...")
        t = threading.Thread(
            target=background_task, args=(encoded_photo, i, output_store)
        )
        threads.append(t)
        t.start()

    # Polling loop
    while any(t.is_alive() for t in threads):
        for i, t in enumerate(threads):
            if not t.is_alive() and output_store[i] is not None:
                with result_blocks[i].container():
                    render_result(i)
        time.sleep(0.2)

    # Final UI update
    status_placeholder.success("✅ Results recorded for evaluation")
    for i in range(K):
        provider, model = MODELS[i]

        with result_blocks[i].container():
            if output_store[i] is not None:
                render_result(i)
                resp, latency = output_store[i]
                if isinstance(resp, list) and user_tags:
                    match = sum(1 for t in resp if t in user_tags)
                    recall = int(100 * match / len(user_tags))
                else:
                    recall = 0

                logger.log(
                    {
                        "model": model,
                        "provider": provider,
                        "photo_id": '',
                        "latency": latency,
                        "batch_id": '',
                        "recall": recall
                    }
                )

