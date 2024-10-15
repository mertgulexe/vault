HUGGINGFACE_TOKEN = None

with open(file="/home/mert/.cache/huggingface/token", mode="r") as f:
    HUGGINGFACE_TOKEN = f.read()
    assert HUGGINGFACE_TOKEN is not None, "HF token couldn't be found."