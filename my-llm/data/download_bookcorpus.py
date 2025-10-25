from datasets import load_dataset
import os

cache_dir="/root/autodl-tmp"
os.environ["HF_HOME"]=cache_dir
os.environ["HF_DATASETS_CACHE"]=cache_dir
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print("Loading dataset...")

load_dataset(
    "rojagtap/bookcorpus",
    split="train",
    cache_dir=cache_dir
)

print("Dataset loaded successfully.")

