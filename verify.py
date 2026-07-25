# import numpy as np
# data = np.random.randn(1000)
# data.mean(), data.std()

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, snapshot_download
import os

load_dotenv()


# dataset = load_dataset("stanfordnlp/imdb")
# print(dataset)
# print(dataset["train"][0])

# dataset_wiki = load_dataset(
#     "wikimedia/wikipedia", "20231101.en", split="train", streaming=True
# )

# for i, example in enumerate(dataset_wiki):
#     print(example["title"])
#     if i >= 4:
#         break

# dataset = load_dataset("stanfordnlp/imdb", split="train")

# dataset.to_csv("stanfordnlp/imdb_train.csv")
# dataset.to_json("stanfordnlp/imdb_train.json")
# dataset.to_parquet("stanfordnlp/imdb_train.parquet")

# split = dataset.train_test_split(test_size=0.2, seed=42)
# train_val = split["train"].train_test_split(test_size=0.125, seed=42)

# train_ds = train_val["train"]
# val_ds = train_val["test"]
# test_ds = split["test"]

# print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")


# model_path = hf_hub_download(
#     repo_id="sentence-transformers/all-MiniLM-L6-v2", filename="config.json"
# )
# print(f"Cached at: {model_path}")

# model_dir = snapshot_download("sentence-transformers/all-MiniLM-L6-v2")
# print(f"Full model at: {model_dir}")


local_path = os.path.expanduser("~/.cache/huggingface/datasets/")

print(local_path)
# s3_path = "s3://my-bucket/datasets/"
# gcs_path = "gs://my-bucket/datasets/"
