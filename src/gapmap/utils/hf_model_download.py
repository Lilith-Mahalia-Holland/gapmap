from __future__ import annotations


from huggingface_hub import hf_hub_download
from os.path import isfile
from tqdm import tqdm
from gapmap.utils.config import settings


download_dict = {
    "PaddlePaddle/PP-DocLayoutV3_safetensors": ["model.safetensors", "preprocessor_config.json", "config.json"],
    "MaartenGr/BERTopic_ArXiv": ["topic_embeddings.safetensors", "ctfidf.safetensors", "topics.json", "config.json"],
    "allenai/scibert_scivocab_uncased": ["pytorch_model.bin", "config.json", "vocab.txt"]
}

base_dir = settings.models_dir

for repo, file_list in tqdm(download_dict.items()):
    repo_folder_name = repo.split("/")[-1].replace("-", "_").strip()
    model_path = base_dir / repo_folder_name
    for file in file_list:
        if not isfile(model_path / file):
            hf_hub_download(repo_id=repo, filename=file, local_dir=model_path)