"""Publish the adapter, datasets and results to Hugging Face, plus public model and dataset collections."""
import sys

from huggingface_hub import HfApi

api = HfApi()
user = api.whoami()["name"]
base_model = sys.argv[1]
model_repo = f"{user}/firstaid-agent-tiny-aya-lora"
data_repo = f"{user}/firstaid-sim"

api.create_repo(model_repo, exist_ok=True, private=False)
api.upload_folder(repo_id=model_repo, folder_path="out/lora", commit_message="LoRA adapter")
api.upload_file(repo_id=model_repo, path_or_fileobj="README.md", path_in_repo="README.md")

api.create_repo(data_repo, repo_type="dataset", exist_ok=True, private=False)
api.upload_folder(repo_id=data_repo, repo_type="dataset", folder_path="artifacts", path_in_repo="data")
api.upload_folder(repo_id=data_repo, repo_type="dataset", folder_path="results", path_in_repo="results")
api.upload_file(repo_id=data_repo, repo_type="dataset", path_or_fileobj="README.md", path_in_repo="README.md")
for f in ("sim.py", "data.py", "evaluate.py", "train.py"):
    api.upload_file(repo_id=data_repo, repo_type="dataset", path_or_fileobj=f, path_in_repo=f"code/{f}")

mc = api.create_collection(title="FirstAid-Agent models", namespace=user, private=False, exists_ok=True,
                           description="Safety-gated multilingual first-aid agent (Lexsi hackathon, Track 3)")
api.add_collection_item(mc.slug, item_id=model_repo, item_type="model", exists_ok=True)
api.add_collection_item(mc.slug, item_id=base_model, item_type="model", exists_ok=True,
                        note="Base model (lower bound)")
dc = api.create_collection(title="FirstAid-Agent data", namespace=user, private=False, exists_ok=True,
                           description="FirstAid-Sim scenarios, teacher trajectories, DPO pairs, eval logs")
api.add_collection_item(dc.slug, item_id=data_repo, item_type="dataset", exists_ok=True)
print("MODEL", f"https://huggingface.co/{model_repo}")
print("DATA", f"https://huggingface.co/datasets/{data_repo}")
print("MODEL_COLLECTION", f"https://huggingface.co/collections/{mc.slug}")
print("DATA_COLLECTION", f"https://huggingface.co/collections/{dc.slug}")
