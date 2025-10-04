%%writefile tourism_project/hosting/hosting.py
# ------------------------------
# Hosting Script — Upload Streamlit App to Hugging Face Space
# ------------------------------

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

# ------------------------------
# Authentication
# ------------------------------
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN not found in environment. Please set it before running this script.")

api = HfApi(token=hf_token)

# ------------------------------
# Configuration
# ------------------------------
local_app_folder = "tourism_project/deployment"
repo_id = "vamsikrishna1516/Tourism_Prediction_App"
repo_type = "space"

# ------------------------------
# Ensure the Space Exists
# ------------------------------
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"⚙️ Space '{repo_id}' not found. Creating a new Streamlit Space...")
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        space_sdk="streamlit",  # important for Streamlit apps
        private=False
    )
    print(f"Space '{repo_id}' created successfully!")

# ------------------------------
# Upload Folder
# ------------------------------
print(f"Uploading folder '{local_app_folder}' to Hugging Face Space: {repo_id}")

api.upload_folder(
    folder_path=local_app_folder,
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo="",
)

print("Streamlit app uploaded successfully to Hugging Face Space!")
