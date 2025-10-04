# tourism_project/hosting/hosting.py
import os
import time
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

REPO_ID = "vamsikrishna1516/Tourism-Prediction-App"
LOCAL_APP_FOLDER = "tourism_project/deployment"
REPO_TYPE = "space"

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN not found in environment. Please set the secret HF_TOKEN with a Hugging Face token that has Space creation permissions.")

api = HfApi(token=hf_token)

# Diagnostics: check token identity
try:
    user = api.whoami()
    print("Authenticated as HF user:", user.get("name"))
except Exception as e:
    print("Failed to get HF identity (token may be invalid):", e)
    raise

# Ensure local folder exists
if not os.path.isdir(LOCAL_APP_FOLDER):
    raise FileNotFoundError(f"Local app folder not found: {LOCAL_APP_FOLDER}. Please ensure files (app.py, requirements.txt) are present.")

# Create Space if not exists (with retries)
space_exists = False
try:
    api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
    print(f"Space '{REPO_ID}' already exists.")
except RepositoryNotFoundError:
    print(f"Space '{REPO_ID}' not found. Creating a new Streamlit Space...")
    create_repo(
        "Tourism-Prediction-App",
        repo_type="space",
        space_sdk="streamlit",
        private=False,
        exist_ok=True
    )
    print(f"Space '{REPO_ID}' created successfully!")

    # wait & verify creation (small backoff)
    for attempt in range(8):
        try:
            api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
            print("Space is now available.")
            space_exists = True
            break
        except RepositoryNotFoundError:
            print(f"Space not available yet, retrying ({attempt+1}/8)...")
            time.sleep(3)
    if not space_exists:
        raise RuntimeError("Space was not found after creation attempts. Check token permissions and that the repo_id is correct.")

# Finally upload the folder
print(f"Uploading '{LOCAL_APP_FOLDER}' to Space: {REPO_ID} ...")
try:
    api.upload_folder(
        folder_path=LOCAL_APP_FOLDER,
        repo_id=REPO_ID,
        repo_type="space",
        path_in_repo="",
    )
    print("Upload completed successfully.")
except Exception as e:
    print("Upload failed:", e)
    raise
