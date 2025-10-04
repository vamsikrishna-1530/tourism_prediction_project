# ------------------------------
# Hosting Script — Upload Streamlit App to Hugging Face Space
# ------------------------------

from huggingface_hub import HfApi
import os

# Initialize API with token from environment
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN not found in environment. Please set it before running this script.")

api = HfApi(token=hf_token)

# ------------------------------
# Configuration
# ------------------------------
# Folder containing your Streamlit app and requirements.txt
local_app_folder = "tourism_project/deployment"

# Target Hugging Face Space repository
repo_id = "vamsikrishna1516/Tourism_Prediction_App"

# Repo type for Spaces
repo_type = "space"

# ------------------------------
# Upload Folder
# ------------------------------
print(f"Uploading '{local_app_folder}' to Hugging Face Space: {repo_id}")

api.upload_folder(
    folder_path=local_app_folder,
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo="",  # Upload to root of the Space
)

print("Streamlit app uploaded successfully to Hugging Face Space!")
