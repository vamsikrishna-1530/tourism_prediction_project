# ------------------------------
# Data Preparation for Tourism Project
# ------------------------------

# Standard imports
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi

# ------------------------------
# Load dataset from Hugging Face
# ------------------------------
hf_token = os.getenv("HF_TOKEN")
api = HfApi(token=hf_token)

DATASET_PATH = "hf://datasets/vamsikrishna1516/tourism_dataset_project/tourism.csv"
print("Loading dataset from:", DATASET_PATH)

df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully — shape:", df.shape)
print("Columns:", df.columns.tolist())

# ------------------------------
# Drop unique identifier
# ------------------------------
if "CustomerID" in df.columns:
    df.drop(columns=["CustomerID"], inplace=True)
    print("🧹 Dropped unique identifier: CustomerID")

# ------------------------------
# Handle missing values (basic)
# ------------------------------
# Fill numeric NaNs with median, categorical with 'Unknown'
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna("Unknown")
    else:
        df[col] = df[col].fillna(df[col].median())

# ------------------------------
# Encode categorical columns
# ------------------------------
label_cols = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "MaritalStatus",
    "Designation",
    "ProductPitched"
]

label_encoder = LabelEncoder()

for col in label_cols:
    if col in df.columns:
        df[col] = label_encoder.fit_transform(df[col].astype(str))
        print(f"🔤 Encoded column: {col}")

# Convert CityTier (if text) to numeric
if "CityTier" in df.columns and df["CityTier"].dtype == "object":
    df["CityTier"] = df["CityTier"].replace(
        {"Tier 1": 1, "Tier 2": 2, "Tier 3": 3}
    )
    print("Converted CityTier to numeric levels")

# Ensure binary fields are integers (Passport, OwnCar)
for binary_col in ["Passport", "OwnCar"]:
    if binary_col in df.columns:
        df[binary_col] = pd.to_numeric(df[binary_col], errors="coerce").fillna(0).astype(int)

# ------------------------------
# Split dataset into features & target
# ------------------------------
target_col = "ProdTaken"

if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset!")

X = df.drop(columns=[target_col])
y = df[target_col]

print("Split complete: X shape =", X.shape, "y shape =", y.shape)

# ------------------------------
# Train-test split
# ------------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
)

# ------------------------------
# Save to CSVs
# ------------------------------
os.makedirs("tourism_project/data", exist_ok=True)

Xtrain_path = "tourism_project/data/Xtrain.csv"
Xtest_path = "tourism_project/data/Xtest.csv"
ytrain_path = "tourism_project/data/ytrain.csv"
ytest_path = "tourism_project/data/ytest.csv"

Xtrain.to_csv(Xtrain_path, index=False)
Xtest.to_csv(Xtest_path, index=False)
ytrain.to_csv(ytrain_path, index=False)
ytest.to_csv(ytest_path, index=False)

print("Saved train/test splits:")
print("  -", Xtrain_path)
print("  -", Xtest_path)
print("  -", ytrain_path)
print("  -", ytest_path)

# ------------------------------
# Upload to Hugging Face Dataset Repo
# ------------------------------
repo_id = "vamsikrishna1516/tourism_dataset_project"
repo_type = "dataset"

try:
    print(f"Uploading data splits to Hugging Face dataset repo: {repo_id}")
    for file_path in [Xtrain_path, Xtest_path, ytrain_path, ytest_path]:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=os.path.basename(file_path),
            repo_id=repo_id,
            repo_type=repo_type,
        )
    print("All files uploaded successfully to Hugging Face!")
except Exception as e:
    print("Upload failed:", e)
