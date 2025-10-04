# ------------------------------
# Model Training for Tourism Project
# ------------------------------

# Imports
import os
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ------------------------------
# Load Data from Hugging Face Dataset Hub
# ------------------------------
Xtrain_path = "hf://datasets/vamsikrishna1516/tourism_dataset_project/Xtrain.csv"
Xtest_path = "hf://datasets/vamsikrishna1516/tourism_dataset_project/Xtest.csv"
ytrain_path = "hf://datasets/vamsikrishna1516/tourism_dataset_project/ytrain.csv"
ytest_path = "hf://datasets/vamsikrishna1516/tourism_dataset_project/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()  # ensure Series
ytest = pd.read_csv(ytest_path).squeeze()

print("Data loaded successfully")
print("Xtrain shape:", Xtrain.shape)
print("ytrain distribution:\n", ytrain.value_counts(normalize=True))

# ------------------------------
# Define numeric & categorical features
# ------------------------------

numeric_features = [
    'Age',
    'NumberOfPersonVisiting',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'NumberOfChildrenVisiting',
    'MonthlyIncome',
    'PitchSatisfactionScore',
    'NumberOfFollowups',
    'DurationOfPitch'
]

categorical_features = [
    'TypeofContact',
    'CityTier',
    'Occupation',
    'Gender',
    'MaritalStatus',
    'Passport',
    'OwnCar',
    'Designation',
    'ProductPitched'
]

# ------------------------------
# Handle class imbalance (scale_pos_weight)
# ------------------------------
neg, pos = ytrain.value_counts().to_list()
scale_pos_weight = neg / pos if pos != 0 else 1
print(f"scale_pos_weight = {scale_pos_weight:.2f}")

# ------------------------------
# Preprocessing Pipeline
# ------------------------------
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# ------------------------------
# XGBoost Classifier + Grid Search
# ------------------------------
xgb_model = xgb.XGBClassifier(
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss"
)

param_grid = {
    'xgbclassifier__n_estimators': [50, 100],
    'xgbclassifier__max_depth': [3, 5, 7],
    'xgbclassifier__learning_rate': [0.05, 0.1],
    'xgbclassifier__subsample': [0.8, 1.0],
    'xgbclassifier__colsample_bytree': [0.8, 1.0],
}

pipeline = make_pipeline(preprocessor, xgb_model)

grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=3,
    scoring='recall',  # recall is important for identifying potential buyers
    n_jobs=-1
)

print("Starting grid search...")
grid_search.fit(Xtrain, ytrain)

best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# ------------------------------
# Evaluate the model
# ------------------------------
y_pred_train = best_model.predict(Xtrain)
y_pred_test = best_model.predict(Xtest)

print("\nTraining Classification Report:")
print(classification_report(ytrain, y_pred_train))

print("\nTest Classification Report:")
print(classification_report(ytest, y_pred_test))

# ------------------------------
# Save Best Model
# ------------------------------
model_path = "best_tourism_model_v1.joblib"
joblib.dump(best_model, model_path)
print(f"Model saved to {model_path}")

# ------------------------------
# Upload to Hugging Face Model Hub
# ------------------------------
repo_id = "vamsikrishna1516/tourism_model"   # use a dedicated model repo
repo_type = "model"

hf_token = os.getenv("HF_TOKEN")
api = HfApi(token=hf_token)

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model repo '{repo_id}' already exists. Uploading model...")
except RepositoryNotFoundError:
    print(f"Model repo '{repo_id}' not found. Creating new repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Repo '{repo_id}' created successfully.")

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=os.path.basename(model_path),
    repo_id=repo_id,
    repo_type=repo_type,
)

print("Model uploaded successfully to Hugging Face Hub!")
