import mlflow
import xgboost as xgb
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5001")

# Get the wine quality dataset from the UCI repo
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, delimiter=";")

# Separate the features (X) and the target (y)
X = data.drop("quality", axis=1)
y = data["quality"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.
def get_or_create_experiment(experiment_name):
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)


experiment_id = get_or_create_experiment("XGBoost Example")

# Set the current active MLflow experiment
mlflow.set_experiment(experiment_id=experiment_id)
mlflow.xgboost.autolog()


# Define objective function and optimze with Optuna
def objective(trial):
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 1000,
        "verbosity": 0,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    predictions = model.predict(X_val)
    rmse = mean_squared_error(y_val, predictions, squared=False)
    return rmse


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=15)

# Fetching best parameters and the best RMSE score
best_params = study.best_params
best_rmse = study.best_value

# Optionally, you can retrain the model with the best parameters and log the model
model = xgb.XGBRegressor(**best_params)
model.fit(X_train, y_train, verbose=False)
mlflow.xgboost.log_model(model, "model")

# Get the logged model URI
model_uri = mlflow.get_artifact_uri("model")
print("Model URI:", model_uri)
