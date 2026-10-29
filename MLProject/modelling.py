"""Training model untuk MLflow Project (dipakai oleh Workflow CI)."""
import argparse
import os
import shutil
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def main(data_path: str, n_estimators: int, max_depth: int) -> None:
    df = pd.read_csv(data_path)
    X = df.drop(columns="Churn")
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Saat dijalankan lewat `mlflow run`, run sudah aktif otomatis
    with mlflow.start_run() as run:
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("test_precision", precision_score(y_test, y_pred))
        mlflow.log_metric("test_recall", recall_score(y_test, y_pred))
        mlflow.log_metric("test_f1", f1_score(y_test, y_pred))
        mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, y_proba))

        tmp = tempfile.mkdtemp()

        # Model disimpan sebagai artefak di folder "model"
        model_dir = os.path.join(tmp, "model")
        mlflow.sklearn.save_model(model, model_dir, serialization_format="cloudpickle")
        mlflow.log_artifacts(model_dir, artifact_path="model")

        # Confusion matrix
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
        p = os.path.join(tmp, "test_confusion_matrix.png")
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(p)

        shutil.rmtree(tmp, ignore_errors=True)
        print(f"Run ID: {run.info.run_id}")
        print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="telco_churn_preprocessing/telco_churn_preprocessed.csv")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()
    main(args.data_path, args.n_estimators, args.max_depth)
