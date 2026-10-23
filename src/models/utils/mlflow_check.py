"""Step 1.5: check that MLflow can log a run to our local SQLite database.

Run from the repo root:
    conda run -n derm python -m src.utils.mlflow_check
Then view it with:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import tempfile
from pathlib import Path

import mlflow

from src import config


def main():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("setup-check")

    with mlflow.start_run(run_name="hello-mlflow") as run:
        mlflow.log_param("seed", config.SEED)          # a setting
        mlflow.log_metric("dummy_accuracy", 0.5)       # a result
        with tempfile.TemporaryDirectory() as tmp:     # a small file (artifact)
            note = Path(tmp) / "note.txt"
            note.write_text("MLflow is working.")
            mlflow.log_artifact(str(note))
        print("Logged run:", run.info.run_id)


if __name__ == "__main__":
    main()
