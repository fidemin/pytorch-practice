from os import environ

from src import run

if __name__ == "__main__":
    args = [
        "src.luna.app.LunaTrainingApp",
        "--candidate-file-path",
        environ.get("CANDIDATE_FILE_PATH"),
        "--annotation-file-path",
        environ.get("ANNOTATION_FILE_PATH"),
        "--ct-files-dir",
        environ.get("CT_FILES_DIR"),
    ]

    training_data_limit = environ.get("TRAINING_DATA_LIMIT")
    if training_data_limit:
        args.extend(["--training-data-limit", training_data_limit])

    validation_data_limit = environ.get("VALIDATION_DATA_LIMIT")
    if validation_data_limit:
        args.extend(["--validation-data-limit", validation_data_limit])

    number_of_workers = environ.get("NUM_WORKERS")
    if number_of_workers:
        args.extend(["--num-workers", number_of_workers])

    number_of_epochs = environ.get("NUM_EPOCHS")
    if number_of_epochs:
        args.extend(["--num-epochs", number_of_epochs])

    tensorboard_log_dir = environ.get("TENSORBOARD_LOG_DIR")
    if tensorboard_log_dir:
        args.extend(["--tensorboard-log-dir", tensorboard_log_dir])

    run(*args)
