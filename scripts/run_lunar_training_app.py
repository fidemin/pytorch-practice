from os import environ

from src import run

if __name__ == "__main__":

    run(
        "src.luna.app.LunaTrainingApp",
        "--candidate-file-path",
        environ.get("CANDIDATE_FILE_PATH"),
        "--annotation-file-path",
        environ.get("ANNOTATION_FILE_PATH"),
        "--ct-files-dir",
        environ.get("CT_FILES_DIR"),
        "--num-workers",
        "4",
    )
