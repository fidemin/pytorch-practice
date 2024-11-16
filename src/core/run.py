import importlib
import logging
import time

from src.core.app import App

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def import_from_str(import_str):
    module_name, attr_name = import_str.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def run(app_str: str, *argv):

    logger.info(f"Running app {app_str} with args: {argv}")

    app: App = import_from_str(app_str)(*argv)

    start_time = time.time()
    app.run()
    duration = time.time() - start_time

    logger.info(f"Finished running app {app_str} with duration {duration:.2f} seconds")
