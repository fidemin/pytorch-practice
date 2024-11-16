import argparse
import sys

from src.core.app import App


class LunaTrainingApp(App):
    def __init__(self, *argv):
        if not argv:
            # get arguments from cli command
            argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--num-workers",
            help="Number of workers for data loader",
            type=int,
            default=8,
        )

        self.args = parser.parse_args(argv)

    def run(self):
        # TODO: implements this
        print(f"Running with args: {self.args}")
