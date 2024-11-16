import argparse
import logging
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.core.app import App
from src.luna.dataset import LunaDataset
from src.luna.model import LunaModel

logger = logging.getLogger(__name__)


class LunaTrainingApp(App):
    def __init__(self, *argv):
        if not argv:
            # get arguments from cli command
            argv = sys.argv[1:]

        parser = argparse.ArgumentParser()

        parser.add_argument("--candidate-file-path", help="Path to candidate file")
        parser.add_argument("--annotation-file-path", help="Path to annotation file")
        parser.add_argument("--ct-files-dir", help="Path to CT files directory")
        parser.add_argument("--batch-size", help="Batch size", type=int, default=128)

        parser.add_argument(
            "--num-workers",
            help="Number of workers for data loader",
            type=int,
            default=8,
        )

        self.args = parser.parse_args(argv)

        self.use_cuda = torch.cuda.is_available()
        self.use_mps = torch.backends.mps.is_available()
        self.device = self._get_device()

        self.model = self._init_model()
        self.optimizer = self._init_optimizer()

    def run(self):
        # TODO: implements this
        print(f"Running with args: {self.args}")

        train_ds = LunaDataset(
            self.args.candidate_file_path,
            self.args.annotation_file_path,
            self.args.ct_files_dir,
            is_validation=False,
            validation_ratio=0.9,
        )

        data_loader = DataLoader(
            train_ds,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.use_cuda or self.use_mps,
        )

    def _get_device(self):
        if self.use_cuda:
            logger.info("Using CUDA")
            return torch.device("cuda")
        elif self.use_mps:
            logger.info("Using MPS")
            return torch.device("mps")
        return torch.device("cpu")

    def _init_model(self) -> nn.Module:
        model = LunaModel()

        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        elif self.use_mps:
            model = model.to(self.device)

        return model

    def _init_optimizer(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.001)
