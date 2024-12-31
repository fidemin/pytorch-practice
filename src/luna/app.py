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


METRICS_LABEL_IDX = 0
METRICS_PREDICTION_IDX = 1
METRICS_LOSS_IDX = 2
METRICS_SIZE = 3


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
            "--num-epochs", help="Number of epochs", type=int, default=10
        )

        parser.add_argument(
            "--num-workers",
            help="Number of workers for data loader",
            type=int,
            default=8,
        )

        self.args = parser.parse_args(argv)

        self.use_cuda = torch.cuda.is_available()
        self.use_mps = False
        self.device = self._get_device()

        self.model = self._init_model()
        self.optimizer = self._init_optimizer()

        # train metrics per epoch
        self._train_metrics_list = []
        self._valid_metrics_list = []

    def run(self):
        print(f"Running with args: {self.args}")

        train_ds = LunaDataset(
            self.args.candidate_file_path,
            self.args.annotation_file_path,
            self.args.ct_files_dir,
            is_validation=False,
            validation_ratio=0.01,
        )

        valid_ds = LunaDataset(
            self.args.candidate_file_path,
            self.args.annotation_file_path,
            self.args.ct_files_dir,
            is_validation=True,
            validation_ratio=0.01,
        )

        train_dl = DataLoader(
            train_ds,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.use_cuda or self.use_mps,
        )

        valid_dl = DataLoader(
            valid_ds,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.use_cuda or self.use_mps,
        )

        for epoch in range(1, self.args.num_epochs + 1):
            self._train_epoch(epoch, train_dl)
            self._validate_epoch(epoch, valid_dl)

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

    def _train_epoch(self, epoch, data_loader):
        self.model.train()

        # save metrics for one epoch
        train_metrics = torch.zeros(
            METRICS_SIZE,  # kind of metrics
            len(data_loader.dataset),
            device=self.device,
        )

        for batch_idx, (data, target) in enumerate(data_loader):
            # reset gradients
            self.optimizer.zero_grad()

            loss_var = self._compute_batch_loss(batch_idx, data, target, train_metrics)

            # backpropagation and optimization
            loss_var.backward()
            self.optimizer.step()

            logger.info(
                f"Train: Epoch={epoch} Batch Index={batch_idx} Loss={loss_var.item()}"
            )

        self._train_metrics_list.append(train_metrics.to("cpu"))

    def _validate_epoch(self, epoch, data_loader):
        self.model.eval()

        # save metrics for one epoch
        valid_metrics = torch.zeros(
            METRICS_SIZE,  # kind of metrics
            len(data_loader.dataset),
            device=self.device,
        )

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                loss_var = self._compute_batch_loss(
                    batch_idx, data, target, valid_metrics
                )

                logger.info(
                    f"Validation: Epoch={epoch} Batch Index={batch_idx} Loss={loss_var.item()}"
                )

        self._valid_metrics_list.append(valid_metrics.to("cpu"))

    def _compute_batch_loss(self, batch_idx, data, target, train_metrics):
        data = data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)

        output, probability = self.model(data)

        loss = nn.CrossEntropyLoss(reduction="none")(output, target)

        self._update_train_metrics(
            train_metrics,
            batch_idx,
            self.args.batch_size,
            target,
            probability,
            loss,
        )
        return loss.mean()

    def _update_train_metrics(
        self, train_metrics, batch_idx, batch_size, target, probability, loss_variable
    ):
        start_ndx = batch_idx * batch_size
        end_idx = start_ndx + batch_size

        train_metrics[METRICS_LABEL_IDX, start_ndx:end_idx] = target[:, 1].detach()
        train_metrics[METRICS_PREDICTION_IDX, start_ndx:end_idx] = probability[
            :, 1
        ].detach()
        train_metrics[METRICS_LOSS_IDX, start_ndx:end_idx] = loss_variable.detach()
