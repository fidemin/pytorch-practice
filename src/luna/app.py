import argparse
import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.core.app import App
from src.core.const import Mode
from src.luna.core.dto import AugmentInfo
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

        parser.add_argument("--batch-size", help="Batch size", type=int, default=32)
        parser.add_argument(
            "--num-epochs", help="Number of epochs", type=int, default=10
        )

        parser.add_argument(
            "--tensorboard-log-dir", help="Tensorboard log directory", type=str
        )
        parser.add_argument(
            "--training-data-limit",
            help="Limit of training data",
            type=int,
            default=None,
        )

        parser.add_argument(
            "--validation-data-limit",
            help="Limit of validation data",
            type=int,
            default=None,
        )

        parser.add_argument(
            "--negative-data-ratio",
            help="Negative data ratio",
            type=int,
            default=None,
        )

        parser.add_argument(
            "--num-workers",
            help="Number of workers for data loader",
            type=int,
            default=4,
        )

        # augmentation arguments
        parser.add_argument(
            "--augment-flip",
            help="Augment the training data by random flipping",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--augment-offset",
            help="Augment the training data with offset",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--augment-scale",
            help="Augment the training data with scaling",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--augment-rotate",
            help="Augment the training data by random rotation",
            action="store_true",
            default=False,
        )

        self.args = parser.parse_args(argv)

        self.use_cuda = torch.cuda.is_available()
        self.use_mps = False
        self.device = self._get_device()

        self.model = self._init_model()
        self.optimizer = self._init_optimizer()
        self._total_training_count = 0

        self._negative_data_ratio = self.args.negative_data_ratio

        # train metrics per epoch
        self._train_metrics_list = []
        self._valid_metrics_list = []

        # init tensorboard writer
        self._training_writer = None
        self._validation_writer = None
        self._init_tensorboard_writer()

        # init augmentation info
        self._augment_info = AugmentInfo(
            use_flip=self.args.augment_flip,
            use_offset=self.args.augment_offset,
            offset_factor=0.1,
            use_scale=self.args.augment_scale,
            scale_factor=0.2,
            use_rotate=self.args.augment_rotate,
        )

    def run(self):
        logger.info(f"Running with args: {self.args}")

        train_ds = LunaDataset(
            self.args.candidate_file_path,
            self.args.annotation_file_path,
            self.args.ct_files_dir,
            is_validation=False,
            validation_ratio=0.05,
            negative_data_ratio=self._negative_data_ratio,
        )

        valid_ds = LunaDataset(
            self.args.candidate_file_path,
            self.args.annotation_file_path,
            self.args.ct_files_dir,
            is_validation=True,
            validation_ratio=0.05,
            negative_data_ratio=self._negative_data_ratio,
        )

        logger.info(
            f"DataLoader preparing starts: train={len(train_ds)}, valid={len(valid_ds)}"
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

        logger.info(
            f"DataLoader preparing ends: train={len(train_ds)}, valid={len(valid_ds)}"
        )
        for epoch in range(1, self.args.num_epochs + 1):
            train_ds.shuffle_samples()

            self._train_epoch(epoch, train_dl, data_limit=self.args.training_data_limit)
            self._validate_epoch(
                epoch, valid_dl, data_limit=self.args.validation_data_limit
            )
            self._log_metrics(epoch, Mode.TRAINING, self._train_metrics_list[epoch - 1])
            self._log_metrics(
                epoch, Mode.VALIDATION, self._valid_metrics_list[epoch - 1]
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

    def _init_tensorboard_writer(self):
        tensorboard_log_dir = self.args.tensorboard_log_dir
        if not tensorboard_log_dir:
            logger.warning(
                "No tensorboard log directory specified. Tensorboard log will not be saved."
            )
            return
        log_path = os.path.join(
            tensorboard_log_dir, datetime.now().strftime("%Y%m%d%H%M%S")
        )
        self._training_writer = SummaryWriter(log_dir=log_path)
        self._validation_writer = SummaryWriter(log_dir=log_path)

    def _train_epoch(self, epoch, data_loader, data_limit=None):
        self.model.train()

        # save metrics for one epoch
        train_metrics = torch.zeros(
            METRICS_SIZE,  # kind of metrics
            len(data_loader.dataset),
            device=self.device,
        )

        processed_data_count = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            logger.info(f"Train starts: Epoch={epoch} Batch Index={batch_idx}")
            start_time = datetime.now()
            processed_data_count += len(data)

            # reset gradients
            self.optimizer.zero_grad()

            loss_var = self._compute_batch_loss(batch_idx, data, target, train_metrics)

            # backpropagation and optimization
            loss_var.backward()
            self.optimizer.step()

            duration = datetime.now() - start_time
            logger.info(
                f"Train ends: Epoch={epoch} Batch Index={batch_idx} Loss={loss_var.item()} duration={duration}"
            )

            if data_limit and processed_data_count >= data_limit:
                logger.info(f"Data count limit reached: {data_limit}")
                break

        self._train_metrics_list.append(train_metrics.to("cpu"))
        self._total_training_count += processed_data_count
        logger.info(
            f"Data count: processed_data={processed_data_count}, total_training_data={self._total_training_count}"
        )

    def _validate_epoch(self, epoch, data_loader, data_limit=None):
        self.model.eval()

        # save metrics for one epoch
        valid_metrics = torch.zeros(
            METRICS_SIZE,  # kind of metrics
            len(data_loader.dataset),
            device=self.device,
        )

        with torch.no_grad():
            processed_data_count = 0
            for batch_idx, (data, target) in enumerate(data_loader):
                loss_var = self._compute_batch_loss(
                    batch_idx, data, target, valid_metrics
                )

                logger.info(
                    f"Validation: Epoch={epoch} Batch Index={batch_idx} Loss={loss_var.item()}"
                )

                processed_data_count += len(data)

                if data_limit and processed_data_count >= data_limit:
                    logger.info(f"Data count limit reached: {data_limit}")
                    break

        self._valid_metrics_list.append(valid_metrics.to("cpu"))
        logger.info(f"Total data count: {processed_data_count}")

    def _compute_batch_loss(self, batch_idx, data, target, metrics):
        data = data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)

        output, probability = self.model(data)

        loss = nn.CrossEntropyLoss(reduction="none")(output, target)

        self._update_train_metrics(
            metrics,
            batch_idx,
            self.args.batch_size,
            target,
            probability,
            loss,
        )
        return loss.mean()

    def _update_train_metrics(
        self, metrics, batch_idx, batch_size, target, probability, loss_variable
    ):
        start_ndx = batch_idx * batch_size
        end_idx = start_ndx + batch_size

        metrics[METRICS_LABEL_IDX, start_ndx:end_idx] = target[:, 1].detach()
        metrics[METRICS_PREDICTION_IDX, start_ndx:end_idx] = probability[:, 1].detach()
        metrics[METRICS_LOSS_IDX, start_ndx:end_idx] = loss_variable.detach()

    def _log_metrics(self, epoch, mode: Mode, metrics_t, threshold=0.5):
        negative_label_mask = metrics_t[METRICS_LABEL_IDX] <= threshold
        negative_prediction_mask = metrics_t[METRICS_PREDICTION_IDX] <= threshold

        positive_label_mask = ~negative_label_mask
        positive_prediction_mask = ~negative_prediction_mask

        negative_count = int(negative_label_mask.sum())
        positive_count = int(positive_label_mask.sum())

        true_positive_count = int(
            (positive_label_mask & positive_prediction_mask).sum()
        )

        true_negative_count = int(
            (negative_label_mask & negative_prediction_mask).sum()
        )

        false_positive_count = positive_count - true_positive_count
        false_negative_count = negative_count - true_negative_count

        precision = true_positive_count / np.float32(
            true_positive_count + false_positive_count
        )
        recall = true_positive_count / np.float32(
            true_positive_count + false_negative_count
        )

        metrics_dict = {
            "loss/all": metrics_t[METRICS_LOSS_IDX].mean(),
            "loss/neg": metrics_t[METRICS_LOSS_IDX, negative_label_mask].mean(),
            "loss/pos": metrics_t[METRICS_LOSS_IDX, positive_label_mask].mean(),
            "correct/all": (true_negative_count + true_positive_count)
            / np.float32(metrics_t.shape[1]),
            "correct/neg": true_negative_count / np.float32(negative_count),
            "correct/pos": true_positive_count / np.float32(positive_count),
            "pr/precision": precision,
            "pr/recall": recall,
            "pr/f1_score": 2 * (precision * recall) / (precision + recall),
        }

        logger.info({**metrics_dict, "epoch": epoch, "mode": mode.value})

        if mode == Mode.TRAINING:
            writer = self._training_writer
        else:
            writer = self._validation_writer

        if writer:
            for key, value in metrics_dict.items():
                writer.add_scalar(key, value, self._total_training_count)
