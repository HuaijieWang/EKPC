import logging

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.inc_net import SimpleVitNet
from utils.loss import AngularPenaltySMLoss

from models.base import BaseLearner
from models.Imp import HiddenImp


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = SimpleVitNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"]
        self.min_lr = args["min_lr"]
        self.task_sizes = []
        self.global_importance = torch.zeros(self.feature_dim, device=self._device)
        adapters = nn.ModuleList(
            [block.adaptmlp for block in self._network.convnet.blocks]
        )
        self.HiddenImp = HiddenImp(adapters, self._device)
        self._importance = None

    def after_task(self):
        self._known_classes = self._total_classes

    def _loader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.args.get("num_workers", 8),
            pin_memory=self._device.type == "cuda",
        )

    @torch.no_grad()
    def extract_features(self, loader, model):
        model.eval()
        features = [
            model.extract_vector(inputs.to(self._device)).cpu()
            for _, inputs, _ in loader
        ]
        return torch.cat(features).numpy()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        task_size = data_manager.get_task_size(self._cur_task)
        self.task_sizes.append(task_size)
        self._total_classes = self._known_classes + task_size
        old_model = self._network.copy().freeze() if self._cur_task > 0 else None
        self._network.update_fc(task_size)
        self._network.to(self._device)
        logging.info("Learning on %s-%s", self._known_classes, self._total_classes)
        classes = np.arange(self._known_classes, self._total_classes)
        self.train_loader = self._loader(
            data_manager.get_dataset(classes, "train", "train"), True
        )
        self.test_loader = self._loader(
            data_manager.get_dataset(np.arange(self._total_classes), "test", "test")
        )
        drift_loader = self._loader(data_manager.get_dataset(classes, "train", "test"))
        old_features = (
            self.extract_features(drift_loader, old_model)
            if old_model is not None
            else None
        )
        self._train(self.train_loader, self.test_loader, old_model)
        if old_model is not None and self.args.get("ssca", True):
            new_features = self.extract_features(drift_loader, self._network)
            self._class_means[: self._known_classes] += self.displacement(
                old_features,
                new_features,
                self._class_means[: self._known_classes],
                self.args.get("drift_sigma", 4.0),
            )
        del old_model
        self._compute_class_mean(data_manager)
        self._update_importance(drift_loader)
        if (
            self._cur_task > 0
            and self.args.get("ca", True)
            and (self.args["ca_epochs"] > 0)
        ):
            self._stage2_compact_classifier(task_size, self.args["ca_epochs"])

    def _train(self, train_loader, test_loader, old_model=None):
        epochs = (
            self.args["init_epochs"] if self._cur_task == 0 else self.args["inc_epochs"]
        )
        parameters = [
            parameter
            for parameter in self._network.parameters()
            if parameter.requires_grad
        ]
        if self.args["optimizer"] == "sgd":
            optimizer = optim.SGD(
                parameters,
                lr=self.init_lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        elif self.args["optimizer"] == "adam":
            optimizer = optim.AdamW(
                parameters, lr=self.init_lr, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.args['optimizer']}")
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=self.min_lr
        )
        self._init_train(
            train_loader, test_loader, optimizer, scheduler, epochs, old_model
        )

    def _init_train(
        self, train_loader, test_loader, optimizer, scheduler, epochs, old_model=None
    ):
        criterion = AngularPenaltySMLoss(s=self.args["scale"], m=self.args["margin"])
        for epoch in tqdm(range(epochs)):
            self._network.train()
            losses = np.zeros(3)
            for _, inputs, targets in train_loader:
                inputs, targets = (inputs.to(self._device), targets.to(self._device))
                outputs = self._network(inputs)
                loss = criterion(
                    outputs["logits"][:, self._known_classes :],
                    targets - self._known_classes,
                )
                loss_IPR = loss.new_zeros(())
                loss_SDC = loss.new_zeros(())
                if old_model is not None:
                    if self.args.get("ipr", True):
                        loss_IPR = self.IPR_loss_adapters_with_similarity(old_model)
                    if self.args.get("tsdc", True):
                        with torch.no_grad():
                            old_features = old_model.extract_vector(inputs)
                        means = torch.as_tensor(
                            self._class_means[: self._known_classes],
                            dtype=outputs["features"].dtype,
                            device=self._device,
                        )
                        drift = self.displacement_w(
                            old_features,
                            outputs["features"],
                            means,
                            self.args.get("drift_sigma", 4.0),
                        )
                        loss_SDC = drift.square().sum()
                total = (
                    loss
                    + self.args.get("w1", 1.0) * loss_IPR
                    + self.args.get("w2", 1.0) * loss_SDC
                )
                optimizer.zero_grad()
                total.backward()
                optimizer.step()
                losses += [loss.item(), loss_IPR.item(), loss_SDC.item()]
            scheduler.step()
            accuracy = self._compute_accuracy(self._network, test_loader)
            logging.info(
                "Task %s, Epoch %s/%s: Cos %.4f, IPR %.4f, TSD %.4f, Test %.2f",
                self._cur_task,
                epoch + 1,
                epochs,
                *losses / len(train_loader),
                accuracy,
            )

    @torch.no_grad()
    def _update_importance(self, loader):
        self._network.eval()
        self.HiddenImp.begin_task()
        for _, inputs, _ in loader:
            self._network.extract_vector(inputs.to(self._device))
            self.HiddenImp.obain_imp(self._network.convnet.feature_list)
        self.HiddenImp.finalize_task()
        means = torch.as_tensor(
            self._class_means[self._known_classes :], device=self._device
        )
        variances = (
            self._class_covs[self._known_classes :]
            .diagonal(dim1=-2, dim2=-1)
            .to(self._device)
        )
        self.global_importance += (
            (means / variances.clamp_min(1e-08)).mean(dim=0).float()
        )
        self._importance = self.imp_network_matrix()

    @torch.no_grad()
    def imp_network_matrix(self):
        global_values = self.global_importance.abs()
        return [
            {
                "down_W": self.args.get("eta2", 1.0) * torch.outer(down, global_values),
                "up_W": self.args.get("eta1", 100.0) * torch.outer(global_values, up),
            }
            for down, up in zip(self.HiddenImp.mean_feature_mag, self.HiddenImp.util)
        ]

    def IPR_loss_adapters_with_similarity(self, old_model):
        loss = self.global_importance.new_zeros(())
        for block, old_block, importance in zip(
            self._network.convnet.blocks, old_model.convnet.blocks, self._importance
        ):
            loss = (
                loss
                + (
                    importance["down_W"]
                    * (
                        block.adaptmlp.down_proj.weight
                        - old_block.adaptmlp.down_proj.weight.detach()
                    ).square()
                ).sum()
            )
            loss = (
                loss
                + (
                    importance["up_W"]
                    * (
                        block.adaptmlp.up_proj.weight
                        - old_block.adaptmlp.up_proj.weight.detach()
                    ).square()
                ).sum()
            )
        return loss
