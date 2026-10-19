import logging

import numpy as np
import torch
from torch import optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.toolkit import accuracy, tensor2numpy

batch_size = 64


class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self.topk = 5
        self._device = args["device"][0]

    @property
    def feature_dim(self):
        return self._network.feature_dim

    def _stage2_compact_classifier(self, task_size, ca_epochs=5):
        for p in self._network.fc.parameters():
            p.requires_grad = True
        run_epochs = ca_epochs
        crct_num = self._total_classes
        param_list = [p for p in self._network.fc.parameters() if p.requires_grad]
        network_params = [
            {
                "params": param_list,
                "lr": self.init_lr,
                "weight_decay": self.weight_decay,
            }
        ]
        optimizer = optim.SGD(
            network_params,
            lr=self.init_lr,
            momentum=0.9,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=run_epochs
        )
        self._network.to(self._device)
        self._network.eval()
        for epoch in range(run_epochs):
            losses = 0.0
            sampled_data = []
            sampled_label = []
            num_sampled_pcls = self.args.get("samples_per_class", 256)
            for c_id in range(crct_num):
                cls_mean = torch.as_tensor(
                    self._class_means[c_id], dtype=torch.float32, device=self._device
                )
                cls_cov = self._class_covs[c_id].to(self._device)
                m = MultivariateNormal(cls_mean.float(), cls_cov.float())
                sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                sampled_data.append(sampled_data_single)
                sampled_label.extend([c_id] * num_sampled_pcls)
            sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
            sampled_label = torch.tensor(sampled_label).long().to(self._device)
            inputs = sampled_data
            targets = sampled_label
            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]
            for _iter in range(crct_num):
                inp = inputs[_iter * num_sampled_pcls : (_iter + 1) * num_sampled_pcls]
                tgt = targets[_iter * num_sampled_pcls : (_iter + 1) * num_sampled_pcls]
                outputs = self._network.ca_forward(inp)
                logits = self.args["scale"] * outputs["logits"]
                loss = F.cross_entropy(logits[:, :crct_num], tgt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
            scheduler.step()
            test_acc = self._compute_accuracy(self._network, self.test_loader)
            info = "CA Task {} => Loss {:.3f}, Test_accy {:.3f}".format(
                self._cur_task, losses / self._total_classes, test_acc
            )
            logging.info(info)

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes, self.task_sizes)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (y_pred.shape[1], 1))).sum()
            * 100
            / len(y_true),
            decimals=2,
        )
        return ret

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)
        return cnn_accy

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = (0, 0)
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = ([], [])
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(
                outputs,
                k=min(self.topk, self._total_classes),
                dim=1,
                largest=True,
                sorted=True,
            )[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return (np.concatenate(y_pred), np.concatenate(y_true))

    @torch.no_grad()
    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = ([], [])
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            _vectors = tensor2numpy(
                self._network.extract_vector(_inputs.to(self._device))
            )
            vectors.append(_vectors)
            targets.append(_targets)
        return (np.concatenate(vectors), np.concatenate(targets))

    def _compute_class_mean(self, data_manager):
        if hasattr(self, "_class_means") and self._class_means is not None:
            ori_classes = self._class_means.shape[0]
            assert ori_classes == self._known_classes
            new_class_means = np.zeros((self._total_classes, self.feature_dim))
            new_class_means[: self._known_classes] = self._class_means
            self._class_means = new_class_means
            new_class_cov = torch.zeros(
                (self._total_classes, self.feature_dim, self.feature_dim)
            )
            new_class_cov[: self._known_classes] = self._class_covs
            self._class_covs = new_class_cov
        else:
            self._class_means = np.zeros((self._total_classes, self.feature_dim))
            self._class_covs = torch.zeros(
                (self._total_classes, self.feature_dim, self.feature_dim)
            )
        for class_idx in range(self._known_classes, self._total_classes):
            idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
            )
            idx_loader = DataLoader(
                idx_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.args.get("num_workers", 8),
            )
            vectors, _ = self._extract_vectors(idx_loader)
            class_mean = np.mean(vectors, axis=0)
            feature_tensor = torch.as_tensor(vectors, dtype=torch.float64)
            if len(feature_tensor) > 1:
                class_cov = torch.cov(feature_tensor.T)
            else:
                class_cov = torch.zeros(
                    self.feature_dim, self.feature_dim, dtype=torch.float64
                )
            class_cov += torch.eye(self.feature_dim, dtype=torch.float64) * 0.001
            self._class_means[class_idx, :] = class_mean
            self._class_covs[class_idx, ...] = class_cov

    def displacement_w(self, old_features, new_features, means, sigma):
        if sigma <= 0:
            raise ValueError("drift_sigma must be positive")
        with torch.no_grad():
            squared_distance = (
                means.square().sum(dim=1, keepdim=True)
                + old_features.square().sum(dim=1).unsqueeze(0)
                - 2 * means @ old_features.T
            ).clamp_min(0)
            weights = torch.softmax(-squared_distance / (2 * sigma**2), dim=1)
        return weights @ (new_features - old_features)

    def displacement(self, old_features, new_features, means, sigma):
        old_tensor = torch.as_tensor(old_features, dtype=torch.float64)
        new_tensor = torch.as_tensor(new_features, dtype=torch.float64)
        mean_tensor = torch.as_tensor(means, dtype=torch.float64)
        return self.displacement_w(old_tensor, new_tensor, mean_tensor, sigma).numpy()
