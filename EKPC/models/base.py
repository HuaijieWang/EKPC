import copy
import logging
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist
import time

EPSILON = 1e-8
batch_size = 64


class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 5  
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(
            self._targets_memory
        ), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def _stage2_compact_classifier(self, task_size, ca_epochs=5):
        for p in self._network.fc.parameters():
            p.requires_grad = True

        run_epochs = ca_epochs
        crct_num = self._total_classes
        param_list = [p for p in self._network.fc.parameters() if p.requires_grad]
        network_params = [{'params': param_list, 'lr': self.init_lr,
                           'weight_decay': self.weight_decay}]

        optimizer = optim.SGD(network_params, lr=self.init_lr, momentum=0.9, weight_decay=self.weight_decay)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)

        self._network.to(self._device)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._network.eval()

        for epoch in range(run_epochs):
            losses = 0.
            sampled_data = []
            sampled_label = []
            num_sampled_pcls = 256

            for c_id in range(crct_num):
                t_id = c_id // task_size
                decay = (t_id + 1) / (self._cur_task + 1) * 0.1
                cls_mean = torch.tensor(self._class_means[c_id], dtype=torch.float64).to(self._device) * (
                            0.9 + decay) 

                cls_cov = (self._class_covs[c_id]).to(self._device)
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
                inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]

                outputs = self._network.ca_forward(inp)
                logits = self.args['scale'] * outputs['logits']

                if self.logit_norm is not None:
                    per_task_norm = []
                    prev_t_size = 0
                    cur_t_size = 0
                    for _ti in range(self._cur_task + 1):
                        cur_t_size += self.task_sizes[_ti]
                        temp_norm = torch.norm(logits[:, prev_t_size:cur_t_size], p=2, dim=-1, keepdim=True) + 1e-7
                        per_task_norm.append(temp_norm)
                        prev_t_size += self.task_sizes[_ti]

                    per_task_norm = torch.cat(per_task_norm, dim=-1)
                    norms = per_task_norm.mean(dim=-1, keepdim=True)
                    norms_all = torch.norm(logits[:, :crct_num], p=2, dim=-1, keepdim=True) + 1e-7
                    decoupled_logits = torch.div(logits[:, :crct_num], norms) / self.logit_norm
                    loss = F.cross_entropy(decoupled_logits, tgt)
                else:
                    loss = F.cross_entropy(logits[:, :crct_num], tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            test_acc = self._compute_accuracy(self._network, self.test_loader)
            info = 'CA Task {} => Loss {:.3f}, Test_accy {:.3f}'.format(
                self._cur_task, losses / self._total_classes, test_acc)
            logging.info(info)


    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        return ret

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)
        return cnn_accy

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
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
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  


    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._network.module.extract_vector(_inputs.to(self._device))
                )
            else:
                _vectors = tensor2numpy(
                    self._network.extract_vector(_inputs.to(self._device))
                )

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _compute_class_mean(self, data_manager, check_diff=False, oracle=False):
        if hasattr(self, '_class_means') and self._class_means is not None and not check_diff:
            ori_classes = self._class_means.shape[0]
            assert ori_classes == self._known_classes
            new_class_means = np.zeros((self._total_classes, self.feature_dim))
            new_class_means[:self._known_classes] = self._class_means
            self._class_means = new_class_means
            new_class_cov = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))
            new_class_cov[:self._known_classes] = self._class_covs
            self._class_covs = new_class_cov
        elif not check_diff:
            self._class_means = np.zeros((self._total_classes, self.feature_dim))
            self._class_covs = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))

        radius = []
        for class_idx in range(self._known_classes, self._total_classes):

            data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx + 1), source='train',
                                                                  mode='test', ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)

            class_mean = np.mean(vectors, axis=0)
            if self._cur_task == 0:
                cov = np.cov(vectors.T)+ np.eye(class_mean.shape[-1]) * 1e-4
                radius.append(np.trace(cov) /768)
            class_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T) + torch.eye(class_mean.shape[-1]) * 1e-3

            self._class_means[class_idx, :] = class_mean
            self._class_covs[class_idx, ...] = class_cov

        if self._cur_task == 0:
                self.radius = np.sqrt(np.mean(radius))
                print(self.radius)

    def displacement_cov(self, Y, class_mean, embedding_old, sigma):
        cov = None
        start_time = time.time()
        for _class in range(self._known_classes):
            loop_start_time = time.time()
            DY = self.cov_computation(Y, class_mean[_class])
            distance = np.sum((np.tile(Y[None, :, :], [1, 1, 1]) - np.tile(
                embedding_old[_class, None, :], [1, Y.shape[0], 1])) ** 2, axis=2)
            W = np.exp(-distance / (2 * sigma ** 2)) + 1e-5
            W_norm = W / np.tile(np.sum(W, axis=1)[:, None], [1, W.shape[1]])
            if cov is None:
                cov = np.sum(np.tile(W_norm[:, :, None, None], [
                    1, 1, DY.shape[1], DY.shape[2]]) * np.tile(DY[None, :, :, :], [W.shape[0], 1, 1, 1]), axis=1)
            else:
                displacement = np.sum(np.tile(W_norm[:, :, None, None], [
                    1, 1, DY.shape[1], DY.shape[2]]) * np.tile(DY[None, :, :, :], [W.shape[0], 1, 1, 1]), axis=1)
                cov = np.concatenate((cov, displacement))
            loop_end_time = time.time()
            print("single loop time: ", loop_end_time - loop_start_time)
        end_time = time.time()
        print("total loop time: ", end_time - start_time)

        cov = torch.tensor(cov)
        return cov
    



    def compute_hessian_vector_product(self, grad_params, model, vector):
        hessian_vector_product = torch.autograd.grad(
            outputs=grad_params,
            inputs=model,
            grad_outputs=vector
        )
        return hessian_vector_product

    
    def imp_network_matrix(self, adapters):
        imp_matrix = []
        with torch.no_grad():
            for i, adapter in enumerate(adapters):
                if len(imp_matrix) < i + 1:
                    imp_matrix.append({})
                    imp_matrix[i]['down_W'] = torch.ones_like(adapter.down_proj.weight)
                    imp_matrix[i]['down_b'] = torch.ones_like(adapter.down_proj.bias)
                    imp_matrix[i]['up_W'] = torch.ones_like(adapter.up_proj.weight)
                    imp_matrix[i]['up_b'] = torch.ones_like(adapter.up_proj.bias)
        return imp_matrix
        
    def swap_softmax_values(self, softmax_sim):
        sorted_indices = torch.argsort(softmax_sim)
        sorted_values = torch.sort(softmax_sim)[0] 
        reversed_sim = torch.zeros_like(softmax_sim)
        for i in range(sorted_indices.size(0)):
            reversed_sim[sorted_indices[i]] = sorted_values[sorted_indices.size(0) - i - 1]
        return reversed_sim
        

    def similarity(self, feature=None, old_feature=None, mean_feature=None, cov_feature=None, sim_channel=False):
        with torch.no_grad():
            feature = F.normalize(feature, p=2, dim=1)  
            mean_feature = F.normalize(mean_feature, p=2, dim=1) 
            k, _ = mean_feature.shape
            mean_feature = mean_feature.to(torch.float32)

            if sim_channel:
                T = 2
                if old_feature is not None:
                    old_feature = F.normalize(old_feature, p=2, dim=1)
                    tmp_feature_sim = feature * old_feature
                    feature_sim = tmp_feature_sim.mean(dim=0)
                    softmax_sim = F.softmax(feature_sim/T, dim=0)
                    softmax_sim = self.swap_softmax_values(softmax_sim)
                else:
                    abs_mean_feature = abs(mean_feature)
                    if cov_feature is not None:
                        abs_cov_feature = abs(cov_feature)
                        tmp_feature_sim = (abs_mean_feature/abs_cov_feature).mean(dim=0)
                    else:
                        tmp_feature_sim = abs_mean_feature.mean(dim=0)
                    softmax_sim = F.softmax(tmp_feature_sim, dim=0)
                    softmax_sim = self.swap_softmax_values(softmax_sim)
                return softmax_sim
            else:
                cos_similarity = torch.matmul(feature, mean_feature.T) 

                topk_cos_similarity, _ = torch.topk(cos_similarity, min(int(k*0.1),5), dim=-1, largest=True, sorted=False)

                topk_cos_similarity = topk_cos_similarity.mean()


                topk_cos_similarity = (topk_cos_similarity + 1) / 2
                
                return topk_cos_similarity


    def IPR_loss_adapters_with_similarity(self, adapters, IPR_matrix, old_adapter, similarity=None, util=None,mean_feature_mag=None):
        
        loss = 0
        if util is not None and mean_feature_mag is not None:
            for i in range(len(util)):
                for j in range(len(util[i])):
                    IPR_matrix[i]['down_W'][j,:] *= (mean_feature_mag[i][j] * similarity)
                    IPR_matrix[i]['down_b'][j] *= mean_feature_mag[i][j] * similarity[j]
                    IPR_matrix[i]['up_W'][:,j] *= (util[i][j]*mean_feature_mag[i][j] * similarity)
                    IPR_matrix[i]['up_b'][j] *= util[i][j]*mean_feature_mag[i][j] * similarity[j]

            w = 1
            for idx, adapter in enumerate(adapters):   
                loss += 1*w*(IPR_matrix[idx]['down_W'] * (adapter.down_proj.weight - old_adapter[idx].down_proj.weight.data) ** 2).sum()
                loss += 1*w*(IPR_matrix[idx]['down_b'] * (adapter.down_proj.bias - old_adapter[idx].down_proj.bias.data) ** 2).sum()
                loss += 1e2*w*(IPR_matrix[idx]['up_W'] * (adapter.up_proj.weight- old_adapter[idx].up_proj.weight.data) ** 2).sum() 
                loss += 1e2*w*(IPR_matrix[idx]['up_b'] * (adapter.up_proj.bias - old_adapter[idx].up_proj.bias.data) ** 2).sum()   
        return loss

    def displacement_w(self, Y1, Y2, embedding_old, sigma):
        with torch.no_grad():
            distance = torch.sum((Y1.unsqueeze(0) - embedding_old.unsqueeze(1)) ** 2, dim=2)
            W = torch.exp(-distance / (2 * sigma ** 2)) + 1e-5  
            W_sum = torch.sum(W, dim=1, keepdim=True)
            W_norm = W / W_sum  
            W_norm_expanded = W_norm.unsqueeze(2)  
        DY = Y2 - Y1 
        DY_expanded = DY.unsqueeze(0)  
        displacement = torch.sum(W_norm_expanded * DY_expanded, dim=1) 
        return displacement



    def displacement(self, Y1, Y2, embedding_old, sigma):
        DY = Y2 - Y1  
        distance = np.sum((np.tile(Y1[None, :, :], [embedding_old.shape[0], 1, 1]) - np.tile(
            embedding_old[:, None, :], [1, Y1.shape[0], 1])) ** 2, axis=2) 
        W = np.exp(-distance / (2 * sigma ** 2)) + 1e-5 
        W_norm = W / np.tile(np.sum(W, axis=1)[:, None], [1, W.shape[1]])  
        displacement = np.sum(np.tile(W_norm[:, :, None], [
            1, 1, DY.shape[1]]) * np.tile(DY[None, :, :], [W.shape[0], 1, 1]), axis=1)
        return displacement

