import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import SimpleVitNet
from torch.distributions.multivariate_normal import MultivariateNormal
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
from utils.loss import AngularPenaltySMLoss
import math
import copy
from models.Imp import HiddenImp
from math import sqrt
num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if 'adapter' not in args["convnet_type"]:
            raise NotImplementedError('Adapter requires Adapter backbone')
        self._network = SimpleVitNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]

        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args

        self._old_most_sentive = []
        self._update_grads = {}

        self.logit_norm = None
        self.tuned_epochs = None

        self.fisher_mat = None

        # IPR
        self.use_Imp = True
        self.HiddenImp = None
        self.util = None
        self.mean_feature_mag = None
        if self.use_Imp:
            self.HiddenImp = HiddenImp(net=nn.ModuleList([block.adaptmlp for block in self._network.convnet.blocks]),
                                 hidden_activation="relu", device=self._device)

        self.imp_network = True
        self.sim_channel = True

        # SDC
        self.SDC = True

    def after_task(self):
        self._known_classes = self._total_classes


    def extract_features(self, trainloader, model, args):
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.cuda()
                label = label.cuda()
                embedding = model.extract_vector(data)  
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())

        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        return embedding_list, label_list

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(data_manager.get_task_size(self._cur_task))  
        flag_old_model = 1 
        flag_mean = 1 
        self._network.adapt_feature(out_dim=self.feature_dim, task_id=self._cur_task)


        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
    
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
                                                 mode="train")

        self.train_dataset = train_dataset
        print("The number of training dataset:", len(self.train_dataset))

        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
        train_dataset_for_protonet = data_manager.get_dataset(np.arange(0, self._total_classes), source="train",
                                                              mode="test")


        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        
        old_model = None
        if self._cur_task >0:
            self._network.to(self._device)
            train_embeddings_old, _ = self.extract_features(self.train_loader, self._network, None) 
            if flag_mean:
                old_class_mean1 = self._class_means[:self._known_classes]
            else:
                old_class_mean1 = None
            if flag_old_model:
                old_model = copy.deepcopy(self._network)
        else:
            old_class_mean1 = None
                
        self._train(self.train_loader, self.test_loader, old_model, old_class_mean1=old_class_mean1,fisher=fisher, SDC=self.SDC)
        
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        if self._cur_task >0:
            train_embeddings_new, _ = self.extract_features(self.train_loader, self._network, None) 
            old_class_mean = self._class_means[:self._known_classes]  
            old_class_mean_copy=copy.deepcopy(old_class_mean)
            
            gap = self.displacement(train_embeddings_old, train_embeddings_new, old_class_mean, 4.0) 
            if self.args['ssca'] is True:
                old_class_mean += gap 
                self._class_means[:self._known_classes] = old_class_mean  
            

        self._network.fc.backup()
        self._compute_class_mean(data_manager, check_diff=False, oracle=False)
        task_size = data_manager.get_task_size(self._cur_task)

        if self._cur_task>0 and self.args['ca_epochs']>0 and self.args['ca'] is True:
            self._stage2_compact_classifier(task_size, self.args['ca_epochs'])
            if len(self._multiple_gpus) > 1:
                self._network = self._network.module

    def _train(self, train_loader, test_loader, old_model=None, old_class_mean1=None, fisher=False, SDC=False):
        self._network.to(self._device)
        w = 1
        if self._cur_task == 0:
            self.tuned_epochs = self.args["init_epochs"]
            param_groups = [
                {'params': self._network.convnet.blocks[-1].parameters(), 'lr': 0.01,
                 'weight_decay': self.args['weight_decay']},

                {'params': self._network.convnet.blocks[:-1].parameters(), 'lr': 0.01,
                 'weight_decay': self.args['weight_decay']},

                {'params': self._network.fc.parameters(), 'lr': 0.01, 'weight_decay': self.args['weight_decay']},
            ] 

            
            if self.args['optimizer'] == 'sgd':
                optimizer = optim.SGD(param_groups, momentum=0.9, lr=self.init_lr, weight_decay=w*self.weight_decay)

            elif self.args['optimizer'] == 'adam':
                optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=w*self.weight_decay)

            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.tuned_epochs,
                                                             eta_min=self.min_lr)
            self._init_train(train_loader, test_loader, optimizer, scheduler, fisher=fisher)
            
        else:
            self.tuned_epochs = self.args['inc_epochs']
            param_groups = []

            param_groups.append(
                {'params': self._network.convnet.parameters(), 'lr': 0.01, 'weight_decay': self.args['weight_decay']})
            param_groups.append(
                {'params': self._network.fc.parameters(), 'lr': 0.01, 'weight_decay': self.args['weight_decay']})

            if self.args['optimizer'] == 'sgd':
                optimizer = optim.SGD(param_groups, momentum=0.9, lr=self.init_lr, weight_decay=self.weight_decay)

            elif self.args['optimizer'] == 'adam':
                optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)

            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.tuned_epochs,
                                                             eta_min=self.min_lr)
            self._init_train(train_loader, test_loader, optimizer, scheduler, old_model=old_model, old_class_mean1=old_class_mean1, fisher=fisher,SDC=self.SDC)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler, old_model=None, old_class_mean1=None, fisher=False, SDC=False):

        prog_bar = tqdm(range(self.tuned_epochs))
        loss_cos=AngularPenaltySMLoss(loss_type='cosface', eps=1e-7, s=self.args["scale"], m=self.args["margin"])  
        loss_mse = nn.MSELoss()

        if old_class_mean1 is not None:
            old_class_mean1 = torch.from_numpy(old_class_mean1)
            old_class_mean1 = old_class_mean1.to(self._device)


        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            SDC_losses = 0.0
            IPR_losses = 0.0
            correct, total = 0, 0

            task_id = self._cur_task

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss=loss_cos(logits[:, self._known_classes:], targets - self._known_classes)

                if self.use_Imp:
                    with torch.no_grad():
                        feature_list = self._network.convnet.feature_list
                else:
                    feature_list = None

                if old_model is not None:
                    embedding = self._network.extract_vector(inputs)  
                    with torch.no_grad():
                        old_embedding = old_model.extract_vector(inputs) 
                    if old_class_mean1 is not None and self.SDC:
                        disten = self.displacement_w(old_embedding, embedding, old_class_mean1, 4.0)
                        zero_d =  torch.zeros_like(disten)
                        loss_SDC = loss_mse(disten, zero_d)
                    else:
                        loss_SDC = 0
                else:
                    loss_SDC = 0


                if self.imp_network and task_id>0 and old_model is not None:
                    bias = self._network.bias
                    adapters = nn.ModuleList([block.adaptmlp for block in self._network.convnet.blocks])
                    with torch.no_grad():
                        old_adapter = nn.ModuleList([block.adaptmlp for block in old_model.convnet.blocks])
                    if old_class_mean1 is not None:
                        cov_feature = self._class_covs.diagonal(dim1=-2, dim2=-1)
                        cov_feature = cov_feature[:self._known_classes].to(self._device)
                        sim = self.similarity(feature=embedding, old_feature=None, mean_feature=old_class_mean1,cov_feature=cov_feature, sim_channel=self.sim_channel)
                    else:
                        sim = None
                    net_matrix = self.imp_network_matrix(adapters=adapters)
                    loss_IPR = self.IPR_loss_adapters_with_similarity(adapters=adapters, IPR_matrix=net_matrix, old_adapter=old_adapter, similarity=sim, util=self.util,mean_feature_mag=self.mean_feature_mag)
                w_SDC = 1 
                if loss_IPR > 0.5:
                    loss_IPR/=1000
                if loss_IPR > 5:
                    loss_IPR/=10


                total_loss = loss + w_SDC*loss_SDC + loss_IPR
                
                optimizer.zero_grad()
                total_loss.backward()

                optimizer.step()  
                if self.use_Imp is True:   
                    self.HiddenImp.obain_imp(feature_list)
                    
                losses += loss.item()

                if old_model is not None and self.SDC:
                    SDC_losses += loss_SDC.item()
                else:
                    SDC_losses += loss_SDC

                if fisher and task_id>0 and self.fisher_mat is not None:
                    IPR_losses += loss_IPR.item()
                else:
                    IPR_losses += loss_IPR
                total_losses = losses + SDC_losses + IPR_losses
                
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Total_Loss {:.3f}, Loss {:.3f}, Distll_Loss {:.3f}, Fisher_Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.tuned_epochs,
                total_losses / len(train_loader),
                losses / len(train_loader),
                SDC_losses / len(train_loader),
                IPR_losses / len(train_loader),
                train_acc,
                test_acc,
            )

            prog_bar.set_description(info)
            logging.info(info)
        

        if self.use_Imp:
            self.util = self.HiddenImp.util
            self.mean_feature_mag = self.HiddenImp.mean_feature_mag
        logging.info(info)

    def compute_irr_ratio(self):
        block_len = self._update_grads[self._cur_task]
        finetune_block = []
        ratio_list = []
        for block in self._update_grads[self._cur_task].keys():
            ratio = self._update_grads[self._cur_task][block] / self._update_grads[self._cur_task - 1][block]
            ratio_list.append(ratio)
            if ratio >= 0.9 and ratio <= 1.1:
                finetune_block.append(block)

        print("ratio", ratio_list)
        return block

    def cnt_match_block(self, old_blocks, new_blocks):
        finetune_block = []
        for nb in new_blocks:
            is_match = False
            for ob in old_blocks:
                if nb == ob:
                    is_match = True
                    break
            if is_match is False:
                finetune_block.append(nb)
        return finetune_block

    def compute_sentive(self):
        # eval module
        self._network.eval()
        sentive_network = copy.deepcopy(self._network)
        param_groups = [
            {'params': sentive_network.convnet.parameters(), 'lr': 0.01, 'weight_decay': self.args['weight_decay']},
            {'params': sentive_network.fc.parameters(), 'lr': 0.01, 'weight_decay': self.args['weight_decay']}
        ]

        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(param_groups, momentum=0.9, lr=self.init_lr, weight_decay=self.weight_decay)

        update_magnitudes = {}
        for i, (_, inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            logits = sentive_network(inputs)["logits"]
            loss = F.cross_entropy(logits[:, self._known_classes:], targets - self._known_classes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for j, (name, param) in enumerate(sentive_network.named_parameters()):
                if "adapt" in name:
                    if name in update_magnitudes:
                        update_magnitudes[name] +=  (param.grad**2)
                    else:
                        update_magnitudes[name] =  (param.grad**2)
        grad_shapes = {}
        grad_shapes_int = {}
        for key in update_magnitudes.keys():
                grad_shapes[key] = update_magnitudes[key].shape
                grad_shapes_int[key] = np.cumprod(list(update_magnitudes[key].shape))[-1]
        large_tensor = torch.cat([update_magnitudes[key].flatten() for key in grad_shapes.keys()])
        _, indexes = large_tensor.topk(math.ceil(0.0001* large_tensor.shape[0]))
        print(indexes)

        tmp_large_tensor = torch.zeros_like(large_tensor, device='cuda')
        tmp_large_tensor[indexes] = 1.

        tmp_large_tensor_list = tmp_large_tensor.split([shape for shape in grad_shapes_int.values()])

        structured_param_num = 0
        structured_names = []
        tuned_vectors = []

        unstructured_param_num = 0
        unstructured_name_shapes = {}
        unstructured_name_shapes_int = {}
        unstructured_grad_mask = {}
        grad_sum_dict = {}
        for i, key in enumerate(grad_shapes.keys()):
            grad_sum = tmp_large_tensor_list[i].view(grad_shapes[key]).sum()
            grad_sum_dict[key] = grad_sum
            cur_param_num = grad_sum.item()

            unstructured_param_num += grad_sum.item()
            unstructured_name_shapes[key] = tmp_large_tensor_list[i].view(grad_shapes[key]).shape
            unstructured_name_shapes_int[key] = np.cumprod(list(update_magnitudes[key].shape))[-1]
            unstructured_grad_mask[key] = tmp_large_tensor_list[i].view(grad_shapes[key])

        return unstructured_grad_mask 

