# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import os

import cv2
import torch
from torch import distributed as dist
from collections import OrderedDict
from tqdm import tqdm
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.nn import functional as F
from basicsr.models.base_model import BaseModel
from basicsr.archs import build_network
from basicsr.utils import get_root_logger, tensor2img
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils.logger import AverageMeter
from basicsr.archs.RAFT.raft import RAFT
from basicsr.archs.RAFT.utils.utils import InputPadder
import math


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


@MODEL_REGISTRY.register()
class ImageRestorationModel1(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageRestorationModel1, self).__init__(opt)

        self.net_g = build_network(opt["network_g"])
        self.net_g = self.model10_to_device(self.net_g)

        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True),
                              param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()
        self.scaler = torch.cuda.amp.GradScaler()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.log_dict = OrderedDict()
        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
            self.log_dict['l_pix'] = AverageMeter()
            self.log_dict['l_pix_object'] = AverageMeter()

        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            # to do
            pass
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        self.setup_optimizers()
        self.setup_schedulers()

    def model10_to_device(self, net):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
            net (nn.Module)
        """

        net = net.to(self.device)
        if self.opt['dist']:
            net = DistributedDataParallel(
                net,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=False
            )
            net._set_static_graph()

        elif self.opt['num_gpu'] > 1:
            net = DataParallel(net)
        return net

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        optim_params_lowlr = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                if k.startswith('module.spynet') or k.startswith('module.dcns'):
                    optim_params_lowlr.append(v)
                    print("lower lr", k)
                else:
                    optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        # print(optim_params)
        ratio = 0.1

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.AdamW(
                [{'params': optim_params}, {'params': optim_params_lowlr, 'lr': train_opt['optim_g']['lr'] * ratio}],
                **train_opt['optim_g'])
        # elif optim_type == 'SGD':
        #     self.optimizer_g = torch.optim.SGD(optim_params,
        #                                        **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)
        # print(self.optimizer_g)
        # exit(0)

    def feed_data(self, data):
        lq, gt, pm, hm, fw, bw = data['lq'], data['gt'], data['pm'], data['hm'], data['fw'], data['bw']
        self.lq = lq.to(self.device)
        self.gt = gt[:, 1:-1, ...].to(self.device)
        self.pm = pm.to(self.device)
        self.hm = hm[:, 1:-1, ...].to(self.device)
        self.fw = fw.to(self.device)
        self.bw = bw.to(self.device)

    def feed_data_test(self, data):
        lq, gt, pm, hm, fw, bw = data['lq'], data['gt'], data['pm'], data['hm'], data['fw'], data['bw']
        self.lq = lq.to(self.device).unsqueeze(0)
        self.gt = gt[1:-1, ...].to(self.device).unsqueeze(0)
        self.pm = pm.to(self.device).unsqueeze(0)
        self.hm = hm[1:-1, ...].to(self.device).unsqueeze(0)
        self.fw = fw.to(self.device).unsqueeze(0)
        self.bw = bw.to(self.device).unsqueeze(0)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        self.lq = self.lq.half()
        with torch.cuda.amp.autocast():
            output = self.net_g(self.lq)
            self.output = output

            loss_dict = OrderedDict()

            l_pix = self.cri_pix(output, self.gt)
            l_pix_object = self.cri_pix(output, self.gt, weight=self.hm)

            loss_dict['l_pix'] = l_pix
            loss_dict['l_pix_object'] = l_pix_object

            # Loss: 0.01 * l_pix + 0.1 * l_pix_object + l_pix_object_local_blur + 0* Regularization
            l_total = 0.01 * l_pix + 0.1 * l_pix_object + 0 * sum(
                p.sum() for p in self.net_g.parameters())

        # l_total.backward()
        self.scaler.scale(l_total).backward()
        self.scaler.unscale_(self.optimizer_g)
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.scaler.step(self.optimizer_g)
        self.scaler.update()

        for k, v in self.reduce_loss_dict(loss_dict).items():
            self.log_dict[k].update(v)

        # exit(0)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.lq)
        self.net_g.train()

    def test_by_patch(self):
        self.net_g.eval()
        lq = self.lq

        with torch.no_grad():
            size_patch_testing = 256
            overlap_size = 64
            b, t, c, h, w = self.gt.shape
            stride = size_patch_testing - overlap_size
            h_idx_list = list(range(0, h - size_patch_testing, stride)) + [max(0, h - size_patch_testing)]
            w_idx_list = list(range(0, w - size_patch_testing, stride)) + [max(0, w - size_patch_testing)]
            E = torch.zeros(b, t, c, h, w)
            W = torch.zeros_like(E)
            focus_E = torch.zeros(b, t, 1, h, w)
            focus_W = torch.zeros_like(focus_E)
            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = lq[..., h_idx:h_idx + size_patch_testing, w_idx:w_idx + size_patch_testing]
                    pm_patch = self.pm[..., h_idx:h_idx + size_patch_testing, w_idx:w_idx + size_patch_testing]

                    out_patch = self.net_g(in_patch)

                    out_patch = out_patch.detach().cpu().reshape(b, t, c, size_patch_testing, size_patch_testing)
                    out_patch_mask = torch.ones_like(out_patch)

                    if True:
                        if h_idx < h_idx_list[-1]:
                            out_patch[..., -overlap_size // 2:, :] *= 0
                            out_patch_mask[..., -overlap_size // 2:, :] *= 0
                        if w_idx < w_idx_list[-1]:
                            out_patch[..., :, -overlap_size // 2:] *= 0
                            out_patch_mask[..., :, -overlap_size // 2:] *= 0
                        if h_idx > h_idx_list[0]:
                            out_patch[..., :overlap_size // 2, :] *= 0
                            out_patch_mask[..., :overlap_size // 2, :] *= 0
                        if w_idx > w_idx_list[0]:
                            out_patch[..., :, :overlap_size // 2] *= 0
                            out_patch_mask[..., :, :overlap_size // 2] *= 0

                    E[..., h_idx:(h_idx + size_patch_testing), w_idx:(w_idx + size_patch_testing)].add_(out_patch)
                    W[..., h_idx:(h_idx + size_patch_testing), w_idx:(w_idx + size_patch_testing)].add_(out_patch_mask)

            output = E.div_(W)
        self.output = output[:, :, :, :, :]
        self.net_g.train()

    def validation(self, dataloader, current_iter, tb_logger, wandb_logger=None, save_img=False,save_img_path=None):
        """Validation function.
        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration.
            tb_logger (tensorboard logger): Tensorboard logger.
            wandb_loggger (wandb logger): wandb runer logger.
            save_img (bool): Whether to save images. Default: False.
        """
        if self.opt['dist']:
            self.dist_validation(dataloader, current_iter, tb_logger, wandb_logger, save_img, rgb2bgr=True,
                                 use_image=True, save_img_path=save_img_path)
        else:
            self.dist_validation(dataloader, current_iter, tb_logger, wandb_logger, save_img, rgb2bgr=True,
                                 use_image=True, save_img_path=save_img_path)

    def dist_validation(self, dataloader, current_iter, tb_logger, wandb_logger, save_img, rgb2bgr=True,
                        use_image=True, save_img_path=None):
        dataset = dataloader.dataset
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {}
            for folder, seq_index in dataset.splite_seqs_index.items():
                self.metric_results[folder] = torch.zeros(
                    len(seq_index["seq_index"]), len(self.opt['val']['metrics']), dtype=torch.float32, device='cuda')
        self._initialize_best_metric_results(dataset_name)

        rank, world_size = get_dist_info()
        num_seq = len(dataset)
        num_pad = (world_size - (num_seq % world_size)) % world_size
        if rank == 0:
            pbar = tqdm(total=len(dataset), unit='image')
        metric_data = dict()
        for i in range(rank, num_seq + num_pad, world_size):
            idx_data = min(i, num_seq - 1)
            # print(idx_data)
            val_data = dataset[idx_data]
            folder = val_data["seq_name"]
            seq_index = val_data["seq"]
            self.feed_data_test(val_data)
            self.test_by_patch()

            visuals = self.get_current_visuals()
            del self.lq
            del self.output
            del self.gt
            del self.hm
            del self.fw
            del self.bw
            torch.cuda.empty_cache()
            if save_img:
                save_img_path_scene = os.path.join(save_img_path, val_data['folder'])
                if not os.path.exists(save_img_path_scene):
                    os.makedirs(save_img_path_scene)
            if i < num_seq:
                for idx in range(visuals['result'].size(1)):
                    result = visuals['result'][0, idx, :, :, :]
                    result_img = tensor2img([result])  # uint8, bgr
                    metric_data['img'] = result_img
                    if save_img:
                        whole_idx = int(folder.split('_')[-1]) + idx
                        cv2.imwrite(os.path.join(save_img_path_scene, f'{whole_idx:05d}.png'), result_img)
                    if 'gt' in visuals:
                        gt = visuals['gt'][0, idx, :, :, :]
                        gt_img = tensor2img([gt])  # uint8, bgr
                        metric_data['img2'] = gt_img
                    if 'hm' in visuals:
                        hm = visuals['hm'][0, idx, :, :, :]
                        hm = tensor2img([hm])  # uint8, bgr
                        metric_data['mask'] = hm

                    if with_metrics:
                        for metric_idx, opt_ in enumerate(self.opt['val']['metrics'].values()):
                            result = calculate_metric(metric_data, opt_)
                            self.metric_results[folder][idx, metric_idx] += result

                if rank == 0:
                    for _ in range(world_size):
                        pbar.update(1)
                        pbar.set_description(f'Folder: {folder}')
        if rank == 0:
            pbar.close()
        if with_metrics:
            if self.opt['dist']:

                # collect data among GPUs
                for _, tensor in self.metric_results.items():
                    dist.reduce(tensor, 0)

                dist.barrier()

            if rank == 0:
                self._log_validation_metric_values(current_iter, dataset_name,
                                                   tb_logger, wandb_logger)

        out_metric = 0.
        for name in self.metric_results.keys():
            out_metric = self.metric_results[name]

        return out_metric

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger, wandb_logger):
        metric_results_avg = {
            folder: torch.mean(tensor, dim=0).cpu()
            for (folder, tensor) in self.metric_results.items()
        }
        total_avg_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        for folder, value in self.metric_results.items():

            for idx, metric in enumerate(total_avg_results.keys()):
                total_avg_results[metric] += metric_results_avg[folder][idx].item()

        for metric in total_avg_results.keys():
            total_avg_results[metric] /= len(metric_results_avg)
            # update the best metric result
            self._update_best_metric_result(dataset_name, metric, total_avg_results[metric], current_iter)
        log_str = f'Validation {dataset_name},\t'
        for metric_idx, (metric, value) in enumerate(total_avg_results.items()):
            log_str += f'\t # {metric}: {value:.4f}'
            for folder, tensor in metric_results_avg.items():
                log_str += f'\t # {folder}: {tensor[metric_idx].item():.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in total_avg_results.items():
                tb_logger.add_scalar(f'{dataset_name}/metrics/{metric}', value, current_iter)
                if wandb_logger is not None:
                    wandb_logger.log({f'{dataset_name}/metrics/{metric}': value}, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq[:, 1:-1 ,...].detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        if hasattr(self, 'hm'):
            out_dict['hm'] = self.hm.detach().cpu()
        if hasattr(self, 'focus'):
            out_dict['focus'] = self.focus.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
