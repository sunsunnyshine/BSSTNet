# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import importlib
import torch
from torch import distributed as dist
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm

from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils.logger import AverageMeter
from basicsr.metrics import calculate_metric


@MODEL_REGISTRY.register()
class ImageRestorationModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageRestorationModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True),
                              param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        self.scale = int(opt['scale'])
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
            pass
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                #         if k.startswith('module.offsets') or k.startswith('module.dcns'):
                #             optim_params_lowlr.append(v)
                #         else:
                optim_params.append(v)
            # else:
            #     logger = get_root_logger()
            #     logger.warning(f'Params {k} will not be optimized.')
        # print(optim_params)
        # ratio = 0.1

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}],
                                                **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params,
                                               **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                                 **train_opt['optim_g'])
            pass
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        lq, gt, pm, hm, fw, bw = data['lq'], data['gt'], data['pm'], data['hm'], data['fw'], data['bw']
        self.lq = lq.to(self.device)
        self.gt = gt.to(self.device)
        self.pm = pm.to(self.device)
        self.hm = hm.to(self.device)
        self.fw = fw.to(self.device)
        self.bw = bw.to(self.device)

    def grids(self):
        b, c, h, w = self.gt.size()
        self.original_size = (b, c, h, w)

        assert b == 1
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale
        # adaptive step_i, step_j
        num_row = (h - 1) // crop_size_h + 1
        num_col = (w - 1) // crop_size_w + 1

        import math
        step_j = crop_size_w if num_col == 1 else math.ceil((w - crop_size_w) / (num_col - 1) - 1e-8)
        step_i = crop_size_h if num_row == 1 else math.ceil((h - crop_size_h) / (num_row - 1) - 1e-8)

        scale = self.scale
        step_i = step_i // scale * scale
        step_j = step_j // scale * scale

        parts = []
        idxes = []

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size_h >= h:
                i = h - crop_size_h
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + crop_size_w >= w:
                    j = w - crop_size_w
                    last_j = True
                parts.append(
                    self.lq[:, :, i // scale:(i + crop_size_h) // scale, j // scale:(j + crop_size_w) // scale])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w))
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i: i + crop_size_h, j: j + crop_size_w] += self.outs[cnt]
            count_mt[0, 0, i: i + crop_size_h, j: j + crop_size_w] += 1.

        self.output = (preds / count_mt).to(self.device)
        self.lq = self.origin_lq

    def optimize_parameters(self, current_iter, tb_logger=None):
        self.optimizer_g.zero_grad()

        if self.opt['train'].get('mixup', False):
            self.mixup_aug()

        preds = self.net_g(self.lq, self.fw, self.bw)
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        loss_dict = OrderedDict()

        l_pix = self.cri_pix(self.output, self.gt)
        l_pix_object = self.cri_pix(self.output, self.gt, weight=self.hm)

        loss_dict['l_pix'] = l_pix
        loss_dict['l_pix_object'] = l_pix_object

        # Loss: 0.01 * l_pix + 0.1 * l_pix_object + l_pix_object_local_blur + 0* Regularization
        l_total = 0.01 * l_pix + 0.1 * l_pix_object + 0 * sum(
            p.sum() for p in self.net_g.parameters())

        # l_total.backward()
        self.scaler.scale(l_total).backward()
        self.scaler.unscale_(self.optimizer_g)
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.001)
        self.scaler.step(self.optimizer_g)
        self.scaler.update()

        for k, v in self.reduce_loss_dict(loss_dict).items():
            self.log_dict[k].update(v)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.lq, self.pm)
        self.net_g.train()

    def test_by_patch(self):
        self.net_g.eval()
        lq = self.lq
        fw = self.fw
        bw = self.bw

        with torch.no_grad():
            size_patch_testing = 256
            overlap_size = 64
            b, c, h, w = lq.shape
            stride = size_patch_testing - overlap_size
            h_idx_list = list(range(0, h - size_patch_testing, stride)) + [max(0, h - size_patch_testing)]
            w_idx_list = list(range(0, w - size_patch_testing, stride)) + [max(0, w - size_patch_testing)]
            E = torch.zeros(b, c, h, w)
            W = torch.zeros_like(E)
            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    with torch.cuda.amp.autocast():
                        in_patch = lq[..., h_idx:h_idx + size_patch_testing, w_idx:w_idx + size_patch_testing]
                        fw_patch = fw[..., h_idx:h_idx + size_patch_testing, w_idx:w_idx + size_patch_testing]
                        bw_patch = bw[..., h_idx:h_idx + size_patch_testing, w_idx:w_idx + size_patch_testing]
                        out_patch = self.net_g(in_patch, fw_patch, bw_patch)

                    out_patch = out_patch.detach().cpu().reshape(b, c, size_patch_testing, size_patch_testing)
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
        self.output = output[:, :, :, :]
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image=False):
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
            self.feed_data(val_data)
            self.test_by_patch()

            visuals = self.get_current_visuals()

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if i < num_seq:
                for idx in range(visuals['lq'].size(1)):
                    result = visuals['result'][:, idx, :, :, :]
                    result_img = tensor2img([result])  # uint8, bgr
                    metric_data['img'] = result_img
                    if 'gt' in visuals:
                        gt = visuals['gt'][:, idx, :, :, :]
                        gt_img = tensor2img([gt])  # uint8, bgr
                        metric_data['img2'] = gt_img
                    if 'hm' in visuals:
                        hm = visuals['hm'][:, idx, :, :, :]
                        focus_mask = tensor2img([hm])
                        metric_data['mask'] = focus_mask

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
                                                   tb_logger, None)

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
        log_str = f'Validation {dataset_name}, \t'
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
        out_dict['lq'] = self.lq.unsqueeze(0).detach().cpu()
        out_dict['result'] = self.output.unsqueeze(0).detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.unsqueeze(0).detach().cpu()
        if hasattr(self, 'hm'):
            out_dict['hm'] = self.hm.unsqueeze(0).detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
