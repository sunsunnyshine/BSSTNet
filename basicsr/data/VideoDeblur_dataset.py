import numpy as np
import random
import torch
from pathlib import Path
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_target_aware
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor, readFlow
from basicsr.utils.flow_util import dequantize_flow
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class DeblurRecurrentDataset(data.Dataset):
    """Debblur dataset for training recurrent networks.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_REDS_GT.txt

    Each line contains:
    1. subfolder (clip) name; 2. frame number; 3. image shape, separated by
    a white space.
    Examples:
    000 100 (720,1280,3)
    001 100 (720,1280,3)
    ...

    Key examples: "000/00000000"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            dataroot_flow (str, optional): Data root path for flow.
            meta_info_file (str): Path for meta information file.
            val_partition (str): Validation partition types. 'REDS4' or
                'official'.
            io_backend (dict): IO backend type and other kwarg.

            num_frame (int): Window size for input frames.
            gt_size (int): Cropped patched size for gt patches.
            interval_list (list): Interval list for temporal augmentation.
            random_reverse (bool): Random reverse input frames.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(DeblurRecurrentDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root, self.pm_root, self.hm_root, self.fw_residual_root, self.bw_residual_root = Path(
            opt['dataroot_gt']), Path(
            opt['dataroot_lq']), Path(
            opt['dataroot_pm']), Path(opt['dataroot_hm']), Path(opt['dataroot_fw_residual']), Path(
            opt['dataroot_bw_residual'])
        self.num_frame = opt['num_frame']
        self.file_end = opt["file_end"]
        self.cache_data = opt["cache_data"]
        self.keys = []
        self.max_frames = {}
        self.data_infos = []
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, frame_num, _ = line.split(' ')
                self.max_frames[folder] = int(frame_num)
                self.keys.extend([f'{folder}/{i:05d}' for i in range(int(frame_num))])
                self.data_infos.append(
                    dict(
                        folder=folder,
                        sequence_length=int(frame_num)
                    )
                )
        # file client (io backend)
        self.file_client = None

        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            if hasattr(self, 'flow_root') and self.flow_root is not None:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root, self.pm_root, self.flow_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt', 'pm', 'hm', 'flow']
            else:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root, self.pm_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt', 'hm', 'pm']

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

        # Blur-aware patch cropping
        self.force_blur_region_p = opt.get('force_blur_region_p', 0.5)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale'] if self.opt['scale'] is not None else 1
        gt_size = self.opt['gt_size']
        # key = self.keys[index]
        index = index % self.data_infos.__len__()
        clip_name = self.data_infos[index]["folder"]
        max_frame = self.data_infos[index]['sequence_length']

        # clip_name, frame_name = key.split('/')  # key example: 000/00000000
        # max_frame = self.max_frames[clip_name]
        # print(max_frame)
        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        # start_frame_idx = int(frame_name)
        # if start_frame_idx > max_frame - self.num_frame * interval:
        # start_frame_idx = random.randint(0, max_frame - self.num_frame * interval)

        start_frame_idx = np.random.randint(0, max_frame - self.num_frame * interval + 1)
        end_frame_idx = min(start_frame_idx + self.num_frame * interval, max_frame)
        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the neighboring LQ and GT frames
        img_lqs = []
        img_gts = []
        img_pms = []
        img_hms = []
        img_fws = []
        img_bws = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip_name}/{neighbor:05d}'
                img_gt_path = f'{clip_name}/{neighbor:05d}'
                img_pm_path = f'{clip_name}/{neighbor:05d}'
                img_hm_path = f'{clip_name}/{neighbor:05d}'
                img_fw_residual_path = f'{clip_name}/{neighbor:05d}'
                img_bw_residual_path = f'{clip_name}/{neighbor:05d}'
            else:
                img_lq_path = self.lq_root / clip_name / f'{neighbor:05d}.{self.file_end}'
                img_gt_path = self.gt_root / clip_name / f'{neighbor:05d}.{self.file_end}'
                img_pm_path = self.pm_root / clip_name / f'{neighbor:05d}.{self.file_end}'
                img_hm_path = self.hm_root / clip_name / f'{neighbor:05d}.jpg'
                img_fw_residual_path = self.fw_residual_root / clip_name / f'{neighbor:05d}.flo'
                img_bw_residual_path = self.bw_residual_root / clip_name / f'{neighbor:05d}.flo'

            # get LQ
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

            # get GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

            # get PM
            img_bytes = self.file_client.get(img_pm_path, 'pm')
            img_pm = imfrombytes(img_bytes, float32=True)
            img_pms.append(img_pm)

            # get HM
            img_bytes = self.file_client.get(img_hm_path, 'hm')
            img_hm = imfrombytes(img_bytes, float32=True)
            img_hms.append(img_hm)

            # get forward residual flow
            img_fw_residual = readFlow(img_fw_residual_path)
            img_fws.append(img_fw_residual)

            # get backward residual flow
            img_bw_residual = readFlow(img_bw_residual_path)
            img_bws.append(img_bw_residual)

        # randomly crop
        img_gts, img_lqs, img_pms, img_hms, img_bws, img_fws = paired_random_crop_target_aware(img_gts, img_lqs,
                                                                                               img_pms, img_hms,
                                                                                               img_bws, img_fws,
                                                                                               gt_size, scale,
                                                                                               self.force_blur_region_p,
                                                                                               img_gt_path)

        # augmentation - flip, rotate
        if isinstance(img_gts, list):
            len = img_gts.__len__()
            img_lqs.extend(img_gts)
            img_lqs.extend(img_pms)
            img_lqs.extend(img_hms)
            img_fws.extend(img_bws)
            img_results, flow_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'], flows=img_fws)

            img_results = img2tensor(img_results)
            img_lqs = torch.stack(img_results[:len], dim=0)
            img_gts = torch.stack(img_results[len:2 * len], dim=0)
            img_pms = torch.stack(img_results[2 * len:3 * len], dim=0)
            img_hms = torch.stack(img_results[3 * len:], dim=0)

            flow_results = img2tensor(flow_results, bgr2rgb=False, float32=True)
            img_fws = torch.stack(flow_results[:len], dim=0)
            img_bws = torch.stack(flow_results[len:], dim=0)

            return {'lq': img_lqs, 'gt': img_gts, 'pm': img_pms, 'hm': img_hms, 'fw': img_fws, 'bw': img_bws,
                    'folder': [f"{clip_name}.{neighbor_list[0]}", f"{clip_name}.{neighbor_list[1]}"]}
        else:
            augment_list = [img_lqs, img_gts, img_hms, img_pms]
            augment_flow_list = [img_fws, img_bws]
            augment_list, flow_results = augment(augment_list, self.opt['use_hflip'], self.opt['use_rot'],
                                                 flows=augment_flow_list)
            img_lqs, img_gts, img_hms, img_pms = img2tensor(augment_list)
            img_fws, img_bws = img2tensor(flow_results, bgr2rgb=False, float32=True)

            return {'lq': img_lqs, 'gt': img_gts, 'pm': img_pms, 'hm': img_hms, 'fw': img_fws, 'bw': img_bws,
                    'folder': [f"{clip_name}.{neighbor_list[0]}"]}

    def __len__(self):
        return len(self.data_infos) * 10000


@DATASET_REGISTRY.register()
class DeblurRecurrentDatasetloadmemory(data.Dataset):
    """Debblur dataset for training recurrent networks.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_REDS_GT.txt

    Each line contains:
    1. subfolder (clip) name; 2. frame number; 3. image shape, separated by
    a white space.
    Examples:
    000 100 (720,1280,3)
    001 100 (720,1280,3)
    ...

    Key examples: "000/00000000"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            dataroot_flow (str, optional): Data root path for flow.
            meta_info_file (str): Path for meta information file.
            val_partition (str): Validation partition types. 'REDS4' or
                'official'.
            io_backend (dict): IO backend type and other kwarg.

            num_frame (int): Window size for input frames.
            gt_size (int): Cropped patched size for gt patches.
            interval_list (list): Interval list for temporal augmentation.
            random_reverse (bool): Random reverse input frames.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(DeblurRecurrentDatasetloadmemory, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.num_frame = opt['num_frame']
        self.file_end = opt["file_end"]

        self.keys = []
        self.max_frames = {}
        self.data_infos = []
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, frame_num, _ = line.split(' ')
                self.max_frames[folder] = int(frame_num)
                self.keys.extend([f'{folder}/{i:05d}' for i in range(int(frame_num))])
                self.data_infos.append(
                    dict(
                        folder=folder,
                        sequence_length=int(frame_num)
                    )
                )

        # remove the video clips used in validation
        """ if opt['val_partition'] == 'REDS4':
            val_partition = ['000', '011', '015', '020']
        elif opt['val_partition'] == 'official':
            val_partition = [f'{v:03d}' for v in range(240, 270)]
        else:
            raise ValueError(f'Wrong validation partition {opt["val_partition"]}.'
                             f"Supported ones are ['official', 'REDS4'].")
        if opt['test_mode']:
            self.keys = [v for v in self.keys if v.split('/')[0] in val_partition]
        else:
            self.keys = [v for v in self.keys if v.split('/')[0] not in val_partition] """

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            if hasattr(self, 'flow_root') and self.flow_root is not None:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root, self.flow_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt', 'flow']
            else:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        self.gt_cache_images = {}
        self.lq_cache_images = {}
        for info_data in self.data_infos:
            folder = info_data['folder']
            sequence_length = info_data['sequence_length']
            self.lq_cache_images[folder] = []
            self.gt_cache_images[folder] = []
            for neighbor in range(sequence_length):
                img_lq_path = self.lq_root / folder / f'{neighbor:05d}.{self.file_end}'
                img_gt_path = self.gt_root / folder / f'{neighbor:05d}.{self.file_end}'

                img_bytes = self.file_client.get(img_lq_path, 'lq')
                img_lq = imfrombytes(img_bytes, float32=True)

                img_bytes = self.file_client.get(img_gt_path, 'gt')
                img_gt = imfrombytes(img_bytes, float32=True)

                self.lq_cache_images[folder] += [img_lq]
                self.gt_cache_images[folder] += [img_gt]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        # key = self.keys[index]
        clip_name = self.data_infos[index]["folder"]
        max_frame = self.data_infos[index]['sequence_length']

        # clip_name, frame_name = key.split('/')  # key example: 000/00000000
        # max_frame = self.max_frames[clip_name]
        # print(max_frame)
        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        # start_frame_idx = int(frame_name)
        # if start_frame_idx > max_frame - self.num_frame * interval:
        # start_frame_idx = random.randint(0, max_frame - self.num_frame * interval)

        start_frame_idx = np.random.randint(0, max_frame - self.num_frame * interval + 1)
        end_frame_idx = start_frame_idx + self.num_frame * interval
        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the neighboring LQ and GT frames
        img_lqs = []
        img_gts = []
        for neighbor in neighbor_list:
            """ if self.is_lmdb:
                img_lq_path = f'{clip_name}/{neighbor:05d}'
                img_gt_path = f'{clip_name}/{neighbor:05d}'
            else:
                img_lq_path = self.lq_root / clip_name / f'{neighbor:05d}.{self.file_end}'
                img_gt_path = self.gt_root / clip_name / f'{neighbor:05d}.{self.file_end}' """
            img_gt_path = self.gt_root / clip_name / f'{neighbor:05d}.{self.file_end}'

            # get LQ
            img_lqs = self.lq_cache_images[clip_name][neighbor_list]

            # get GT
            img_gts = self.gt_cache_images[clip_name][neighbor_list]

        # randomly crop
        img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size, scale, img_gt_path)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_gts = torch.stack(img_results[len(img_lqs) // 2:], dim=0)
        img_lqs = torch.stack(img_results[:len(img_lqs) // 2], dim=0)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        # return {'lq': img_lqs, 'gt': img_gts, 'key': key}
        return {'lq': img_lqs, 'gt': img_gts,
                'folder': [f"{clip_name}.{neighbor_list[0]}", f"{clip_name}.{neighbor_list[1]}"]}

    def __len__(self):
        return len(self.data_infos)
