import glob
import torch
from os import path as osp
from torch.utils import data as data

from basicsr.data.data_util import duf_downsample, generate_frame_indices, read_img_seq, read_flo_seq
from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class VideoTestDataset(data.Dataset):
    """Video test dataset.

    Supported datasets: Vid4, REDS4, REDSofficial.
    More generally, it supports testing dataset with following structures:

    dataroot
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── ...

    For testing datasets, there is no need to prepare LMDB files.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type and other kwarg.
            cache_data (bool): Whether to cache testing datasets.
            name (str): Dataset name.
            meta_info_file (str): The path to the file storing the list of test
                folders. If not provided, all the folders in the dataroot will
                be used.
            num_frame (int): Window size for input frames.
            padding (str): Padding mode.
    """

    def __init__(self, opt):
        super(VideoTestDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.data_info = {'lq_path': [], 'gt_path': [], 'folder': [], 'idx': [], 'border': []}
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        assert self.io_backend_opt['type'] != 'lmdb', 'No need to use lmdb during validation/test.'

        logger = get_root_logger()
        logger.info(f'Generate data info for VideoTestDataset - {opt["name"]}')
        self.imgs_lq, self.imgs_gt = {}, {}
        if 'meta_info_file' in opt:
            with open(opt['meta_info_file'], 'r') as fin:
                subfolders = [line.split(' ')[0] for line in fin]
                subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
                subfolders_gt = [osp.join(self.gt_root, key) for key in subfolders]
        else:
            subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, '*')))
            subfolders_gt = sorted(glob.glob(osp.join(self.gt_root, '*')))

        if opt['name'].lower() in ['vid4', 'reds4', 'redsofficial']:
            for subfolder_lq, subfolder_gt in zip(subfolders_lq, subfolders_gt):
                # get frame list for lq and gt
                subfolder_name = osp.basename(subfolder_lq)
                img_paths_lq = sorted(list(scandir(subfolder_lq, full_path=True)))
                img_paths_gt = sorted(list(scandir(subfolder_gt, full_path=True)))

                max_idx = len(img_paths_lq)
                assert max_idx == len(img_paths_gt), (f'Different number of images in lq ({max_idx})'
                                                      f' and gt folders ({len(img_paths_gt)})')

                self.data_info['lq_path'].extend(img_paths_lq)
                self.data_info['gt_path'].extend(img_paths_gt)
                self.data_info['folder'].extend([subfolder_name] * max_idx)
                for i in range(max_idx):
                    self.data_info['idx'].append(f'{i}/{max_idx}')
                border_l = [0] * max_idx
                for i in range(self.opt['num_frame'] // 2):
                    border_l[i] = 1
                    border_l[max_idx - i - 1] = 1
                self.data_info['border'].extend(border_l)

                # cache data or save the frame list
                if self.cache_data:
                    logger.info(f'Cache {subfolder_name} for VideoTestDataset...')
                    self.imgs_lq[subfolder_name] = read_img_seq(img_paths_lq)
                    self.imgs_gt[subfolder_name] = read_img_seq(img_paths_gt)
                else:
                    self.imgs_lq[subfolder_name] = img_paths_lq
                    self.imgs_gt[subfolder_name] = img_paths_gt
        else:
            raise ValueError(f'Non-supported video test dataset: {type(opt["name"])}')

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]
        lq_path = self.data_info['lq_path'][index]

        select_idx = generate_frame_indices(idx, max_idx, self.opt['num_frame'], padding=self.opt['padding'])

        if self.cache_data:
            imgs_lq = self.imgs_lq[folder].index_select(0, torch.LongTensor(select_idx))
            img_gt = self.imgs_gt[folder][idx]
        else:
            img_paths_lq = [self.imgs_lq[folder][i] for i in select_idx]
            imgs_lq = read_img_seq(img_paths_lq)
            img_gt = read_img_seq([self.imgs_gt[folder][idx]])
            img_gt.squeeze_(0)

        return {
            'lq': imgs_lq,  # (t, c, h, w)
            'gt': img_gt,  # (c, h, w)
            'folder': folder,  # folder name
            'idx': self.data_info['idx'][index],  # e.g., 0/99
            'border': border,  # 1 for border, 0 for non-border
            'lq_path': lq_path  # center frame
        }

    def __len__(self):
        return len(self.data_info['gt_path'])


@DATASET_REGISTRY.register()
class VideoTestVimeo90KDataset(data.Dataset):
    """Video test dataset for Vimeo90k-Test dataset.

    It only keeps the center frame for testing.
    For testing datasets, there is no need to prepare LMDB files.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type and other kwarg.
            cache_data (bool): Whether to cache testing datasets.
            name (str): Dataset name.
            meta_info_file (str): The path to the file storing the list of test
                folders. If not provided, all the folders in the dataroot will
                be used.
            num_frame (int): Window size for input frames.
            padding (str): Padding mode.
    """

    def __init__(self, opt):
        super(VideoTestVimeo90KDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        if self.cache_data:
            raise NotImplementedError('cache_data in Vimeo90K-Test dataset is not implemented.')
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.data_info = {'lq_path': [], 'gt_path': [], 'folder': [], 'idx': [], 'border': []}
        neighbor_list = [i + (9 - opt['num_frame']) // 2 for i in range(opt['num_frame'])]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        assert self.io_backend_opt['type'] != 'lmdb', 'No need to use lmdb during validation/test.'

        logger = get_root_logger()
        logger.info(f'Generate data info for VideoTestDataset - {opt["name"]}')
        with open(opt['meta_info_file'], 'r') as fin:
            subfolders = [line.split(' ')[0] for line in fin]
        for idx, subfolder in enumerate(subfolders):
            gt_path = osp.join(self.gt_root, subfolder, 'im4.png')
            self.data_info['gt_path'].append(gt_path)
            lq_paths = [osp.join(self.lq_root, subfolder, f'im{i}.png') for i in neighbor_list]
            self.data_info['lq_path'].append(lq_paths)
            self.data_info['folder'].append('vimeo90k')
            self.data_info['idx'].append(f'{idx}/{len(subfolders)}')
            self.data_info['border'].append(0)

    def __getitem__(self, index):
        lq_path = self.data_info['lq_path'][index]
        gt_path = self.data_info['gt_path'][index]
        imgs_lq = read_img_seq(lq_path)
        img_gt = read_img_seq([gt_path])
        img_gt.squeeze_(0)

        return {
            'lq': imgs_lq,  # (t, c, h, w)
            'gt': img_gt,  # (c, h, w)
            'folder': self.data_info['folder'][index],  # folder name
            'idx': self.data_info['idx'][index],  # e.g., 0/843
            'border': self.data_info['border'][index],  # 0 for non-border
            'lq_path': lq_path[self.opt['num_frame'] // 2]  # center frame
        }

    def __len__(self):
        return len(self.data_info['gt_path'])


@DATASET_REGISTRY.register()
class VideoTestDUFDataset(VideoTestDataset):
    """ Video test dataset for DUF dataset.

    Args:
        opt (dict): Config for train dataset.
            Most of keys are the same as VideoTestDataset.
            It has the following extra keys:

            use_duf_downsampling (bool): Whether to use duf downsampling to
                generate low-resolution frames.
            scale (bool): Scale, which will be added automatically.
    """

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]
        lq_path = self.data_info['lq_path'][index]

        select_idx = generate_frame_indices(idx, max_idx, self.opt['num_frame'], padding=self.opt['padding'])

        if self.cache_data:
            if self.opt['use_duf_downsampling']:
                # read imgs_gt to generate low-resolution frames
                imgs_lq = self.imgs_gt[folder].index_select(0, torch.LongTensor(select_idx))
                imgs_lq = duf_downsample(imgs_lq, kernel_size=13, scale=self.opt['scale'])
            else:
                imgs_lq = self.imgs_lq[folder].index_select(0, torch.LongTensor(select_idx))
            img_gt = self.imgs_gt[folder][idx]
        else:
            if self.opt['use_duf_downsampling']:
                img_paths_lq = [self.imgs_gt[folder][i] for i in select_idx]
                # read imgs_gt to generate low-resolution frames
                imgs_lq = read_img_seq(img_paths_lq, require_mod_crop=True, scale=self.opt['scale'])
                imgs_lq = duf_downsample(imgs_lq, kernel_size=13, scale=self.opt['scale'])
            else:
                img_paths_lq = [self.imgs_lq[folder][i] for i in select_idx]
                imgs_lq = read_img_seq(img_paths_lq)
            img_gt = read_img_seq([self.imgs_gt[folder][idx]], require_mod_crop=True, scale=self.opt['scale'])
            img_gt.squeeze_(0)

        return {
            'lq': imgs_lq,  # (t, c, h, w)
            'gt': img_gt,  # (c, h, w)
            'folder': folder,  # folder name
            'idx': self.data_info['idx'][index],  # e.g., 0/99
            'border': border,  # 1 for border, 0 for non-border
            'lq_path': lq_path  # center frame
        }


@DATASET_REGISTRY.register()
class VideoDeblurTestDataset(data.Dataset):
    """Video test dataset.

    Supported datasets: Vid4, REDS4, REDSofficial.
    More generally, it supports testing dataset with following structures:

    dataroot
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── ...

    For testing datasets, there is no need to prepare LMDB files.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type and other kwarg.
            cache_data (bool): Whether to cache testing datasets.
            name (str): Dataset name.
            meta_info_file (str): The path to the file storing the list of test
                folders. If not provided, all the folders in the dataroot will
                be used.
            num_frame (int): Window size for input frames.
            padding (str): Padding mode.
    """

    def __init__(self, opt):
        super(VideoDeblurTestDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.gt_root, self.lq_root, self.pm_root, self.hm_root, self.fw_root, self.bw_root = opt['dataroot_gt'], opt[
            'dataroot_lq'], opt[
            'dataroot_pm'], opt['dataroot_hm'], opt['dataroot_fw_residual'], opt['dataroot_bw_residual']
        self.data_info = {'lq_path': [], 'gt_path': [], 'pm_path': [], 'hm_path': [], 'fw_path': [], 'bw_path': [],
                          'folder': [], 'idx': [],
                          'border': []}
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        assert self.io_backend_opt['type'] != 'lmdb', 'No need to use lmdb during validation/test.'

        logger = get_root_logger()
        logger.info(f'Generate data info for VideoTestDataset - {opt["name"]}')
        self.imgs_lq, self.imgs_gt, self.imgs_pm, self.imgs_hm, self.imgs_fw, self.imgs_bw = {}, {}, {}, {}, {}, {}
        if 'meta_info_file' in opt:
            with open(opt['meta_info_file'], 'r') as fin:
                subfolders = [line.split(' ')[0] for line in fin]
                subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
                subfolders_gt = [osp.join(self.gt_root, key) for key in subfolders]
                subfolders_pm = [osp.join(self.pm_root, key) for key in subfolders]
                subfolders_hm = [osp.join(self.hm_root, key) for key in subfolders]
                subfolders_fw = [osp.join(self.fw_root, key) for key in subfolders]
                subfolders_bw = [osp.join(self.bw_root, key) for key in subfolders]
        else:
            subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, '*')))
            subfolders_gt = sorted(glob.glob(osp.join(self.gt_root, '*')))
            subfolders_pm = sorted(glob.glob(osp.join(self.pm_root, '*')))
            subfolders_hm = sorted(glob.glob(osp.join(self.hm_root, '*')))
            subfolders_fw = sorted(glob.glob(osp.join(self.fw_root, '*')))
            subfolders_bw = sorted(glob.glob(osp.join(self.bw_root, '*')))

        if opt['name'].lower() in ['vid4', 'reds4', 'redsofficial', 'dvd', 'gopro', 'vdv']:
            for subfolder_lq, subfolder_gt, subfolder_pm, subfolder_hm, subfolder_fw, subfolder_bw in zip(subfolders_lq,
                                                                                                          subfolders_gt,
                                                                                                          subfolders_pm,
                                                                                                          subfolders_hm,
                                                                                                          subfolders_fw,
                                                                                                          subfolders_bw):
                # get frame list for lq and gt
                subfolder_name = osp.basename(subfolder_lq)
                img_paths_lq = sorted(list(scandir(subfolder_lq, full_path=True)))
                img_paths_gt = sorted(list(scandir(subfolder_gt, full_path=True)))
                img_paths_pm = sorted(list(scandir(subfolder_pm, full_path=True)))
                img_paths_hm = sorted(list(scandir(subfolder_hm, full_path=True)))
                img_paths_fw = sorted(list(scandir(subfolder_fw, full_path=True)))
                img_paths_bw = sorted(list(scandir(subfolder_bw, full_path=True)))
                max_idx = len(img_paths_lq)
                assert max_idx == len(img_paths_gt), (f'Different number of images in lq ({max_idx})'
                                                      f' and gt folders ({len(img_paths_gt)})')

                self.data_info['lq_path'].extend(img_paths_lq)
                self.data_info['gt_path'].extend(img_paths_gt)
                self.data_info['pm_path'].extend(img_paths_pm)
                self.data_info['hm_path'].extend(img_paths_hm)
                self.data_info['fw_path'].extend(img_paths_fw)
                self.data_info['bw_path'].extend(img_paths_bw)
                self.data_info['folder'].extend([subfolder_name] * max_idx)
                for i in range(max_idx):
                    self.data_info['idx'].append(f'{i}/{max_idx}')
                border_l = [0] * max_idx
                for i in range(self.opt['num_frame'] // 2):
                    border_l[i] = 1
                    border_l[max_idx - i - 1] = 1
                self.data_info['border'].extend(border_l)

                # cache data or save the frame list
                if self.cache_data:
                    logger.info(f'Cache {subfolder_name} for VideoDeblurTestDataset...')
                    # print(img_paths_lq)
                    self.imgs_lq[subfolder_name] = read_img_seq(img_paths_lq)
                    self.imgs_gt[subfolder_name] = read_img_seq(img_paths_gt)
                    self.imgs_pm[subfolder_name] = read_img_seq(img_paths_pm)
                    self.imgs_hm[subfolder_name] = read_img_seq(img_paths_hm)
                    self.imgs_fw[subfolder_name] = read_img_seq(img_paths_fw)
                    self.imgs_bw[subfolder_name] = read_img_seq(img_paths_bw)
                else:
                    self.imgs_lq[subfolder_name] = img_paths_lq
                    self.imgs_gt[subfolder_name] = img_paths_gt
                    self.imgs_pm[subfolder_name] = img_paths_pm
                    self.imgs_hm[subfolder_name] = img_paths_hm
                    self.imgs_fw[subfolder_name] = img_paths_fw
                    self.imgs_bw[subfolder_name] = img_paths_bw
        else:
            raise ValueError(f'Non-supported video test dataset: {type(opt["name"])}')

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]
        lq_path = self.data_info['lq_path'][index]

        select_idx = generate_frame_indices(idx, max_idx, self.opt['num_frame'], padding=self.opt['padding'])

        if self.cache_data:
            imgs_lq = self.imgs_lq[folder].index_select(0, torch.LongTensor(select_idx))
            img_gt = self.imgs_gt[folder][idx]
        else:
            img_paths_lq = [self.imgs_lq[folder][i] for i in select_idx]
            imgs_lq = read_img_seq(img_paths_lq)
            img_gt = read_img_seq([self.imgs_gt[folder][idx]])
            img_gt.squeeze_(0)

        return {
            'lq': imgs_lq,  # (t, c, h, w)
            'gt': img_gt,  # (c, h, w)
            'folder': folder,  # folder name
            'idx': self.data_info['idx'][index],  # e.g., 0/99
            'border': border,  # 1 for border, 0 for non-border
            'lq_path': lq_path  # center frame
        }

    def __len__(self):
        return len(self.data_info['gt_path'])


@DATASET_REGISTRY.register()
class VideoRecurrentTestDataset(VideoDeblurTestDataset):
    """Video test dataset for recurrent architectures, which takes LR video
    frames as input and output corresponding HR video frames.

    Args:
        Same as VideoTestDataset.
        Unused opt:
            padding (str): Padding mode.

    """

    def __init__(self, opt):
        super(VideoRecurrentTestDataset, self).__init__(opt)
        # Find unique folder strings
        self.folders = sorted(list(set(self.data_info['folder'])))

    def __getitem__(self, index):
        folder = self.folders[index]

        if self.cache_data:
            imgs_lq = self.imgs_lq[folder]
            imgs_gt = self.imgs_gt[folder]
            imgs_pm = self.imgs_pm[folder]
        else:
            # raise NotImplementedError('Without cache_data is not implemented.')
            """ if len(self.imgs_lq[folder]) > 100:
                imgs_lq = read_img_seq(self.imgs_lq[folder][:100])
                imgs_gt = read_img_seq(self.imgs_gt[folder][:100])
            else:
                imgs_lq = read_img_seq(self.imgs_lq[folder])
                imgs_gt = read_img_seq(self.imgs_gt[folder]) """

            imgs_lq = read_img_seq(self.imgs_lq[folder])
            imgs_gt = read_img_seq(self.imgs_gt[folder])

        return {
            'lq': imgs_lq,
            'gt': imgs_gt,
            'folder': folder,
            'seq': len(self.imgs_lq[folder])
        }

    def __len__(self):
        return len(self.folders)


@DATASET_REGISTRY.register()
class VideoRecurrentTestDatasetlocal(VideoDeblurTestDataset):
    """Video test dataset for recurrent architectures, which takes LR video
    frames as input and output corresponding HR video frames.

    Args:
        Same as VideoTestDataset.
        Unused opt:
            padding (str): Padding mode.

    """

    def __init__(self, opt):
        super(VideoRecurrentTestDatasetlocal, self).__init__(opt)
        # Find unique folder strings
        self.folders = sorted(list(set(self.data_info['folder'])))
        self.splite_seqs_index = {}
        num_frame_testing = opt['num_frame']
        num_frame_overlapping = 4
        stride = max(1, num_frame_testing - num_frame_overlapping)
        self.seq_names = []
        if opt['num_frame'] != -1:
            for folder in self.folders:

                d = len(self.imgs_gt[folder])

                if d < num_frame_testing:
                    selected_index = [
                        i for i in range(0, d)
                    ]

                    self.splite_seqs_index[f"{folder}_{d_index}"] = {
                        "seq_index": selected_index,
                        "seq_from_folder": folder
                    }
                    self.seq_names += [f"{folder}_{d_index}"]
                else:
                    d_idx_list = list(range(0, d - num_frame_testing, stride)) + [max(0, d - num_frame_testing)]
                    for d_index in d_idx_list:
                        selected_index = [
                            i for i in range(d_index, d_index + num_frame_testing)
                        ]
                        self.splite_seqs_index[f"{folder}_{d_index}"] = {
                            "seq_index": selected_index,
                            "seq_from_folder": folder
                        }
                        self.seq_names += [f"{folder}_{d_index}"]

        else:
            self.seq_names = self.folders
            for folder in self.folders:
                d = len(self.imgs_gt[folder])
                self.splite_seqs_index[folder] = {
                    "seq_index": list(range(0, d)),
                    "seq_from_folder": folder
                }
        self.scale_factor = opt['scale_factor']

    def __getitem__(self, index):
        # folder = self.folders[index]
        seq_name = self.seq_names[index]

        selected_index, folder = self.splite_seqs_index[seq_name]["seq_index"], self.splite_seqs_index[seq_name][
            "seq_from_folder"]
        lq_paths = [self.imgs_lq[folder][p] for p in selected_index]
        gt_paths = [self.imgs_gt[folder][p] for p in selected_index]
        pm_paths = [self.imgs_pm[folder][p] for p in selected_index]
        hm_paths = [self.imgs_hm[folder][p] for p in selected_index]
        fw_paths = [self.imgs_fw[folder][p] for p in selected_index]
        bw_paths = [self.imgs_bw[folder][p] for p in selected_index]

        imgs_lq = read_img_seq(lq_paths)
        imgs_gt = read_img_seq(gt_paths)
        imgs_pm = read_img_seq(pm_paths)
        imgs_hm = read_img_seq(hm_paths)
        imgs_fw = read_flo_seq(fw_paths)
        imgs_bw = read_flo_seq(bw_paths)

        # resize the images,gts,pms to the same size with hms [720,1280]
        H, W = imgs_hm.shape[2], imgs_hm.shape[3]
        imgs_lq = torch.nn.functional.interpolate(imgs_lq, size=(H, W), mode='bilinear', align_corners=False)
        imgs_gt = torch.nn.functional.interpolate(imgs_gt, size=(H, W), mode='bilinear', align_corners=False)
        imgs_pm = torch.nn.functional.interpolate(imgs_pm, size=(H, W), mode='bilinear', align_corners=False)
        imgs_fw = torch.nn.functional.interpolate(imgs_fw, size=(H // self.scale_factor, W // self.scale_factor),
                                                  mode='bilinear', align_corners=False)
        imgs_bw = torch.nn.functional.interpolate(imgs_bw, size=(H // self.scale_factor, W // self.scale_factor),
                                                  mode='bilinear', align_corners=False)
        return {
            'lq': imgs_lq,
            'gt': imgs_gt,
            'pm': imgs_pm,
            'hm': imgs_hm,
            'fw': imgs_fw,
            'bw': imgs_bw,
            'folder': folder,
            'seq_name': seq_name,
            'seq': selected_index
        }

    def __len__(self):
        return len(self.seq_names)
