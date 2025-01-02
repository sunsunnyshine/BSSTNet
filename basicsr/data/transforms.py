import cv2
import random
import torch
import numpy as np


def mod_crop(img, scale):
    """Mod crop images, used during testing.

    Args:
        img (ndarray): Input image.
        scale (int): Scale factor.

    Returns:
        ndarray: Result image.
    """
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[:h - h_remainder, :w - w_remainder, ...]
    else:
        raise ValueError(f'Wrong img ndim: {img.ndim}.')
    return img


def paired_random_crop_target_aware(img_gts, img_lqs, img_pms, img_hms, img_bws, img_fws, gt_patch_size, scale,
                                    force_blur_region_p=0.5, gt_path=None):
    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]
    if not isinstance(img_gts, list):
        img_pms = [img_pms]
    if not isinstance(img_hms, list):
        img_hms = [img_hms]
    if not isinstance(img_bws, list):
        img_bws = [img_bws]
    if not isinstance(img_fws, list):
        img_fws = [img_fws]

    # resize the images,gts,pms to the same size with hms [720,1280]
    scale_factor = 4
    img_gts = [cv2.resize(v, (img_hms[0].shape[1], img_hms[0].shape[0])) for v in img_gts]
    img_lqs = [cv2.resize(v, (img_hms[0].shape[1], img_hms[0].shape[0])) for v in img_lqs]
    img_pms = [cv2.resize(v, (img_hms[0].shape[1], img_hms[0].shape[0])) for v in img_pms]
    img_fws = [cv2.resize(v, (img_hms[0].shape[1] // scale_factor, img_hms[0].shape[0] // scale_factor)) for v in
               img_fws]
    img_bws = [cv2.resize(v, (img_hms[0].shape[1] // scale_factor, img_hms[0].shape[0] // scale_factor)) for v in
               img_bws]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq, w_lq = img_lqs[0].size()[-2:]
        h_gt, w_gt = img_gts[0].size()[-2:]
        h_pm, w_pm = img_pms[0].size()[-2:]
        h_hm, w_hm = img_hms[0].size()[-2:]
    else:
        h_lq, w_lq = img_lqs[0].shape[0:2]
        h_gt, w_gt = img_gts[0].shape[0:2]
        h_pm, w_pm = img_pms[0].shape[0:2]
        h_hm, w_hm = img_hms[0].shape[0:2]

    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                         f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    centor_h = random.randint(lq_patch_size // 2, h_pm - lq_patch_size // 2)
    centor_w = random.randint(lq_patch_size // 2, w_pm - lq_patch_size // 2)
    # whether to use the Blur-Aware Patch Cropping Strategy
    force_blur_region = force_blur_region_p > 0 and random.random() < force_blur_region_p
    if force_blur_region:
        ps_aware = torch.from_numpy(sum(img_hms))
        R_blur = ps_aware.sum() / (np.prod(img_hms[0].shape) * img_hms.__len__())
        if R_blur > 0.05:
            blur_pixs = torch.nonzero(
                ps_aware[lq_patch_size // 2:-lq_patch_size // 2, lq_patch_size // 2:-lq_patch_size // 2, 0],
                as_tuple=False)
            if len(blur_pixs) > 0:
                centor_h, centor_w = blur_pixs[random.randint(0, len(blur_pixs) - 1)].tolist()
                centor_h += lq_patch_size // 2
                centor_w += lq_patch_size // 2

    # crop lq patch
    if input_type == 'Tensor':
        img_lqs = [v[centor_h - lq_patch_size // 2:centor_h + lq_patch_size // 2,
                   centor_w - lq_patch_size // 2:centor_w + lq_patch_size // 2, :] for v in img_lqs]
    else:
        img_lqs = [v[centor_h - lq_patch_size // 2:centor_h + lq_patch_size // 2,
                   centor_w - lq_patch_size // 2:centor_w + lq_patch_size // 2] for v in img_lqs]

    # crop the corresponding pm patch
    if input_type == 'Tensor':
        img_pms = [v[centor_h - lq_patch_size // 2:centor_h + lq_patch_size // 2,
                   centor_w - lq_patch_size // 2:centor_w + lq_patch_size // 2, :] for v in img_pms]
    else:
        img_pms = [v[centor_h - lq_patch_size // 2:centor_h + lq_patch_size // 2,
                   centor_w - lq_patch_size // 2:centor_w + lq_patch_size // 2] for v in img_pms]

    # crop the corresponding hm patch
    if input_type == 'Tensor':
        img_hms = [v[centor_h - lq_patch_size // 2:centor_h + lq_patch_size // 2,
                   centor_w - lq_patch_size // 2:centor_w + lq_patch_size // 2, :] for v in img_hms]
    else:
        img_hms = [v[centor_h - lq_patch_size // 2:centor_h + lq_patch_size // 2,
                   centor_w - lq_patch_size // 2:centor_w + lq_patch_size // 2] for v in img_hms]
    # crop corresponding gt patch
    centor_h, centor_w = int(centor_h * scale), int(centor_w * scale)
    if input_type == 'Tensor':
        img_gts = [v[centor_h - lq_patch_size // 2:centor_h + lq_patch_size // 2,
                   centor_w - lq_patch_size // 2:centor_w + lq_patch_size // 2, :] for v in img_gts]
    else:
        img_gts = [v[centor_h - lq_patch_size // 2:centor_h + lq_patch_size // 2,
                   centor_w - lq_patch_size // 2:centor_w + lq_patch_size // 2] for v in img_gts]

    # crop the corresponding bw and fw patch
    centor_w = centor_w // scale_factor
    centor_h = centor_h // scale_factor
    lq_patch_size = lq_patch_size // scale_factor
    if input_type == 'Tensor':
        img_fws = [v[centor_h - lq_patch_size // 2:centor_h + lq_patch_size // 2,
                   centor_w - lq_patch_size // 2:centor_w + lq_patch_size // 2, :] for v in img_fws]
        img_bws = [v[centor_h - lq_patch_size // 2:centor_h + lq_patch_size // 2,
                   centor_w - lq_patch_size // 2:centor_w + lq_patch_size // 2, :] for v in img_bws]
    else:
        img_fws = [v[centor_h - lq_patch_size // 2:centor_h + lq_patch_size // 2,
                   centor_w - lq_patch_size // 2:centor_w + lq_patch_size // 2] for v in img_fws]
        img_bws = [v[centor_h - lq_patch_size // 2:centor_h + lq_patch_size // 2,
                   centor_w - lq_patch_size // 2:centor_w + lq_patch_size // 2] for v in img_bws]

    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    if len(img_pms) == 1:
        img_pms = img_pms[0]
    if len(img_hms) == 1:
        img_hms = img_hms[0]
    if len(img_fws) == 1:
        img_fws = img_fws[0]
    if len(img_bws) == 1:
        img_bws = img_bws[0]

    return img_gts, img_lqs, img_pms, img_hms, img_bws, img_fws


def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path=None):
    """Paired random crop. Support Numpy array and Tensor inputs.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq, w_lq = img_lqs[0].size()[-2:]
        h_gt, w_gt = img_gts[0].size()[-2:]
    else:
        h_lq, w_lq = img_lqs[0].shape[0:2]
        h_gt, w_gt = img_gts[0].shape[0:2]
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                         f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    if input_type == 'Tensor':
        img_lqs = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lqs]
    else:
        img_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    if input_type == 'Tensor':
        img_gts = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_gts]
    else:
        img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs


def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


def img_rotate(img, angle, center=None, scale=1.0):
    """Rotate image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees. Positive values mean
            counter-clockwise rotation.
        center (tuple[int]): Rotation center. If the center is None,
            initialize it as the center of the image. Default: None.
        scale (float): Isotropic scale factor. Default: 1.0.
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img
