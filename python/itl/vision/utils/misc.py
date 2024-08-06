"""
Miscellaneous utility methods that don't classify into other files in utils
"""
from math import sqrt
from itertools import product

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
from PIL import ImageFilter
from skimage.morphology import dilation
from scipy.optimize import linear_sum_assignment


BLUR_RADIUS = 5                 # Gaussian blur kernel radius for background image

def blur_and_grayscale(image):
    """ Mainly for use as 'background' in 'visual prompting' """
    bg_image = image.convert("L").convert("RGB")
    bg_image = bg_image.filter(ImageFilter.GaussianBlur(BLUR_RADIUS))
    
    return bg_image


def visual_prompt_by_mask(image, bg_image, masks):
    """ 'Visual prompt engineering' (cf. CLIPSeg paper) """
    # Obtain visual prompts to process by mixing image & bg_image per each mask...
    images_mixed = [
        (image * msk[:,:,None] + bg_image * (1-msk[:,:,None])).astype("uint8")
        for msk in masks
    ]

    # ... then cropping with some context & square pad as needed
    visual_prompts = []
    for i, msk in enumerate(masks):
        nonzero_y, nonzero_x = msk.nonzero()
        x1 = nonzero_x.min(); x2 = nonzero_x.max(); w = x2-x1
        y1 = nonzero_y.min(); y2 = nonzero_y.max(); h = y2-y1

        pad_ratio = 1/16          # Relative size to one side
        if w >= h:
            w_pad = int(w*pad_ratio)
            target_size = w + 2*w_pad
            h_pad = (target_size-h) // 2
            # w_pad = int(w*pad_ratio)
            # h_pad = w_pad
        else:
            h_pad = int(h*pad_ratio)
            target_size = h + 2*h_pad
            w_pad = (target_size-w) // 2
            # h_pad = int(h*pad_ratio)
            # w_pad = h_pad

        x1_crp = max(0, x1-w_pad); x2_crp = min(image.width, x2+w_pad)
        y1_crp = max(0, y1-h_pad); y2_crp = min(image.height, y2+h_pad)
        pad_spec = (
            -min(0, x1-w_pad), max(image.width, x2+w_pad) - image.width,
            -min(0, y1-h_pad), max(image.height, y2+h_pad) - image.height
        )

        # Draw contour as guided by mask
        contour = dilation(dilation(msk)) & ~msk
        images_mixed[i][contour] = [255,0,0]

        cropped = torch.tensor(images_mixed[i][y1_crp:y2_crp, x1_crp:x2_crp])
        cropped = F.pad(cropped.permute(2,0,1), pad_spec)
        visual_prompts.append(cropped)
    
    return visual_prompts


def crop_images_by_masks(images, masks):
    """
    Crop each image by area specified by the corresponding mask (encasing bounding
    box). Return cropped images along with cropped masks, and coordinates of
    cropping box boundaries. Each image is a PIL.Image instance, each mask is
    a two-dimensional binary mask (H * W)
    """
    assert len(images) == len(masks)
    assert all(
        img.width==masks[i].shape[1] and img.height==masks[i].shape[0]
        for i, img in images.items()
    )

    cropped_images = []; cropped_masks = []; dimensions = []
    for i, img in images.items():
        # Get mask extents
        msk = masks[i]
        mask_extents = msk.nonzero()
        x_min = mask_extents[1].min(); x_max = mask_extents[1].max()
        y_min = mask_extents[0].min(); y_max = mask_extents[0].max()
        crop_width = x_max - x_min; crop_height = y_max - y_min

        # Bounding box extremities; 10% margin of width and height
        pad_length = min(crop_width*0.05, crop_height*0.05)
        left = max(0, int(x_min - pad_length))
        right = min(img.width, int(x_max + pad_length))
        top = max(0, int(y_min - pad_length))
        bottom = min(img.height, int(y_max + pad_length))

        # Append values to return lists
        cropped_images.append(img.crop((left, top, right, bottom)))
        cropped_masks.append(msk[top:bottom, left:right])
        dimensions.append((left, right, top, bottom))

    return cropped_images, cropped_masks, dimensions


def quat2rmat(quat):
    """ Convert rotation quaternion to matrix representation """
    qw, qx, qy, qz = quat

    # First row
    r11 = 2 * (qw * qw + qx * qx) - 1
    r12 = 2 * (qx * qy - qw * qz)
    r13 = 2 * (qx * qz + qw * qy)
     
    # Second row
    r21 = 2 * (qx * qy + qw * qz)
    r22 = 2 * (qw * qw + qy * qy) - 1
    r23 = 2 * (qy * qz - qw * qx)
     
    # Third row
    r31 = 2 * (qx * qz - qw * qy)
    r32 = 2 * (qy * qz + qw * qx)
    r33 = 2 * (qw * qw + qz * qz) - 1

    rmat = np.array([
        [r11, r12, r13],
        [r21, r22, r23],
        [r31, r32, r33]
    ])

    return rmat

def rmat2quat(rmat):
    """ Convert rotation matrix to quaternion representation (Shepperds's method) """
    r11, r12, r13 = rmat[0].tolist()
    r21, r22, r23 = rmat[1].tolist()
    r31, r32, r33 = rmat[2].tolist()

    w_mag = sqrt((1 + r11 + r22 + r33) / 4)
    x_mag = sqrt((1 + r11 - r22 - r33) / 4)
    y_mag = sqrt((1 - r11 + r22 - r33) / 4)
    z_mag = sqrt((1 - r11 - r22 + r33) / 4)
    magnitudes = [w_mag, x_mag, y_mag, z_mag]

    match magnitudes.index(max(magnitudes)):
        case 0:
            qw = w_mag
            qx = (r32 - r23) / (4 * qw)
            qy = (r13 - r31) / (4 * qw)
            qz = (r21 - r12) / (4 * qw)
        case 1:
            qx = x_mag
            qw = (r32 - r23) / (4 * qx)
            qy = (r12 + r21) / (4 * qx)
            qz = (r13 + r31) / (4 * qx)
        case 2:
            qy = y_mag
            qw = (r13 - r31) / (4 * qy)
            qx = (r12 + r21) / (4 * qy)
            qz = (r23 + r32) / (4 * qy)
        case 3:
            qz = z_mag
            qw = (r21 - r12) / (4 * qz)
            qx = (r13 + r31) / (4 * qz)
            qy = (r23 + r32) / (4 * qz)

    quat = (qw, qx, qy, qz)

    return quat


def xyzw2wxyz(quat):
    """
    Quaternions in Unity are represented in (x,y,z,w) form; 'shifting' to (w,x,y,z)
    """
    assert len(quat) == 4
    if isinstance(quat, np.ndarray):
        return np.concatenate([quat[-1:], quat[:-1]])
    else:
        return quat[-1:] + quat[:-1]


def flip_position_y(pos):
    """
    Unity uses left-handed world coordinate where y-axis faces downward, which
    is in opposition to the right-handed coordinate (e.g. OpenCV, Open3D, COLMAP)
    where y-axis points downward. Account for the difference by flipping re. y-axis.
    """
    # Flip position by negating the y-entry
    return (pos[0], -pos[1], pos[2])
def flip_quaternion_y(quat):
    """ Analogous method for flipping rotation quaternion """
    return (quat[0], -quat[1], quat[2], -quat[3])       # Entry order: wxyz


def transformation_matrix(rotation, translation):
    """
    Compile rotation vector and translation vector into a single 4*4 matrix
    representing the transformation. Rotation may be in Rodrigues vector
    format or 3*3 matrix format.
    """
    rotation = np.asarray(rotation)
    translation = np.asarray(translation)

    # Convert into 3*3 rotation matrix...
    if rotation.shape == (4,):
        # From quaternion
        rotation = quat2rmat(rotation)
    elif rotation.shape == (3, 1):
        # From Rodrigues vector
        rotation = cv.Rodrigues(rotation)[0]

    # Make sure translation vector is in (3,1) shape
    if len(translation.shape) == 1:
        translation = translation[None].T
    if translation.shape == (1, 3):
        translation = translation.T

    # Collect into [[R|t],[0|1]] matrix (4*4)
    tr_mat = np.concatenate(
        [
            np.concatenate([rotation, translation], axis=1),
            np.array([[0.0, 0.0, 0.0, 1.0]])
        ]
    )

    return tr_mat


def masks_bounding_boxes(masks):
    """
    Axis-aligned bounding boxes (in xyxy format) from min/max indices for nonzero
    value in masks
    """
    boxes = []

    for msk in masks:
        nz_inds_y, nz_inds_x = msk.nonzero()
        x_min, x_max = nz_inds_x.min(), nz_inds_x.max()
        y_min, y_max = nz_inds_y.min(), nz_inds_y.max()
        boxes.append([x_min, y_min, x_max, y_max])

    return boxes


def mask_iou(masks_1, masks_2):
    """
    Compute pairwise mask-IoUs between two lists of segmentation masks (torchvision.ops
    only has box-IoU). Masks_1 & 2 are two list-likes of segmentation masks of same
    size ((N1, H, W), (N2, H, W)), returns N * M array of float values between 0 and 1.
    """
    intersections = [np.minimum(m1, m2) for m1, m2 in product(masks_1, masks_2)]
    unions = [np.maximum(m1, m2) for m1, m2 in product(masks_1, masks_2)]
    ious = np.array([i.sum() / u.sum() for i, u in zip(intersections, unions)])
    ious = ious.reshape(len(masks_1), len(masks_2))

    return ious


def mask_nms(masks, scores, iou_thres):
    """
    Non-maximum suppression by mask-IoU (torchvision.ops only has NMS based
    on box-IoU). Return list of indices of kept masks, sorted.
    """
    # Return value
    kept_indices = []

    # Index queue sorted by score
    queue = np.argsort(scores)

    while len(queue) > 0:
        # Pop index of highest score from queue and add to return list
        ind = queue[-1]
        queue = queue[:-1]
        kept_indices.append(ind)

        # Can break if no more remaining items in queue
        if len(queue) == 0: break

        # Compute mask IoUs between the popped mask vs. remaining masks
        ious = mask_iou(masks[ind, None], masks[queue])[0]

        # Filter out masks with IoUs higher than threshold
        queue = queue[ious < iou_thres]

    return kept_indices


def masked_patch_match(features_1, features_2, mask_1, width_1, width_2):
    """
    Matching patch-level features between two image patches by minimum weight
    matching, for finding correspondences (point pairs) between the patches.
    Applicable where the first patch has corresponding 'ground-truth' mask
    info (provided as `masks_1` parameter), which serves to filter out invalid
    correspondences.

    Assume input features and mask are flattened so that their shape is (H*W, D).
    """
    assert features_1.shape[-1] == features_2.shape[-1]

    # Nonzero indices of the provided (flattened) mask
    nonzero_inds_1 = mask_1.nonzero()[0]

    # Compute cosine similarities between patches
    normalize = lambda fts: fts / np.linalg.norm(fts, axis=1, keepdims=True)
    features_nrm_1 = normalize(features_1)
    features_nrm_2 = normalize(features_2)
    S = features_nrm_1 @ features_nrm_2.T
    # Forward matching
    match_forward = linear_sum_assignment(S[mask_1], maximize=True)
    # Reverse matching
    match_reverse = linear_sum_assignment(S.T[match_forward[1]], maximize=True)
    # Mask filtering
    retain_inds = np.isin(match_reverse[1], nonzero_inds_1)
    # Fetch and unflatten matched patch indices
    matched_patches_1 = np.stack([
        i_1 := nonzero_inds_1[match_forward[0][retain_inds]],
        i_1 % width_1, i_1 // width_1,
    ], axis=1)
    matched_patches_2 = np.stack([
        i_2 := match_forward[1][retain_inds],
        i_2 % width_2, i_2 // width_2,
    ], axis=1)

    return matched_patches_1, matched_patches_2
