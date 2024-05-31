"""
Miscellaneous utility methods that don't classify into other files in utils
"""
from itertools import product
import numpy as np


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
        left = max(0, int(x_min - crop_width*0.1))
        right = min(img.width, int(x_max + crop_width*0.1))
        top = max(0, int(y_min - crop_height*0.1))
        bottom = min(img.height, int(y_max + crop_height*0.1))

        # Append values to return lists
        cropped_images.append(img.crop((left, top, right, bottom)))
        cropped_masks.append(msk[top:bottom, left:right])
        dimensions.append((left, right, top, bottom))

    return cropped_images, cropped_masks, dimensions


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
