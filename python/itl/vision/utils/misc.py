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
from numpy.linalg import norm
from scipy.optimize import linear_sum_assignment
from skimage.morphology import dilation, disk
from sklearn.metrics import pairwise_distances


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

    # Values for determining the first quaternion element to compute
    selectors = [r11 + r22 + r33, r11, r22, r33]

    match selectors.index(max(selectors)):
        case 0:
            qw = sqrt((1 + r11 + r22 + r33) / 4)
            qx = (r32 - r23) / (4 * qw)
            qy = (r13 - r31) / (4 * qw)
            qz = (r21 - r12) / (4 * qw)
        case 1:
            qx = sqrt((1 + r11 - r22 - r33) / 4)
            qw = (r32 - r23) / (4 * qx)
            qy = (r12 + r21) / (4 * qx)
            qz = (r13 + r31) / (4 * qx)
        case 2:
            qy = sqrt((1 - r11 + r22 - r33) / 4)
            qw = (r13 - r31) / (4 * qy)
            qx = (r12 + r21) / (4 * qy)
            qz = (r23 + r32) / (4 * qy)
        case 3:
            qz = sqrt((1 - r11 - r22 + r33) / 4)
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


## Left as legacy code, commented out, as I've decided to simply use ground-truth
## object poses passed from Unity environment
# def pose_estimation_with_mask(image, mask, vis_model, structure_3d, cam_coeffs):
#     """
#     Estimate 3D pose of an object whose 3D structure (as point cloud with feature
#     descriptors) is known, where estimation is guided and evaluated by the provided
#     segmentation mask of the object. Returns estimation results with confidence
#     scores per viewpoint as result.
#     """
#     # Unpack arguments
#     cam_K, distortion_coeffs = cam_coeffs
#     points, views, descriptors = structure_3d

#     # Shortcut helper for normalizing set of features by L2 norm, so that cosine
#     # similarities can be computed
#     normalize = lambda fts: fts / norm(fts, axis=1, keepdims=True)

#     results_per_view = []           # Return value

#     # Extract patch-level features as guided by masks
#     # mask = dilation(mask, footprint=disk(3))
#     zoomed_image, zoomed_msk, crop_dim = crop_images_by_masks(
#         { 0: image }, [mask]
#     )
#     patch_features, lr_msk, lr_dim = vis_model.lr_features_from_masks(
#         zoomed_image, zoomed_msk, 1200, 2
#     )
#     D = patch_features[0].shape[-1]
#     (cr_x, _, cr_y, _), (lr_w, lr_h) = crop_dim[0], lr_dim[0]
#     zoomed_image = zoomed_image[0]; lr_msk = lr_msk[0]

#     msk_flattened = lr_msk.reshape(-1)
#     nonzero_inds = msk_flattened.nonzero()[0]

#     # Scale ratio between zoomed image vs. low-res feature map
#     x_ratio = zoomed_image.width / lr_w
#     y_ratio = zoomed_image.height / lr_h

#     # Compare against downsampled point descriptors stored in XB, per each view
#     pth_fh_flattened = patch_features[0].cpu().numpy().reshape(-1, D)
#     for vi, view_info in views.items():
#         # Fetch descriptors of visible points
#         visible_pts_sorted = sorted(view_info["visible_points"])
#         visible_pts_features = np.stack(
#             [descriptors[pi][vi] for pi in visible_pts_sorted]
#         )

#         # Compute cosine similarities between patches
#         features_nrm_1 = normalize(view_info["pca"].transform(pth_fh_flattened))
#         features_nrm_2 = normalize(visible_pts_features)
#         S = features_nrm_1 @ features_nrm_2.T
#         S = (S + 1) / 2

#         # Obtaining (u,v)-coordinates of projected (downsampled) points at the
#         # viewpoint pose, needed for proximity score computation
#         rmat_view = quat2rmat(view_info["cam_quaternion"])
#         tvec_view = view_info["cam_position"]
#         proj_at_view = cv.projectPoints(
#             points,
#             cv.Rodrigues(rmat_view)[0], tvec_view,
#             cam_K, distortion_coeffs
#         )[0][:,0,:]
#         u_min, u_max = proj_at_view[:,0].min(), proj_at_view[:,0].max()
#         v_min, v_max = proj_at_view[:,1].min(), proj_at_view[:,1].max()
#         proj_w = u_max - u_min; proj_h = v_max - v_min
#         # Scale and align point coordinates
#         proj_aligned = proj_at_view[visible_pts_sorted]
#         proj_aligned[:,0] -= u_min; proj_aligned[:,1] -= v_min
#         # Scale so that the bounding box would encase the provided object
#         # mask (object may be occluded while projection is never occluded)
#         obj_box = masks_bounding_boxes([mask])[0]
#         obj_w = obj_box[2] - obj_box[0]; obj_h = obj_box[3] - obj_box[1]
#         obj_cu = (obj_box[0]+obj_box[2]) / 2; obj_cv = (obj_box[3]+obj_box[1]) / 2
#         scale_ratio = min(proj_w / obj_w, proj_h / obj_h)
#         proj_aligned /= scale_ratio
#         # Align by box center coordinates
#         proj_aligned[:,0] += obj_cu - (proj_w / 2) / scale_ratio
#         proj_aligned[:,1] += obj_cv - (proj_h / 2) / scale_ratio

#         # Proximity scores (w.r.t. to downsampled point projection) computed
#         # with RBF kernel; for giving slight advantages to pixels close to
#         # initial guess projections
#         uv_coords = np.stack([
#             np.tile((np.arange(lr_w)*x_ratio + cr_x)[None], [lr_h, 1]),
#             np.tile((np.arange(lr_h)*y_ratio + cr_y)[:,None], [1, lr_w])
#         ], axis=-1)
#         sigma = min(zoomed_image.width, zoomed_image.height)
#         proximity = norm(
#             uv_coords[:,:,None] - proj_aligned[None,None], axis=-1
#         )
#         proximity = np.exp(-np.square(proximity) / (2 * (sigma ** 2)))

#         # Forward matching
#         agg_scores = S + 0.5 * proximity.reshape(-1, len(visible_pts_sorted))
#         match_forward = linear_sum_assignment(
#             agg_scores[msk_flattened], maximize=True
#         )

#         points_2d = [
#             (i % lr_w, i // lr_w)
#             for i in nonzero_inds[match_forward[0]]
#         ]
#         points_2d = np.array([
#             (cr_x + lr_x * x_ratio, cr_y + lr_y * y_ratio)
#             for lr_x, lr_y in points_2d
#         ])
#         points_3d = np.array([
#             points[visible_pts_sorted[i]]
#             for i in match_forward[1]
#         ])

#         # Pose estimation by PnP with USAC (MAGSAC)
#         output_valid, rvec, tvec, _ = cv.solvePnPRansac(
#             points_3d, points_2d, cam_K, distortion_coeffs,
#             flags=cv.USAC_MAGSAC
#         )
#         assert output_valid

#         # Evaluate estimated pose by obtaining mean similarity scores at
#         # reprojected downsampled points
#         estim_reprojections = cv.projectPoints(
#             points, rvec, tvec, cam_K, distortion_coeffs
#         )[0][:,0,:]
#         visible_reprojections = np.array([
#             estim_reprojections[visible_pts_sorted[i]]
#             for i in match_forward[1]
#         ])
#         dists_to_reprojs = norm(
#             uv_coords[:,:,None] - visible_reprojections[None,None], axis=-1
#         )
#         dists_to_reprojs = dists_to_reprojs.reshape(-1, len(visible_reprojections))
#         reproj_coords = np.unravel_index(
#             dists_to_reprojs.argmin(axis=0), (lr_h, lr_w)
#         )
#         reproj_score_inds = reproj_coords + (np.arange(len(visible_pts_sorted)),)
#         reproj_scores = S.reshape(lr_h, lr_w, -1)[reproj_score_inds]
#         # Only consider points that belong to the primary cluster; accounts
#         # for occlusions
#         within_mask_inds = [
#             lr_msk[(reproj_coords[0][i], reproj_coords[1][i])]
#             for i in range(len(visible_pts_sorted))
#         ]
#         reproj_scores = reproj_scores[within_mask_inds]

#         # Also evaluate by overlap between provided object mask vs. mask made
#         # from reprojected visible points dilated with disk-shaped footprints.
#         # Size of disk determined by mean nearest distances between reprojected
#         # visible points.
#         vps_prm_reproj = estim_reprojections[visible_pts_sorted]
#         pdists = pairwise_distances(vps_prm_reproj, vps_prm_reproj)
#         pdists[np.diag_indices(len(pdists))] = float("inf")
#         median_nn_dist = np.median(pdists.min(axis=0))
#             # Occasionally some reprojected points are placed very far
#             # from the rest... Using median instead of mean to eliminate
#             # effects by outliers
#         proj_x, proj_y = vps_prm_reproj.round().astype(np.int64).T
#         proj_msk = np.zeros_like(mask)
#         valid_inds = valid_inds = np.logical_and(
#             np.logical_and(0 <= proj_y, proj_y < mask.shape[0]),
#             np.logical_and(0 <= proj_x, proj_x < mask.shape[1])
#         )
#         proj_msk[proj_y[valid_inds], proj_x[valid_inds]] = True
#         disk_size = min(
#             max(2*median_nn_dist, 1), min(obj_w, obj_h) / 10
#         )       # Max cap disk size with 0.1 * min(mask_w, mask_h)
#         proj_msk = dilation(proj_msk, footprint=disk(disk_size))
#         mask_overlap = mask_iou([mask], [proj_msk])[0][0]

#         results_per_view.append(
#             ((rvec, tvec), reproj_scores.mean().item(), mask_overlap)
#         )

#     return results_per_view
