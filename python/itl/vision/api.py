"""
Vision processing module API that exposes the high-level functionalities required
by the ITL agent inference (concept classification of object instances in image,
concept instance search in image). Implemented by building upon the pretrained
Segment Anything Model (SAM).
"""
import os
import shutil
from PIL import Image
from itertools import product
from collections import defaultdict

import torch
import cv2 as cv
import numpy as np
import open3d as o3d
from pycocotools import mask
from sklearn.decomposition import PCA

from .modeling import VisualSceneAnalyzer
from .utils import (
    crop_images_by_masks, quat2rmat, masked_patch_match, xyzw2wxyz
)
from .utils.colmap.reconstruction import reconstruct_with_known_poses


DEF_CON = 0.6       # Default confidence value in absence of binary concept classifier

class VisionModule:

    def __init__(self, cfg):
        self.cfg = cfg

        self.scene = None
        self.latest_inputs = []         # Latest raw inputs

        # Inventory of distinct visual concepts that the module (and thus the agent
        # equipped with this module) is aware of, one per concept category. Right now
        # I cannot think of any specific type of information that has to be stored in
        # this module (exemplars are stored in long term memory), so let's just keep
        # only integer sizes of inventories for now...
        self.inventories = VisualConceptInventory()

        self.model = VisualSceneAnalyzer(self.cfg)
        
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

        self.camera_intrinsics = None

        self.confusions = set()

    def predict(self, image, exemplars, reclassify=False, masks=None, specs=None):
        """
        Model inference in either one of four modes:
            (1) full scene graph generation mode, where the module is only given
                an image and needs to return its estimation of the full scene
                graph for the input
            (2) concept reclassification mode, where the scene objects and their
                visual embeddings are preserved intact and only the concept tests
                are run again (most likely called when agent exemplar base is updated)
            (3) instance classification mode, where a number of masks are given
                along with the image and category predictions are made for only
                those instances
            (4) instance search mode, where a specification is provided in the
                form of FOL formula with a variable and best fitting instance(s)
                should be searched

        (3) and (4) are 'incremental' in the sense that they should add to an existing
        scene graph which is already generated with some previous execution of this
        method.
        
        Provide reclassify=True to run in (2) mode. Provide masks arg to run in (3) mode,
        or spec arg to run in (4) mode. All prediction modes are mutually exclusive in
        the sense that only one of their corresponding 'flags' should be active.
        """
        masks_provided = masks is not None
        specs_provided = specs is not None
        ensemble = not (reclassify or masks_provided or specs_provided)

        # Image must be provided for ensemble prediction
        if ensemble: assert image is not None

        if image is not None:
            if isinstance(image, str):
                image = Image.open(image)
            else:
                assert isinstance(image, Image.Image)

        self.model.eval()
        with torch.no_grad():
            # Prediction modes
            if ensemble:
                # Full (ensemble) prediction. Obtain output from 'segment everything'
                # pipeline, then filter overlapping masks so that only minimally
                # overlapping masks remain (i.e., so that the image is literally
                # 'segmented')
                vis_embs, masks_out = self.model(image)

                # Heuristic: Keep adding to the list of potential objects from
                # the top of the list, handling overlaps appropriately. SAM pipeline
                # tends to prefer single object wholes vs. parts or multiple objects,
                # though exceptions occasionally arise and should be aptly handled.
                # 
                # Specifically, if a mask candidate is neither a 'submask' or a
                # 'supermask' of an existing mask, simply add to the candidate list.
                # If a submask, ignore; this will be fine in most cases. If a supermask,
                # replace overlapping submasks only if it leads to recognition of object
                # class concept with stronger confidence.
                filtered_masks = []
                for f_vec, msk in zip(vis_embs, masks_out):
                    # Add to filtered list only if it is not a submask of any existing
                    # mask in the list
                    if len(filtered_masks) == 0:
                        # First entry, just append
                        filtered_masks.append((f_vec, msk))
                        continue

                    intersections = [
                        np.minimum(msk, f_msk) for _, f_msk in filtered_masks
                    ]
                    submask_scores = np.array([
                        i_msk.sum() / msk.sum() for i_msk in intersections
                    ])
                    supermask_scores = np.array([
                        i_msk.sum() / f_msk.sum()
                        for i_msk, (_, f_msk) in zip(intersections, filtered_masks)
                    ])
                    if submask_scores.max() > 0.7:
                        # Submask of an existing mask; okay to ignore in most cases
                        pass
                    elif supermask_scores.max() > 0.7:
                        # Supermask of an existing mask; 'semantic test' needed to
                        # see if the supermask yields stronger confidence (across
                        # all concepts) than the submasks it covers
                        submask_inds = (supermask_scores > 0.7).nonzero()[0]

                        # Few-shot prediction on current mask
                        fs_pred = self.fs_conc_pred(exemplars, f_vec, "pcls")
                        # Few-shot predictions on submasks
                        sm_fs_preds = np.stack([
                            self.fs_conc_pred(exemplars, filtered_masks[i][0], "pcls")
                            for i in submask_inds
                        ])

                        if len(fs_pred) == 0:
                            # Just add if visual concept inventory is empty
                            filtered_masks.append((f_vec, msk))
                        elif fs_pred.max() > sm_fs_preds.max():
                            # Replace submasks with supermask only if max prediction
                            # score is higher for the supermask than the highest for
                            # the submasks
                            filtered_masks = [
                                entry for i, entry in enumerate(filtered_masks)
                                if i not in submask_inds
                            ]
                            filtered_masks.append((f_vec, msk))
                    else:
                        # No overlap, simply add
                        filtered_masks.append((f_vec, msk))

                # Newly compose a scene graph with the output; filter patches to
                # leave top-k detections
                self.scene = {
                    f"o{i}": {
                        "vis_emb": filtered_masks[i][0],
                        "pred_mask": filtered_masks[i][1],
                        "pred_cls": self.fs_conc_pred(exemplars, vis_embs[i], "pcls"),
                        "pred_rel": {
                            f"o{j}": np.zeros(self.inventories.prel)
                            for j in range(len(filtered_masks)) if i != j
                        },
                        "exemplar_ind": None       # Store exemplar storage index, if applicable
                    }
                    for i in range(len(filtered_masks))
                }

                for oi, obj_i in self.scene.items():
                    oi_msk = obj_i["pred_mask"]

                    # Relation concepts (Only for "have" concept, manually determined
                    # by the geomtrics of the bounding boxes; note that this is quite
                    # an abstraction. In distant future, relation concepts may also be
                    # open-vocabulary and neurally predicted...)
                    for oj, obj_j in self.scene.items():
                        if oi==oj: continue     # Dismiss self-self object pairs
                        oj_msk = obj_j["pred_mask"]

                        intersection_A = np.minimum(oi_msk, oj_msk).sum()
                        mask2_A = oj_msk.sum()

                        obj_i["pred_rel"][oj][0] = intersection_A / mask2_A

            elif reclassify:
                # Concept reclassification with the same set of objects
                for obj_i in self.scene.values():
                    vis_emb = obj_i["vis_emb"]
                    obj_i["pred_cls"] = self.fs_conc_pred(exemplars, vis_emb, "pcls")

            else:
                # Incremental scene graph expansion
                if masks_provided:
                    # Instance classification mode
                    incr_preds = self.model(image, masks=list(masks.values()))

                else:
                    # Instance search mode
                    assert specs_provided
                    incr_preds = self._instance_search(image, exemplars, specs)

                # Selecting names of new objects to be added to the scene
                if masks_provided:
                    # Already provided with appropriate object identifier
                    new_objs = list(masks.keys())
                else:
                    # Come up with new object identifiers for valid search matches
                    incr_masks_out = incr_preds[1]

                    new_objs = []; ind_offset = 0
                    for msk in incr_masks_out:
                        if msk is None:
                            # Null match
                            new_objs.append(None)
                        else:
                            new_objs.append(f"o{len(self.scene)+ind_offset}")
                            ind_offset += 1

                # Update visual scene
                self._incremental_scene_update(incr_preds, new_objs, exemplars)

    def fs_conc_pred(self, exemplars, emb, conc_type):
        """
        Helper method factored out for few-shot concept probability estimation
        """
        bin_clfs = exemplars.binary_classifiers_2d[conc_type]

        conc_inventory_count = getattr(self.inventories, conc_type)
        if conc_inventory_count > 0:
            # Non-empty concept inventory, most cases
            predictions = []
            for ci in range(conc_inventory_count):
                if ci in bin_clfs:
                    if bin_clfs[ci] is not None:
                        # Binary classifier induced from pos/neg exemplars exists
                        clf = bin_clfs[ci]
                        pred = clf.predict_proba(emb[None])[0]
                        pred = max(min(pred[1], 0.99), 0.01)    # Soften absolute certainty)
                        predictions.append(pred)
                    else:
                        # No binary classifier exists due to lack of either positive
                        # or negative exemplars, fall back to default estimation
                        predictions.append(DEF_CON)
                else:
                    # Not a concept to be visually recognized
                    predictions.append(0.0)
            return np.stack(predictions)
        else:
            # Empty concept inventory, likely at the very beginning of training
            # an agent from scratch,
            return np.empty(0, dtype=np.float32)

    def _instance_search(self, image, exemplars, search_specs):
        """
        Helper method factored out for running model in incremental search mode
        """
        # Mapping between existing entity IDs and their numeric indexing
        exs_idx_map = { i: ent for i, ent in enumerate(self.scene) }
        exs_idx_map_inv = { ent: i for i, ent in enumerate(self.scene) }

        # Return values
        incr_vis_embs = []
        incr_masks_out = []
        incr_scores = []

        for s_vars, description, pred_glossary in search_specs:
            # Prepare search conditions to feed into model.forward()
            search_conds = []
            for d_lit in description:
                if d_lit.name.startswith("disj"):
                    # A predicate standing for a disjunction of elementary concepts
                    # (as listed in `pred_glossary`), fetch all positive exemplar sets
                    # and binary classifiers for each disjunct concept
                    disj_cond = []
                    for conc in pred_glossary[d_lit.name][1]:
                        conc_type, conc_ind = conc.split("_")
                        conc_ind = int(conc_ind)

                        match conc_type:
                            case "pcls":
                                # Fetch set of positive exemplars
                                pos_exs_inds = exemplars.object_2d_pos[conc_type][conc_ind]
                                pos_exs_info = [
                                    (
                                        exemplars.scenes_2d[scene_id][0],
                                        mask.decode(
                                            exemplars.scenes_2d[scene_id][1][obj_id]["mask"]
                                        ).astype(bool),
                                        exemplars.scenes_2d[scene_id][1][obj_id]["f_vec"]
                                    )
                                    for scene_id, obj_id in pos_exs_inds
                                ]
                                bin_clf = exemplars.binary_classifiers_2d[conc_type][conc_ind]
                                disj_cond.append((pos_exs_info, bin_clf))

                            case "prel":
                                # Relations are not neurally predicted, nor used for search
                                # (for now, at least...)
                                continue

                            case _:
                                raise ValueError("Invalid concept type")

                    search_conds.append(disj_cond)

                else:
                    # Single elementary predicate, fetch positive exemplars for the
                    # corresponding concept
                    conc_type, conc_ind = d_lit.name.split("_")
                    conc_ind = int(conc_ind)

                    match conc_type:
                        case "pcls":
                            # Fetch set of positive exemplars
                            pos_exs_inds = exemplars.object_2d_pos[conc_type][conc_ind]
                            pos_exs_info = [
                                (
                                    exemplars.scenes_2d[scene_id][0],
                                    mask.decode(
                                        exemplars.scenes_2d[scene_id][1][obj_id]["mask"]
                                    ).astype(bool),
                                    exemplars.scenes_2d[scene_id][1][obj_id]["f_vec"]
                                )
                                for scene_id, obj_id in pos_exs_inds
                            ]
                            bin_clf = exemplars.binary_classifiers_2d[conc_type][conc_ind]
                            search_conds.append([(pos_exs_info, bin_clf)])

                        case "prel":
                            # Relations are not neurally predicted, nor used for search
                            # (for now, at least...)
                            continue

                        case _:
                            raise ValueError("Invalid concept type")

            # Run search conditioned on the exemplars
            vis_embs, masks_out, scores = self.model(image, search_conds=search_conds)

            if len(masks_out) == 0:
                # Search didn't return any match
                incr_vis_embs.append(None)
                incr_masks_out.append(None)
                incr_scores.append(None)
                continue

            # If len(masks_out) > 0, search returned some matches

            # Test each candidate to select the one that's most compatible
            # to the search spec
            agg_compatibility_scores = torch.ones(
                len(masks_out), device=self.model.device
            )
            for d_lit in description:
                if d_lit.name.startswith("disj_"):
                    # Use the returned scores as compatibility scores
                    comp_scores = torch.tensor(scores, device=self.model.device)

                else:
                    conc_type, conc_ind = d_lit.name.split("_")
                    conc_ind = int(conc_ind)

                    match conc_type:
                        case "pcls":
                            # Use the returned scores as compatibility scores
                            comp_scores = torch.tensor(scores, device=self.model.device)

                        case "prel":
                            # Compatibility scores geometric relations, which are not neurally
                            # predicted int the current module implementation

                            # Cannot process relations other than "have" for now...
                            assert conc_ind == 0

                            # Cannot process search specs with more than one variables for
                            # now (not planning to address that for a good while!)
                            assert len(s_vars) == 1

                            # Handles to literal args; either search target variable or
                            # previously identified entity
                            arg_handles = [
                                ("v", s_vars.index(arg[0]))
                                    if arg[0] in s_vars
                                    else ("e", exs_idx_map_inv[arg[0]])
                                for arg in d_lit.args
                            ]

                            # Mask areas for all candidates
                            masks_out_A = masks_out.sum(axis=(-2,-1))

                            # Fetch bbox of reference entity, against which bbox area
                            # ratios will be calculated among candidates
                            reference_ent = [
                                arg_ind for arg_type, arg_ind in arg_handles
                                if arg_type=="e"
                            ][0]
                            reference_ent = exs_idx_map[reference_ent]
                            reference_mask = self.scene[reference_ent]["pred_mask"]

                            # Compute area ratio between the reference mask and all proposals
                            intersections = masks_out * reference_mask[None]
                            intersections_A = intersections.sum(axis=(-2,-1))

                            comp_scores = torch.tensor(
                                intersections_A / masks_out_A, device=self.model.device
                            )

                        case _:
                            raise ValueError("Invalid concept type")

                # Update aggregate compatibility score; using min function
                # as the t-norm (other options: product, ...)
                agg_compatibility_scores = torch.minimum(
                    agg_compatibility_scores, comp_scores
                )

            # Finally choose and keep the best search output
            best_match_ind = agg_compatibility_scores.max(dim=0).indices
            incr_vis_embs.append(vis_embs[best_match_ind])
            incr_masks_out.append(masks_out[best_match_ind])
            incr_scores.append(scores[best_match_ind])

        return incr_vis_embs, incr_masks_out, incr_scores

    def _incremental_scene_update(self, incr_preds, new_objs, exemplars):
        """
        Helper method factored out updating current scene with new incrementally
        predicted instances (by masks or search specs)
        """
        incr_vis_embs = incr_preds[0]
        incr_masks_out = incr_preds[1]

        # Incrementally update the existing scene graph with the output with the
        # detections best complying with the conditions provided
        if self.scene is None:
            self.scene = {}         # Initialize a new scene
            existing_objs = []
        else:
            existing_objs = [
                oi for oi, obj in self.scene.items() if "pred_mask" in obj
            ]

        for oi, oj in product(existing_objs, new_objs):
            # Add new relation score slots for existing objects
            self.scene[oi]["pred_rel"][oj] = np.zeros(self.inventories.prel)

        for oi, vis_emb, msk in zip(new_objs, incr_vis_embs, incr_masks_out):
            # Pass null entry
            if oi is None: continue

            # Register new objects into the existing scene
            self.scene[oi] = {
                "vis_emb": vis_emb,
                "pred_mask": msk,
                "pred_cls": self.fs_conc_pred(exemplars, vis_emb, "pcls"),
                "pred_rel": {
                    **{
                        oj: np.zeros(self.inventories.prel)
                        for oj in existing_objs
                    },
                    **{
                        oj: np.zeros(self.inventories.prel)
                        for oj in new_objs if oi != oj
                    }
                },
                "exemplar_ind": None
            }

        for oi in new_objs:
            # Pass null entry
            if oi is None: continue

            oi_msk = self.scene[oi]["pred_mask"]

            # Relation concepts (Within new detections)
            for oj in new_objs:
                # Pass null entry
                if oj is None: continue

                if oi==oj: continue     # Dismiss self-self object pairs
                oj_msk = self.scene[oj]["pred_mask"]

                intersection_A = np.minimum(oi_msk, oj_msk).sum()
                mask2_A = oj_msk.sum()

                self.scene[oi]["pred_rel"][oj][0] = intersection_A / mask2_A

            # Relation concepts (Between existing detections)
            for oj in existing_objs:
                oj_msk = self.scene[oj]["pred_mask"]

                intersection_A = np.minimum(oi_msk, oj_msk).sum()
                mask1_A = oi_msk.sum()
                mask2_A = oj_msk.sum()

                self.scene[oi]["pred_rel"][oj][0] = intersection_A / mask2_A
                self.scene[oj]["pred_rel"][oi][0] = intersection_A / mask1_A

    def add_concept(self, conc_type):
        """
        Register a novel visual concept to the model, expanding the concept inventory
        of corresponding category type (pcls, prel). Note that visual concepts are not
        inseparably attached to some linguistic symbols; such connections are rather
        incidental and should be established independently (consider synonyms, homonyms).
        Plus, this should allow more flexibility for, say, multilingual agents, though
        there is no plan to address that for now...

        Returns the index of the newly added concept.
        """
        C = getattr(self.inventories, conc_type)
        setattr(self.inventories, conc_type, C+1)
        return C

    def calibrate_camera(self, images, points_3d):
        """
        Compute camera intrinsic parameters from a set of chessboard images and 3d
        coordinates of chessboard corners, store results in module
        """
        points_2d = []
        refinement_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        for img in images:
            # Find corners from image, then refine the 2d coordinates with grayscale version
            gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            found, corners = cv.findChessboardCorners(img, (7, 6), None)
            if not found:
                # Generally happens if the chessboard texture doesn't have 'border area'
                raise ValueError("Calibration chessboard pattern not found")
            corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), refinement_criteria)

            points_2d.append(corners)

        # Obtain camera calibration results and store to module
        _, cam_K, distortion_coeffs, _, _ = cv.calibrateCamera(
            points_3d, points_2d, gray.shape[::-1], None, None
        )
        self.camera_intrinsics = (cam_K, distortion_coeffs)

    def reconstruct_3d_structure(
        self, images, masks, viewpoint_poses, con_graph, store_vp_inds,
        resolution_multiplier=1
    ):
        """
        Given lists of images & binary masks of an object viewed from different
        viewpoints and the poses of the viewpoints, estimate the (approximate) 3D
        structure of the object, represented as a point cloud. Also provided with
        an undirected graph that represents pairs of images to be cross-referenced.

        Steps: (1) Obtain patch-level features from the feature extraction module,
        (2) obtain matches between patches on feature basis, (3) invoke pycolmap's
        geometric verification & point triangulation methods based on the matches.
        """
        assert len(images) == len(masks) == len(viewpoint_poses) == len(con_graph.nodes)

        # Crop images & masks according to masks, effectively zooming in
        zoomed_images, zoomed_masks, crop_dims = crop_images_by_masks(images, masks)

        self.model.eval()
        with torch.no_grad():
            # Obtain (flattened) patch-level features and corresponding masks extracted from
            # the zoomed images
            lr_mask_area = 800
            patch_features, lr_masks, lr_dims = self.model.lr_features_from_masks(
                zoomed_images, zoomed_masks, lr_mask_area, resolution_multiplier
            )
            D = patch_features[0].shape[-1]

        # Storing full 2d-visual features from deep vision model in XB would
        # require too much space; apply dimensionality reduction by PCA (specific
        # to each view) and store analysis results
        dim_reducs = {
            i: PCA(n_components=128).fit(pth_ft[0][lr_msk].cpu())
            for i, (pth_ft, lr_msk) in enumerate(zip(patch_features, lr_masks))
            if i in store_vp_inds
        }

        # Flatten patch features
        patch_features = [
            pth_ft.cpu().numpy().reshape(-1, D) for pth_ft in patch_features
        ]

        # Iterating over all edges (image pairs), starting from ones involving lowest-degree
        # nodes, until no unprocessed edges are left
        node_unprocessed_neighbors = { n: set(con_graph.adj[n]) for n in con_graph.nodes }
        rough_matches = {}        # To store initial inter-patch matches, verified later
        point_coords = defaultdict(dict)
        while not all(len(neighbors)==0 for neighbors in node_unprocessed_neighbors.values()):
            # Fetch a node with the currently lowest degree w.r.t. unprocessed neighbors
            lowest_degree_nodes = sorted(
                [n for n in con_graph.nodes if len(node_unprocessed_neighbors[n]) > 0],
                key=lambda x: len(node_unprocessed_neighbors[x])
            )
            u = lowest_degree_nodes[0]

            # Iterate over unprocessed connected neighbors
            processed_edges = set()
            for v in node_unprocessed_neighbors[u]:
                # Bidirectional matches between masked patches indexed by u vs. v
                matched_patches_u, matched_patches_v = masked_patch_match(
                    patch_features[u], patch_features[v],
                    lr_masks[u].reshape(-1), lr_dims[u][0], lr_dims[v][0]
                )

                # Record matches
                x_ratio_u = zoomed_images[u].width / lr_dims[u][0]
                y_ratio_u = zoomed_images[u].height / lr_dims[u][1]
                x_ratio_v = zoomed_images[v].width / lr_dims[v][0]
                y_ratio_v = zoomed_images[v].height / lr_dims[v][1]
                rough_matches[(u, v)] = []
                for (i_u, x_u, y_u), (i_v, x_v, y_v) in zip(matched_patches_u, matched_patches_v):
                    rough_matches[(u, v)].append((i_u, i_v))
                    point_coords[u][i_u] = np.array([[
                        crop_dims[u][0] + x_u * x_ratio_u,
                        crop_dims[u][2] + y_u * y_ratio_u
                    ]])
                    point_coords[v][i_v] = np.array([[
                        crop_dims[v][0] + x_v * x_ratio_v,
                        crop_dims[v][2] + y_v * y_ratio_v
                    ]])

                processed_edges.add((u, v))

            # Check off processed edges                
            for u, v in processed_edges:
                node_unprocessed_neighbors[u].remove(v)
                node_unprocessed_neighbors[v].remove(u)

        assert self.camera_intrinsics is not None
        reconstruction, colmap2patch_map = reconstruct_with_known_poses(
            point_coords, rough_matches, viewpoint_poses, self.camera_intrinsics,
            images, self.cfg.paths.outputs_dir
        )
        # COLMAP temporary working directory cleanup
        colmap_path = os.path.join(self.cfg.paths.outputs_dir, "colmap")
        shutil.rmtree(colmap_path)

        # Rough initial filtering by estimated reconstruction error
        points3d = [
            (pi, pt) for pi, pt in reconstruction.points3D.items()
            if pt.error < 5
        ]

        # Collect points and filter by 2D reprojection vs. masks
        for msk, (quat, trns) in zip(masks.values(), viewpoint_poses):
            # Project to 2D image coordinate from this viewing angle
            cam_K, distortion_coeffs = self.camera_intrinsics
            rmat = quat2rmat(quat)
            projections = cv.projectPoints(
                np.stack([pt.xyz for _, pt in points3d]),
                cv.Rodrigues(rmat)[0], np.array(trns),
                cam_K, distortion_coeffs
            )[0]
            projections = [prj[0] for prj in projections]

            # We have ground truth segmentation masks, filter out points whose
            # 2D reprojections fall out of the masks
            points3d = [
                (pi, pt) for (pi, pt), (u, v) in zip(points3d, projections)
                if (0 <= round(u) < msk.shape[1]) and (0 <= round(v) < msk.shape[0]) \
                    and msk[round(v), round(u)]
            ]
        points3d = dict(points3d)

        # Organize reconstruction output into return values

        # Format as Open3d point cloud data structure
        reindex_map = dict(enumerate(sorted(points3d)))
        reindex_map_inv = { v: k for k, v in reindex_map.items() }
        reindexed_points = np.array([
            points3d[reindex_map[i]].xyz for i in range(len(reindex_map))
        ])
        point_cloud = o3d.geometry.PointCloud(
            points=o3d.utility.Vector3dVector(reindexed_points)
        )

        # Collect view info and point descriptors
        views = {}
        descriptors_full = defaultdict(dict); descriptors_reduc = defaultdict(dict)
        for i, (_, img) in enumerate(sorted(reconstruction.images.items())):
            if i not in store_vp_inds: continue
            st_i = store_vp_inds.index(i)

            views[st_i] = {
                "cam_quaternion": xyzw2wxyz(img.cam_from_world.rotation.quat),
                "cam_position": img.cam_from_world.translation,
                "visible_points": set(),
                "pca": dim_reducs[i]
            }

            # Fetch 2d visual features (while registering as visible at each view)
            for p2i, p2d in enumerate(img.points2D):
                p3i = p2d.point3D_id
                if p3i not in reindex_map_inv: continue     # Ignore invalid p2ds
                p3i_reind = reindex_map_inv[p3i]
                views[st_i]["visible_points"].add(p3i_reind)

                patch_index = colmap2patch_map[i][p2i]
                descriptors_full[p3i_reind][i] = patch_features[i][patch_index]

            # Applying dimensionality reduction to the descriptors
            for p3i, per_view in descriptors_full.items():
                if i in per_view:
                    fvec_reduc = views[st_i]["pca"].transform(per_view[i][None])[0]
                    descriptors_reduc[p3i][st_i] = fvec_reduc

        descriptors = dict(descriptors_reduc)

        return point_cloud, views, descriptors


class VisualConceptInventory:
    def __init__(self):
        # Inventory of (visually perceivable) relation concept is a fixed singleton
        # set, containing "have"
        self.pcls = 0
        self.prel = 1
