"""
3D structure reconstruction by COLMAP library for Structure-from-Motion (SfM)
with known camera poses. Factored out as it got too lengthy to be included in
invoking methods.
"""
import os
import shutil
import pathlib
from collections import defaultdict

import pycolmap
pycolmap.logging.minloglevel = pycolmap.logging.ERROR
import numpy as np

from .database import COLMAPDatabase


def reconstruct_with_known_poses(
        points_in_views, correspondences, viewpoint_poses, camera_intrinsics,
        images, working_directory
    ):
    # Call pycolmap methods needed for 3D reconstruction; first prepare directory
    # structure properly containing necessary data
    colmap_in_path = os.path.join(working_directory, "colmap", "input")
    colmap_out_path = os.path.join(working_directory, "colmap", "output")
    if os.path.exists(colmap_in_path):
        for path in pathlib.Path(colmap_in_path).glob("**/*"):
            if path.is_file(): path.unlink()
            elif path.is_dir(): shutil.rmtree(path)
    else:
        os.makedirs(colmap_in_path, exist_ok=True)
    if os.path.exists(colmap_out_path):
        for path in pathlib.Path(colmap_out_path).glob("**/*"):
            if path.is_file(): path.unlink()
            elif path.is_dir(): shutil.rmtree(path)
    else:
        os.makedirs(colmap_out_path, exist_ok=True)

    # Create empty points3D.txt file
    with open(os.path.join(colmap_in_path, "points3D.txt"), mode='w'): pass
    # Create cameras.txt file and add camera info
    with open(os.path.join(colmap_in_path, "cameras.txt"), mode='w') as cams_txt_f:
        cam_K, distortion_coeffs = camera_intrinsics
        fx = cam_K[0][0]; fy = cam_K[1][1]; cx = cam_K[0][2]; cy = cam_K[1][2]
        k1, k2, p1, p2, *_ = distortion_coeffs[0]
        line = "1 OPENCV 800 600 "
        line += f"{fx:.5f} {fy:.5f} {cx:.5f} {cy:.5f} {k1:.5f} {k2:.5f} {p1:.5f} {p2:.5f}"
        cams_txt_f.write(line + "\n")
    # Create images.txt file and add image info
    with open(os.path.join(colmap_in_path, "images.txt"), mode='w') as imgs_txt_f:
        for id in images:
            (qw, qx, qy, qz), (tx, ty, tz) = viewpoint_poses[id]
            line = f"{id+1} {qw:.5f} {qx:.5f} {qy:.5f} {qz:.5f} "
            line += f"{tx:.5f} {ty:.5f} {tz:.5f} 1 {id+1}.png"
            imgs_txt_f.write(line + "\n\n")
    # Create a subdirectory and save image files there
    os.makedirs(os.path.join(colmap_in_path, "images"), exist_ok=True)
    for id, img in images.items():
        img.save(os.path.join(colmap_in_path, "images", f"{id+1}.png"))

    # pycolmap reconstruction object
    reconstruction_template = pycolmap.Reconstruction(colmap_in_path)

    # Initialize a SQL database, populate with cameras, images and keypoints info
    colmap_db = COLMAPDatabase(os.path.join(colmap_in_path, "database.db"))
    colmap_db.create_tables()
    for cam in reconstruction_template.cameras.values():
        colmap_db.add_camera(
            cam.model.value, cam.width, cam.height, cam.params,
            camera_id=cam.camera_id, prior_focal_length=True    # Calibrated camera
        )
    for img in reconstruction_template.images.values():
        qx, qy, qz, qw = img.cam_from_world.rotation.quat
        txyz = img.cam_from_world.translation
        colmap_db.add_image(
            img.name, img.camera_id, prior_q=[qw, qx, qy, qz], prior_t=txyz,
            image_id=img.image_id
        )
    n2db_map = defaultdict(dict); db2n_map = defaultdict(dict)
    for n, points in points_in_views.items():
        colmap_db.add_keypoints(n+1, np.concatenate(list(points.values())))
        for i_db, i_n in enumerate(points):
            n2db_map[n][i_n] = i_db; db2n_map[n][i_db] = i_n

    # Geometric verification by pycolmap two-view geometry estimation
    for (u, v), matches in correspondences.items():
        # Verified matches stored as inlier_matches
        two_view_geom = pycolmap.estimate_two_view_geometry(
            reconstruction_template.cameras[1],
            np.concatenate([
                points_in_views[u][db2n_map[u][i]] for i in range(len(db2n_map[u]))
            ]),
            reconstruction_template.cameras[1],
            np.concatenate([
                points_in_views[v][db2n_map[v][i]] for i in range(len(db2n_map[v]))
            ]),
            np.array([[n2db_map[u][i_u], n2db_map[v][i_v]] for i_u, i_v in matches])
        )

        # Add matches data to colmap db
        colmap_db.add_matches(u+1, v+1, two_view_geom.inlier_matches)

        # Add verification results to colmap db
        colmap_db.add_two_view_geometry(
            u+1, v+1,
            two_view_geom.inlier_matches,
            F=two_view_geom.F, E=two_view_geom.E, H=two_view_geom.H
        )

    colmap_db.commit()

    # Final step: point triangulation
    reconstruction = pycolmap.triangulate_points(
        reconstruction_template,
        os.path.join(colmap_in_path, "database.db"),
        os.path.join(colmap_in_path, "images"),
        colmap_out_path
    )

    return reconstruction, db2n_map
