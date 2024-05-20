# --------------------------------------------------------
# 3D Model Generation Script without gradio
# --------------------------------------------------------
import argparse
import os
import torch
import numpy as np
import copy
import trimesh
from scipy.spatial.transform import Rotation
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
batch_size = 1

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, nargs='+', required=True, help="Paths to the images")
    parser.add_argument("--weights", type=str, help="Path to the model weights", required=True)
    parser.add_argument("--device", type=str, default='cuda', help="PyTorch device")
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="Image size")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save the output")
    parser.add_argument("--silent", action='store_true', default=False, help="Silence logs")
    return parser

def _convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.05, cam_color=None, as_pointcloud=False, transparent_cams=False, silent=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    for i, pose_c2w in enumerate(cams2world):
        camera_edge_color = cam_color[i] if isinstance(cam_color, list) else cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color, None if transparent_cams else imgs[i], focals[i], imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    outfile = os.path.join(outdir, 'scene.glb')
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile

def get_3D_model_from_scene(outdir, silent, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False, clean_depth=False, transparent_cams=False, cam_size=0.05):
    if scene is None:
        return None
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())
    return _convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud, transparent_cams=transparent_cams, cam_size=cam_size, silent=silent)

def generate_3D_model(image_paths, model, device, image_size, output_dir, silent):
    imgs = load_images(image_paths, size=image_size, verbose=not silent)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1

    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size, verbose=not silent)

    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        scene.compute_global_alignment(init='mst', niter=300, schedule='linear', lr=lr)

    outfile = get_3D_model_from_scene(output_dir, silent, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False, clean_depth=True, transparent_cams=False, cam_size=0.05)

    print(f"3D model saved to {outfile}")

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    model = AsymmetricCroCo3DStereo.from_pretrained(args.weights).to(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    image_paths = args.images
    generate_3D_model(image_paths, model, args.device, args.image_size, args.output_dir, args.silent)
