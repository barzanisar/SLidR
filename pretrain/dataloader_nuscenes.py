import os
import copy
import torch
import numpy as np
from PIL import Image
import MinkowskiEngine as ME
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud


CUSTOM_SPLIT = [
    "scene-0008", "scene-0009", "scene-0019", "scene-0029", "scene-0032", "scene-0042",
    "scene-0045", "scene-0049", "scene-0052", "scene-0054", "scene-0056", "scene-0066",
    "scene-0067", "scene-0073", "scene-0131", "scene-0152", "scene-0166", "scene-0168",
    "scene-0183", "scene-0190", "scene-0194", "scene-0208", "scene-0210", "scene-0211",
    "scene-0241", "scene-0243", "scene-0248", "scene-0259", "scene-0260", "scene-0261",
    "scene-0287", "scene-0292", "scene-0297", "scene-0305", "scene-0306", "scene-0350",
    "scene-0352", "scene-0358", "scene-0361", "scene-0365", "scene-0368", "scene-0377",
    "scene-0388", "scene-0391", "scene-0395", "scene-0413", "scene-0427", "scene-0428",
    "scene-0438", "scene-0444", "scene-0452", "scene-0453", "scene-0459", "scene-0463",
    "scene-0464", "scene-0475", "scene-0513", "scene-0533", "scene-0544", "scene-0575",
    "scene-0587", "scene-0589", "scene-0642", "scene-0652", "scene-0658", "scene-0669",
    "scene-0678", "scene-0687", "scene-0701", "scene-0703", "scene-0706", "scene-0710",
    "scene-0715", "scene-0726", "scene-0735", "scene-0740", "scene-0758", "scene-0786",
    "scene-0790", "scene-0804", "scene-0806", "scene-0847", "scene-0856", "scene-0868",
    "scene-0882", "scene-0897", "scene-0899", "scene-0976", "scene-0996", "scene-1012",
    "scene-1015", "scene-1016", "scene-1018", "scene-1020", "scene-1024", "scene-1044",
    "scene-1058", "scene-1094", "scene-1098", "scene-1107",
]


def minkunet_collate_pair_fn(list_data):
    """
    Collate function adapted for creating batches with MinkowskiEngine.
    """
    (
        coords,
        feats,
        images,
        pairing_points,
        pairing_images,
        inverse_indexes,
        superpixels,
    ) = list(zip(*list_data))
    batch_n_points, batch_n_pairings = [], []

    offset = 0
    for batch_id in range(len(coords)):

        # Move batchids to the beginning
        coords[batch_id][:, 0] = batch_id
        pairing_points[batch_id][:] += offset
        pairing_images[batch_id][:, 0] += batch_id * images[0].shape[0]

        batch_n_points.append(coords[batch_id].shape[0]) #num voxels in this pc
        batch_n_pairings.append(pairing_points[batch_id].shape[0]) #num fov pts in this pc
        offset += coords[batch_id].shape[0]

    # Concatenate all lists
    coords_batch = torch.cat(coords, 0).int() #(num voxels in batch, 4=bid,rho,phi,z vox coordinate)
    pairing_points = torch.tensor(np.concatenate(pairing_points))#(num pairing pts in batch)
    pairing_images = torch.tensor(np.concatenate(pairing_images))#(num pairing pts in batch, 3)
    feats_batch = torch.cat(feats, 0).float() #(num voxels in batch, 1=intensity)
    images_batch = torch.cat(images, 0).float() #(6 views*4 bs, 3, 224, 416)
    superpixels_batch = torch.tensor(np.concatenate(superpixels)) #(6 views*4 bs, 224, 416)
    return {
        "sinput_C": coords_batch,
        "sinput_F": feats_batch,
        "input_I": images_batch,
        "pairing_points": pairing_points,
        "pairing_images": pairing_images,
        "batch_n_pairings": batch_n_pairings, #num fov pts in each pc of batch
        "inverse_indexes": inverse_indexes, # list of 4=bs inverse_indices (each element of len num pts in pc)
        "superpixels": superpixels_batch,
    }

def visualize(pc, intensity, images, pairing_points, pairing_images, superpixels):
    import matplotlib.pyplot as plt

    cam_view = 0
    # fig = plt.figure(figsize=(8.32,4.48))
    # ax = fig.add_axes([0, 0, 1, 1])
    # plt.axis('off')
    # ax.imshow(images[cam_view].permute(1,2,0))
    # fig.show()

    # fig = plt.figure(figsize=(8.32,4.48))
    # ax = fig.add_axes([0, 0, 1, 1])
    # plt.axis('off')
    cam_view_mask = pairing_images[:,0] == cam_view
    point_intensity_cam_view = intensity[pairing_points[cam_view_mask]]
    vu_pix_cam_view = pairing_images[cam_view_mask, 1:]
    uv_pix = np.flip(vu_pix_cam_view, axis=1)
    image_per_point_cam_view = images[cam_view].permute(1,2,0)[vu_pix_cam_view[:,0], vu_pix_cam_view[:,1], :]
    # ax.scatter(uv_pix[:,0], uv_pix[:, 1], c=point_intensity_cam_view, s=15)
    # ax.imshow(np.zeros((224,416,4)))
    # fig.show()

    fig = plt.figure(figsize=(8.32,4.48))
    ax = fig.add_axes([0, 0, 1, 1])
    plt.axis('off')
    ax.imshow(superpixels[cam_view], cmap='gray', alpha=0.5)
    ax.scatter(uv_pix[:,0], uv_pix[:, 1], c=image_per_point_cam_view, s=15, alpha=0.8)
    fig.show()

    fig = plt.figure(figsize=(8.32,4.48))
    ax = fig.add_axes([0, 0, 1, 1])
    plt.axis('off')
    ax.imshow(images[cam_view].permute(1,2,0))
    ax.scatter(uv_pix[:,0], uv_pix[:, 1], c=point_intensity_cam_view, s=15, alpha=0.5)
    fig.show()

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pc[:,0], pc[:,1], pc[:,2], s=10, c=intensity)
    fig.show()

    b=1


class NuScenesMatchDataset(Dataset):
    """
    Dataset matching a 3D points cloud and an image using projection.
    """

    def __init__(
        self,
        phase,
        config,
        shuffle=False,
        cloud_transforms=None,
        mixed_transforms=None,
        **kwargs,
    ):
        self.phase = phase
        self.shuffle = shuffle
        self.cloud_transforms = cloud_transforms
        self.mixed_transforms = mixed_transforms
        self.voxel_size = config["voxel_size"]
        self.cylinder = config["cylindrical_coordinates"]
        self.superpixels_type = config["superpixels_type"]
        self.bilinear_decoder = config["decoder"] == "bilinear"
        self.sample_points = config.get("sample_points", None)
        self.voxel_decimation = config.get("voxel_decimation", False)

        version=config['version']
        nuscenes_path = f"datasets/nuscenes/{version}"
        if "cached_nuscenes" in kwargs:
            self.nusc = kwargs["cached_nuscenes"]
        else:
            self.nusc = NuScenes(
                version=version, dataroot=nuscenes_path, verbose=False
            )

        self.list_keyframes = []
        # a skip ratio can be used to reduce the dataset size and accelerate experiments
        try:
            skip_ratio = config["dataset_skip_step"]
        except KeyError:
            skip_ratio = 1
        skip_counter = 0
        if phase in ("train", "val", "test"):
            phase_scenes = create_splits_scenes()[phase]
        elif phase == "parametrizing":
            phase_scenes = list(
                set(create_splits_scenes()["train"]) - set(CUSTOM_SPLIT)
            )
        elif phase == "verifying":
            phase_scenes = CUSTOM_SPLIT
        # create a list of camera & lidar scans
        for scene_idx in range(len(self.nusc.scene)):
            scene = self.nusc.scene[scene_idx]
            if scene["name"] in phase_scenes:
                skip_counter += 1
                if skip_counter % skip_ratio == 0:
                    self.create_list_of_scans(scene)

    def create_list_of_scans(self, scene):
        # Get first and last keyframe in the scene
        current_sample_token = scene["first_sample_token"]

        # Loop to get all successive keyframes
        list_data = []
        while current_sample_token != "":
            current_sample = self.nusc.get("sample", current_sample_token)
            list_data.append(current_sample["data"])
            current_sample_token = current_sample["next"]

        # Add new scans in the list
        self.list_keyframes.extend(list_data)

    def map_pointcloud_to_image(self, data, min_dist: float = 1.0):
        """
        Given a lidar token and camera sample_data token, load pointcloud and map it to
        the image plane. Code adapted from nuscenes-devkit
        https://github.com/nutonomy/nuscenes-devkit.
        :param min_dist: Distance from the camera below which points are discarded.
        """
        pointsensor = self.nusc.get("sample_data", data["LIDAR_TOP"])
        pcl_path = os.path.join(self.nusc.dataroot, pointsensor["filename"])
        pc_original = LidarPointCloud.from_file(pcl_path)
        
        if self.voxel_decimation:
            pos = pc_original.points.T
            pos = (pos/0.1).astype(int) # each point represented as voxel coordinates
            num_pts = pos.shape[0]

            # Numpy version
            _, indices = np.unique(pos, return_index=True, axis=0) #find unique voxels
            pc_original.points = pc_original.points[:, indices]


        if self.sample_points is not None:
            indices_selected = np.random.choice(pc_original.points.shape[1], self.sample_points, replace=False)
            pc_original.points = pc_original.points[:,indices_selected]
        
        
            
        pc_ref = pc_original.points

        images = []
        superpixels = []
        pairing_points = np.empty(0, dtype=np.int64)
        pairing_images = np.empty((0, 3), dtype=np.int64)
        camera_list = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_FRONT_LEFT",
        ]
        if self.shuffle:
            np.random.shuffle(camera_list)
        for i, camera_name in enumerate(camera_list):
            pc = copy.deepcopy(pc_original)
            cam = self.nusc.get("sample_data", data[camera_name])
            im = np.array(Image.open(os.path.join(self.nusc.dataroot, cam["filename"])))
            sp = Image.open(
                f"superpixels/nuscenes/"
                f"superpixels_{self.superpixels_type}/{cam['token']}.png"
            )
            superpixels.append(np.array(sp))

            # Points live in the point sensor frame. So they need to be transformed via
            # global to the image plane.
            # First step: transform the pointcloud to the ego vehicle frame for the
            # timestamp of the sweep.
            cs_record = self.nusc.get(
                "calibrated_sensor", pointsensor["calibrated_sensor_token"]
            )
            pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
            pc.translate(np.array(cs_record["translation"]))

            # Second step: transform from ego to the global frame.
            poserecord = self.nusc.get("ego_pose", pointsensor["ego_pose_token"])
            pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
            pc.translate(np.array(poserecord["translation"]))

            # Third step: transform from global into the ego vehicle frame for the
            # timestamp of the image.
            poserecord = self.nusc.get("ego_pose", cam["ego_pose_token"])
            pc.translate(-np.array(poserecord["translation"]))
            pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)

            # Fourth step: transform from ego into the camera.
            cs_record = self.nusc.get(
                "calibrated_sensor", cam["calibrated_sensor_token"]
            )
            pc.translate(-np.array(cs_record["translation"]))
            pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)

            # Fifth step: actually take a "picture" of the point cloud.
            # Grab the depths (camera frame z axis points away from the camera).
            depths = pc.points[2, :]

            # Take the actual picture
            # (matrix multiplication with camera-matrix + renormalization).
            points = view_points(
                pc.points[:3, :],
                np.array(cs_record["camera_intrinsic"]),
                normalize=True,
            ) #u,v,1 pixel coords for points

            # Remove points that are either outside or behind the camera.
            # Also make sure points are at least 1m in front of the camera to avoid
            # seeing the lidar points on the camera
            # casing for non-keyframes which are slightly out of sync.
            points = points[:2].T
            mask = np.ones(depths.shape[0], dtype=bool)
            mask = np.logical_and(mask, depths > min_dist)
            mask = np.logical_and(mask, points[:, 0] > 0)
            mask = np.logical_and(mask, points[:, 0] < im.shape[1] - 1)
            mask = np.logical_and(mask, points[:, 1] > 0)
            mask = np.logical_and(mask, points[:, 1] < im.shape[0] - 1)
            matching_points = np.where(mask)[0]
            matching_pixels = np.round(
                np.flip(points[matching_points], axis=1)
            ).astype(np.int64) #u, v pixel floats to v,u pixel integers 
            images.append(im / 255)
            pairing_points = np.concatenate((pairing_points, matching_points))
            pairing_images = np.concatenate(
                (
                    pairing_images,
                    np.concatenate(
                        (
                            np.ones((matching_pixels.shape[0], 1), dtype=np.int64) * i,
                            matching_pixels,
                        ),
                        axis=1,
                    ),
                )
            )

        # pc_ref: (4, num pts in this pc) orig pc in lidar frame
        # images: a list of 6 camera images (diff cam views) of size (900, 1600, 3) i.e. see camera_list
        # pairing_points: (N=num FOV pts) indices of pts projected onto image views
        # pairing_images: (N, 3) [cam_idx in camera_list, v pixel, u pixel] of the points projected on all 6 image views
        # np.stack(superpixels): (6, 900, 1600) slic segmentation for 6 image views, each pixel is given a segment id it belongs to
        return pc_ref.T, images, pairing_points, pairing_images, np.stack(superpixels) 

    def __len__(self):
        return len(self.list_keyframes)

    def __getitem__(self, idx):
        (
            pc,
            images,
            pairing_points,
            pairing_images,
            superpixels,
        ) = self.map_pointcloud_to_image(self.list_keyframes[idx])
        superpixels = torch.tensor(superpixels)

        intensity = torch.tensor(pc[:, 3:]) # (num pts pc, 1)
        pc = torch.tensor(pc[:, :3]) # (num pts pc, 3)
        images = torch.tensor(np.array(images, dtype=np.float32).transpose(0, 3, 1, 2))# (6,3,900, 1600)

        # visualize(pc, intensity, images, pairing_points, pairing_images, superpixels)
        if self.cloud_transforms:
            pc = self.cloud_transforms(pc)
        if self.mixed_transforms:
            (
                pc,
                intensity,
                images,
                pairing_points,
                pairing_images,
                superpixels,
            ) = self.mixed_transforms(
                pc, intensity, images, pairing_points, pairing_images, superpixels
            )

        # visualize(pc, intensity, images, pairing_points, pairing_images, superpixels)
        if self.cylinder:
            # Transform to cylinder coordinate and scale for voxel size
            x, y, z = pc.T
            rho = torch.sqrt(x ** 2 + y ** 2) / self.voxel_size
            phi = torch.atan2(y, x) * 180 / np.pi  # corresponds to a split each 1°
            z = z / self.voxel_size
            coords_aug = torch.cat((rho[:, None], phi[:, None], z[:, None]), 1)
        else:
            coords_aug = pc / self.voxel_size

        # Voxelization with MinkowskiEngine
        # inverse_indexes (len == num pts): for each pt, gives the corresponding index of discrete coord i.e. used to retrieve point wise features from voxel features
        # indexes (len == num voxels or discrete coords): for each discrete coord, gives its corresponding pt index i.e. used to retrieve voxel intensities from point intensities
        discrete_coords, indexes, inverse_indexes = ME.utils.sparse_quantize(
            coords_aug.contiguous(), return_index=True, return_inverse=True
        )
        # pts here are the indexes of voxels corresponding to pairing points 
        pairing_points = inverse_indexes[pairing_points] #inverse indexes for pairing points i.e. used to retrieve pairing-point wise features from voxel features

        unique_feats = intensity[indexes]

        discrete_coords = torch.cat(
            (
                torch.zeros(discrete_coords.shape[0], 1, dtype=torch.int32),
                discrete_coords,
            ),
            1,
        ) #0, rho, phi, z voxel coord --> (num voxels, 4)

        return (
            discrete_coords, #(num voxels, 4= 0, rho, phi, z of the voxel)
            unique_feats, #(num voxels, 1=voxel intensity)
            images, # (6,3,224,416)
            pairing_points, #(Num pairing pts, 1=inverse index (voxel2pc))
            pairing_images, #(Num pairing pts, 3=cam id, v pixel, u pixel)
            inverse_indexes, #(num pts, 1=inverse index)
            superpixels, #(6 img views, 224, 416)
        )
