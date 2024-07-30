import os
import numpy as np


class Pointcloud:
    """
    Class for working with LiDAR pointclouds
    """

    def __init__(self, points):
        assert points.shape[1] == 4 or points.shape[1] == 5
        self.points = points


    def transform(self, T, in_place=True):
        """Transforms points given a transform, T. x_out = np.matmul(T, x)
        Args:
            T (np.ndarray): 4x4 transformation matrix
            in_place (bool): if True, self.points is updated
        Returns:
            points (np.ndarray): The transformed points
        """
        assert T.shape[0] == 4 and T.shape[1] == 4
        if in_place:
            points = self.points
        else:
            points = np.copy(self.points)
        p = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))
        points[:, :3] = np.matmul(p, T.transpose())[:, :3]
        return points


    def passthrough(self, bounds=[], in_place=True):
        """Removes points outside the specified bounds
        Args:
            bounds (list): [xmin, xmax, ymin, ymax, zmin, zmax]
            in_place (bool): if True, self.points is updated
        Returns:
            points (np.ndarray): the remaining points after the filter is applied
        """
        if len(bounds) < 6:
            print("Warning: len(bounds) = {} < 6 is incorrect!".format(len(bounds)))
            return self.points
        p = self.points[
            np.where(
                (self.points[:, 0] >= bounds[0])
                & (self.points[:, 0] <= bounds[1])
                & (self.points[:, 1] >= bounds[2])
                & (self.points[:, 1] <= bounds[3])
                & (self.points[:, 2] >= bounds[4])
                & (self.points[:, 2] <= bounds[5])
            )
        ]
        if in_place:
            self.points = p
        return p


    def remove_point_inside(self, filtered_range=[], in_place=True):
        """Removes points inside the specified bounds
        Args:
            filtered_range (list): [xmin, xmax, ymin, ymax, zmin, zmax]
            in_place (bool): if True, self.points is updated
        Returns:
            points (np.ndarray): the remaining points after the filter is applied
        """
        if len(filtered_range) < 6:
            print(
                "Warning: len(bounds) = {} < 6 is incorrect!".format(
                    len(filtered_range)
                )
            )
            return self.points
        pc = self.points
        filtered_range = np.array(filtered_range)

        filtered_range[3:6] -= 0.01  # np -> cuda .999999 = 1.0

        pc_x = (pc[:, 0] > -9999) & (pc[:, 0] < 9999)
        pc_y = (pc[:, 1] > -9999) & (pc[:, 1] < 9999)
        pc_z = (pc[:, 2] > -9999) & (pc[:, 2] < 9999)
        pc_mask = pc_x & pc_y & pc_z

        # 1. first filtered the z-axis
        mask_z = (pc[:, 2] > filtered_range[4]) & (pc[:, 2] < filtered_range[5])

        # 2. then filtered the x,y-axis
        mask_x = (pc[:, 0] > filtered_range[0]) & (pc[:, 0] < filtered_range[1])
        mask_y = (pc[:, 1] > filtered_range[2]) & (pc[:, 1] < filtered_range[3])

        # 3. do logical and of the mask filter z and xy
        mask_region = mask_x & mask_y & np.logical_not(mask_z)
        mask = np.logical_xor(pc_mask, mask_region)

        p = pc[mask]
        if in_place:
            self.points = p
        return p


    def voxelize(self, voxel_size=0.175, extents=None, return_indices=False):
        """
        Voxelize the input point cloud. We only record if a given voxel is occupied or not, which is just binary indicator.

        The input for the voxelization is expected to be a PointCloud
        with N points in 4 dimension (x,y,z,i). Voxel size is the quantization size for the voxel grid.

        voxel_size: I.e. if voxel size is 1 m, the voxel space will be
        divided up within 1m x 1m x 1m space. This space will be 0 if free/occluded and 1 otherwise.
        min_voxel_coord: coordinates of the minimum on each axis for the voxel grid
        max_voxel_coord: coordinates of the maximum on each axis for the voxel grid
        num_divisions: number of grids in each axis
        leaf_layout: the voxel grid of size (numDivisions) that contain -1 for free, 0 for occupied

        :param pts: Point cloud as N x [x, y, z, i]
        :param voxel_size: Quantization size for the grid, vd, vh, vw
        :param extents: Optional, specifies the full extents of the point cloud.
                        Used for creating same sized voxel grids. Shape (3, 2)
        :param return_indices: Whether to return the non-empty voxel indices.
        """
        pts = self.points[:, :4]
        # Function Constants
        VOXEL_EMPTY = 0
        VOXEL_FILLED = 1

        # Check if points are 3D, otherwise early exit
        if pts.shape[1] < 3 or pts.shape[1] > 4:
            raise ValueError("Points have the wrong shape: {}".format(pts.shape))

        if extents is not None:
            if extents.shape != (3, 2):
                raise ValueError("Extents are the wrong shape {}".format(extents.shape))

            filter_idx = np.where(
                (extents[0, 0] < pts[:, 0])
                & (pts[:, 0] < extents[0, 1])
                & (extents[1, 0] < pts[:, 1])
                & (pts[:, 1] < extents[1, 1])
                & (extents[2, 0] < pts[:, 2])
                & (pts[:, 2] < extents[2, 1])
            )[0]
            pts = pts[filter_idx]

        # Discretize voxel coordinates to given quantization size
        discrete_pts = np.floor(pts[:, :3] / voxel_size).astype(np.int32)

        # Use Lex Sort, sort by x, then y, then z
        x_col = discrete_pts[:, 0]
        y_col = discrete_pts[:, 1]
        z_col = discrete_pts[:, 2]
        sorted_order = np.lexsort((z_col, y_col, x_col))

        # Save original points in sorted order
        discrete_pts = discrete_pts[sorted_order]

        # Format the array to c-contiguous array for unique function
        contiguous_array = np.ascontiguousarray(discrete_pts).view(
            np.dtype((np.void, discrete_pts.dtype.itemsize * discrete_pts.shape[1]))
        )

        # The new coordinates are the discretized array with its unique indexes
        _, unique_indices = np.unique(contiguous_array, return_index=True)

        # Sort unique indices to preserve order
        unique_indices.sort()

        voxel_coords = discrete_pts[unique_indices]
        # Compute the minimum and maximum voxel coordinates
        if extents is not None:
            min_voxel_coord = np.floor(extents.T[0] / voxel_size)
            max_voxel_coord = np.ceil(extents.T[1] / voxel_size) - 1
        else:
            min_voxel_coord = np.amin(voxel_coords, axis=0)
            max_voxel_coord = np.amax(voxel_coords, axis=0)

        # Get the voxel grid dimensions
        num_divisions = ((max_voxel_coord - min_voxel_coord) + 1).astype(np.int32)
        # Bring the min voxel to the origin
        voxel_indices = (voxel_coords - min_voxel_coord).astype(int)

        # Create Voxel Object with -1 as empty/occluded
        leaf_layout = VOXEL_EMPTY * np.ones(num_divisions.astype(int), dtype=np.float32)

        # Fill out the leaf layout
        leaf_layout[
            voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]
        ] = VOXEL_FILLED

        if return_indices:
            return leaf_layout, voxel_indices
        else:
            return leaf_layout


    def crop_point_cloud(self, start_angle, end_angle):
        """
        crop the lidar point cloud to a specific angle range

        Params
        ---
        :param start_angle: start angle in degree
        :param end_angle: end angle in degree
        """
        # Define the start and end angles for the desired field of view
        start_angle = start_angle
        end_angle = end_angle

        # Extract the azimuth angle from the point cloud
        azimuth = np.arctan2(self.points[:, 1], -self.points[:, 0]) * 180 / np.pi + 180

        # Filter out the points that fall outside the desired field of view
        cropped_cloud = self.points[(azimuth <= end_angle) & (azimuth >= start_angle)]

        return cropped_cloud


    def random_downsample(self, downsample_rate, in_place=True):
        rand_idx = np.random.choice(
            self.points.shape[0],
            size=int(self.points.shape[0] * downsample_rate),
            replace=False,
        )
        p = self.points[rand_idx, :]
        if in_place:
            self.points = p
        return p
