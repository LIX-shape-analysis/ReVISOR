import os
import sys
import torch
import numpy as np

from plyfile import PlyData, PlyElement


def fastcdist(X, Y, squared=False):
    """
    (By Robin Magnet)
    Compute pairwise euclidean distance between two collections of vectors in a k-dimensional space

    Parameters
    --------------
    X       : (n1, k) first collection
    Y       : (n2, k) second collection
    squared : bool - whether to compute the squared euclidean distance

    Output
    --------------
    distmat : (n1, n2) distance matrix
    """
    normX = np.linalg.norm(X, axis=1)**2
    normY = np.linalg.norm(Y, axis=1)**2

    # Compute ||x||^2 + ||y||^2 - 2<x,y> for all pairs
    distmat = X @ Y.T
    distmat *= -2
    distmat += normX[:, None]
    distmat += normY[None, :]

    # Rarely useful
    # np.maximum(distmat, 0, out=distmat)

    if not squared:
        np.sqrt(distmat, out=distmat)

    return distmat


def create_dir_if_required(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("created directory {}.".format(directory))

def get_dtype_dict(name):
    dtype_dict = {'names':["scalar_{}".format(name)],'formats':['f4']}
    if name=="vertex" or name=="point":
        dtype_dict = {'names':['x','y','z'], 'formats':['f4','f4','f4']}
    elif name=="intensity":
        dtype_dict = {'names':['variation'],'formats':['f4']}
    elif name=="normal":
        dtype_dict = {'names':['nx','ny','nz'], 'formats':['f4','f4','f4']}
    elif name=="color":
        dtype_dict = {'names':['R','G','B'], 'formats':['i4','i4','i4']}
    elif name=="U_CHAR":
        dtype_dict = {'names':['U_CHAR'], 'formats':['B']}
    return dtype_dict

def graph2ply(filename, graph):
    """write a ply displaying the graph by adding edges between its nodes"""
    vertex_prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertex_val = np.empty((graph['pos']).shape[0], dtype=vertex_prop)
    for i in range(0, 3):
        vertex_val[vertex_prop[i][0]] = graph['pos'][:, i]
    edges_prop = [('vertex1', 'int32'), ('vertex2', 'int32')]
    edges_val = np.empty((graph['edge_index'][0,:]).shape[0], dtype=edges_prop)
    edges_val[edges_prop[0][0]] = graph['edge_index'][0,:].flatten()
    edges_val[edges_prop[1][0]] = graph['edge_index'][1,:].flatten()
    ply = PlyData([PlyElement.describe(vertex_val, 'vertex'), PlyElement.describe(edges_val, 'edge')])
    ply.write(filename)

def read_ply_ls(directory,ls,print_infos=False):
    plydata = PlyData.read(directory)
    if print_infos:
        print(plydata)
    out_dict = {}
    for name in ls:
        name = (name,name)
        dtype_dict = get_dtype_dict(name[1])
        dat = plydata[name[0]]
        elems_ls = []
        for channel in dtype_dict["names"]:
            elems_ls.append(dat[channel])
        out_dict[name[1]] = np.transpose(np.array(elems_ls,dtype=np.float32))

    return out_dict

def write_ply(filename,params_in_ls,params_names_ls):
    el_ls = []
    for idx,param in enumerate(params_in_ls):
        if len(param.shape)==1:
            param = np.expand_dims(param,1)

        cur_name = params_names_ls[idx]
        dtype_dict = get_dtype_dict(cur_name)
        str_array = np.zeros(param.shape[0],dtype=dtype_dict)

        for i,name in enumerate(dtype_dict["names"]):
            str_array[name] = param[:,i]

        el = PlyElement.describe(str_array,cur_name,comments=["Generated with write_ply.py"])
        el_ls.append(el)

    PlyData(el_ls).write(filename)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'ops'))
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling


def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.compute(points, sampleDl=sampleDl, verbose=verbose)
    elif (labels is None):
        return cpp_subsampling.compute(points, features=features, sampleDl=sampleDl, verbose=verbose)
    elif (features is None):
        return cpp_subsampling.compute(points, classes=labels, sampleDl=sampleDl, verbose=verbose)
    else:
        return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=sampleDl, verbose=verbose)


def angle_axis(angle, axis):
    # type: (float, np.ndarray) -> float
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle

    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about

    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    # yapf: disable
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                               [u[2], 0.0, -u[0]],
                               [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    )
    # yapf: enable
    return R.float()


class PointcloudScale(object):
    def __init__(self, scale_low=0.8, scale_high=1.25):
        self.scale_low, self.scale_high = scale_low, scale_high

    def __call__(self, points):
        scaler = np.random.uniform(self.scale_low, self.scale_high, size=[3])
        scaler = torch.from_numpy(scaler).float()
        points[:, 0:3] *= scaler
        return points


class PointcloudRotate(object):
    def __init__(self, axis=np.array([0.0, 1.0, 0.0])):
        self.axis = axis

    def __call__(self, points):
        rotation_angle = np.random.uniform() * 2 * np.pi
        rotation_matrix = angle_axis(rotation_angle, self.axis)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points


class PointcloudRotatePerturbation(object):
    def __init__(self, angle_sigma=0.06, angle_clip=0.18):
        self.angle_sigma, self.angle_clip = angle_sigma, angle_clip

    def _get_angles(self):
        angles = np.clip(
            self.angle_sigma * np.random.randn(3), -self.angle_clip, self.angle_clip
        )

        return angles

    def __call__(self, points):
        angles_ = self._get_angles()
        Rx = angle_axis(angles_[0], np.array([1.0, 0.0, 0.0]))
        Ry = angle_axis(angles_[1], np.array([0.0, 1.0, 0.0]))
        Rz = angle_axis(angles_[2], np.array([0.0, 0.0, 1.0]))

        rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points


class PointcloudRandomRotate(object):
    def __init__(self, x_range=np.pi, y_range=np.pi, z_range=np.pi):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

    def _get_angles(self):
        x_angle = np.random.uniform(-self.x_range, self.x_range)
        y_angle = np.random.uniform(-self.y_range, self.y_range)
        z_angle = np.random.uniform(-self.z_range, self.z_range)

        return np.array([x_angle, y_angle, z_angle])

    def __call__(self, points):
        angles_ = self._get_angles()
        Rx = angle_axis(angles_[0], np.array([1.0, 0.0, 0.0]))
        Ry = angle_axis(angles_[1], np.array([0.0, 1.0, 0.0]))
        Rz = angle_axis(angles_[2], np.array([0.0, 0.0, 1.0]))

        rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points


class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, points):
        jittered_data = (
            points.new(points.size(0), 3)
                .normal_(mean=0.0, std=self.std)
                .clamp_(-self.clip, self.clip)
        )
        points[:, 0:3] += jittered_data
        return points


class PointcloudTranslate(object):
    def __init__(self, translate_range=0.1):
        self.translate_range = translate_range

    def __call__(self, points):
        translation = np.random.uniform(-self.translate_range, self.translate_range, size=[3])
        translation = torch.from_numpy(translation)
        points[:, 0:3] += translation
        return points


class PointcloudToTensor(object):
    def __call__(self, points):
        return torch.from_numpy(points).contiguous().type(torch.float32)#.float()


class PointcloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.875):
        assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, points):
        pc = points.numpy()

        dropout_ratio = np.random.random() * self.max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            pc[drop_idx] = pc[0]  # set to the first point

        return torch.from_numpy(pc).float()


class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
        xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
        pc[:, 0:3] = torch.mul(pc[:, 0:3], torch.from_numpy(xyz1).float()) + torch.from_numpy(
            xyz2).float()

        return pc


class PointcloudScaleAndJitter(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., std=0.01, clip=0.05, augment_symmetries=[0, 0, 0]):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.std = std
        self.clip = clip
        self.augment_symmetries = augment_symmetries

    def __call__(self, pc):
        xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
        symmetries = np.round(np.random.uniform(low=0, high=1, size=[3])) * 2 - 1
        symmetries = symmetries * np.array(self.augment_symmetries) + (1 - np.array(self.augment_symmetries))
        xyz1 *= symmetries
        xyz2 = np.clip(np.random.normal(scale=self.std, size=[pc.shape[0], 3]), a_min=-self.clip, a_max=self.clip)
        pc[:, 0:3] = torch.mul(pc[:, 0:3], torch.from_numpy(xyz1).float()) + torch.from_numpy(
            xyz2).float()

        return pc


class BatchPointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])

            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().to(pc.device)) + torch.from_numpy(
                xyz2).float().to(pc.device)

        return pc


class BatchPointcloudScaleAndJitter(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., std=0.01, clip=0.05, augment_symmetries=[0, 0, 0]):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.std, self.clip = std, clip
        self.augment_symmetries = augment_symmetries

    def __call__(self, pc):
        bsize = pc.size()[0]
        npoint = pc.size()[1]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            symmetries = np.round(np.random.uniform(low=0, high=1, size=[3])) * 2 - 1
            symmetries = symmetries * np.array(self.augment_symmetries) + (1 - np.array(self.augment_symmetries))
            xyz1 *= symmetries
            xyz2 = np.clip(np.random.normal(scale=self.std, size=[npoint, 3]), a_max=self.clip, a_min=-self.clip)

            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().to(pc.device)) + torch.from_numpy(
                xyz2).float().to(pc.device)

        return pc


class BatchPointcloudRandomRotate(object):
    def __init__(self, x_range=np.pi, y_range=np.pi, z_range=np.pi):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

    def _get_angles(self):
        x_angle = np.random.uniform(-self.x_range, self.x_range)
        y_angle = np.random.uniform(-self.y_range, self.y_range)
        z_angle = np.random.uniform(-self.z_range, self.z_range)

        return np.array([x_angle, y_angle, z_angle])

    def __call__(self, pc):
        bsize = pc.size()[0]
        normals = pc.size()[2] > 3
        for i in range(bsize):
            angles_ = self._get_angles()
            Rx = angle_axis(angles_[0], np.array([1.0, 0.0, 0.0]))
            Ry = angle_axis(angles_[1], np.array([0.0, 1.0, 0.0]))
            Rz = angle_axis(angles_[2], np.array([0.0, 0.0, 1.0]))

            rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx).to(pc.device)

            if not normals:
                pc[i, :, 0:3] = torch.matmul(pc[i, :, 0:3], rotation_matrix.t())
            else:
                pc[i, :, 0:3] = torch.matmul(pc[i, :, 0:3], rotation_matrix.t())
                pc[i, :, 3:] = torch.matmul(pc[i, :, 3:], rotation_matrix.t())
        return pc
