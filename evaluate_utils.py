import torch
import numpy as np

from data_utils import grid_subsampling, read_ply_ls, create_dir_if_required
from sklearn.neighbors import KDTree

from utils.config import config, update_config

import glob
import os

def create_cfg(diameter_percent=20,path2plyclouds="/gpfsgaia/projets/maquette3d/DATA/SAL1_CHB2_GRA2_CRU3/",path2basecfgs="./cfgs/",features=["intensity","visibility"]):
    '''
    Creates the config Namespace
    Inputs:
    diameter_percent: determines the receptive field size.
    path2plyclouds: where the ply files containing the point clouds are stored.
    path2basecfgs: where the base config files are stored (e.g. grid.yaml, pool.yaml, ...)
    '''
    cfg = glob.glob("{}*grid*.yaml".format(path2basecfgs))[0]

    update_config(cfg)

    config.features = features
    config.katz_params = [3.2]

    config.dataset = "EDFM"

    shape_diameter = 10. # 5m radius spherical scenes
    config.sampleDl = 0.0
    # sous-échantillonnages
    if config.in_radius>=0.25:
        config.sampleDl = 0.02
    if config.in_radius>=1:
        config.sampleDl = 0.04
    if config.in_radius>1.5:
        config.sampleDl = 0.06
    if config.in_radius>2.1:
        config.sampleDl = 0.09
    config.data_root = path2plyclouds

    config.local_aggregator = "grid"
    
    features_raw = []
    if "intensity" in features:
        features_raw.append("int")
    if "visibility" in features:
        features_raw.append("ktz")

    original_warmup_name = "{}_{}_{:03d}_no_feat".format(config.dataset,config.local_aggregator,int(diameter_percent))
                                                         
    if not features:
        job_name = "finetuned_"+original_warmup_name
    else:
        job_name = "finetuned_{}_{}_{:03d}_".format(config.dataset,config.local_aggregator,int(diameter_percent))+"_".join(features_raw)
    if "ktz" in features_raw:
        job_name = job_name+"_{}_".format(config.katz_type)+"_".join(["{:.2f}".format(k) for k in config.katz_params])
    config.job_name = job_name

    config.DEBUG = 0

    config.in_radius = 0.5*shape_diameter*diameter_percent/100.

    config.radius = max(config.in_radius*np.sqrt(3)/32.,0.025) # before, 0.1

    num_points = int(15000*config.in_radius*0.5)
    print("Num. points: {}".format(num_points))
    if num_points==15000:
        config.nsamples = [26,31,38,41,39]
        config.npoints = [4096,1152,304,88]
    else:
        config.nsamples = [2*26,int(1.5*26),int(1.25*26),26,26]
        config.npoints = [max(int(num_points/4.),1),max(int(num_points/16.),1),max(int(num_points/32.),1),max(int(num_points/128.),1)]



    print("RADII: in_radius={:.2f}, smallest radius={:.5f}".format(config.in_radius,config.radius))
    print("SAMPLEDL = {:.2f}".format(config.sampleDl))

    config.noise_std = 0.001*config.in_radius*0.5
    config.noise_clip = 0.05*config.in_radius*0.5

    config.input_features_dim = 0
    for f in config.features:
        if f=="normal":
            config.input_features_dim += 3
        if "katz" in f:
            config.input_features_dim += len(config.katz_params)
        if f=="intensity":
            config.input_features_dim += 1
    rem = abs(3 - config.input_features_dim%3)%3

    config.input_features_dim += rem

    config.local_rank = 0

    config.log_dir = "./VIEW/"


    config.backbone = "resnet"
    
    return config


def softmax(x,axis=0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x,axis=axis,keepdims=True))
    return e_x / e_x.sum(axis=axis,keepdims=True)


from scipy.spatial import ConvexHull
def HPR_op(pos, pos_norm, pos_dir, parameter,ktype="std"):
    if ktype=="std":
        # Original R
        R = np.max(pos_norm)*10**parameter # biggest distance from scanning device
        pos_hat = pos+2*(R-pos_norm)*pos_dir
    elif ktype=="exp":
        # Exp ktz
        pos_hat = pos_dir*(pos_norm/np.max(pos_norm))**parameter

    pos_hat = np.concatenate([pos_hat,np.zeros((1,3))],axis=0)

    hull = ConvexHull(pos_hat)

    visible_indices = hull.vertices

    return visible_indices[:-1] # removing vertex corresponding to zero point

# Standard Katz visibility
def compute_katz(pos,parameters,ktype="std"):
    pos_norm = np.linalg.norm(pos,axis=1,ord=2)[:,None]
    pos_norm[pos_norm<1e-12]=1e-12
    pos_dir = pos/pos_norm

    k_ls = []
    for parameter in parameters:
        cur = np.ones((pos.shape[0],1)).astype(np.float32)
        indices = HPR_op(pos, pos_norm, pos_dir, parameter,ktype)
        cur[indices] = 0.
        k_ls.append(cur)

    katz = np.concatenate(k_ls,axis=1)
    return katz,pos_norm,pos_dir


def compute_features(config,cloud_name,cloud_points):
    cloud_katz = np.zeros((cloud_points.shape[0],1))
    if len(config.katz_params)>0:
        katz_features_ls = []
        for cur_val in config.katz_params:
            katz_file = os.path.join(config.data_root, "katz_values", "{}Ktz{:.3f}_".format(config.katz_type,cur_val) + cloud_name + '.npy')

            if os.path.exists(katz_file):
                katz = np.load(katz_file)
            else:
                katz,_,_ = compute_katz(cloud_points,[cur_val],config.katz_type)
                create_dir_if_required(os.path.join(config.data_root, "katz_values"))
                with open(katz_file,"wb") as f:
                    np.save(f,katz)
            katz_features_ls.append(katz)

        cloud_katz = np.concatenate(katz_features_ls,axis=1)
    
    return cloud_katz


def process_cloud(path2cloud,config,overlap=0.25):
    ply = read_ply_ls(path2cloud,["vertex","intensity"])
    hd_points = ply["vertex"]
    cloud_intensity = ply["intensity"] / 255.
    
    cloud_name = os.path.basename(path2cloud)
    
    if config.sampleDl>0:
        cloud_points = grid_subsampling(hd_points,sampleDl=config.sampleDl)
        hd_tree = KDTree(hd_points,leaf_size=20)
        low2high = hd_tree.query(cloud_points,k=1,return_distance=False)
        
        tree = KDTree(cloud_points,leaf_size=20)
        NN = tree.query(hd_points,k=1,return_distance=False)
    else:
        cloud_points = hd_points
        low2high = np.arange(hd_points.shape[0])
        
        tree = KDTree(cloud_points,leaf_size=20)
        NN = None
    
    # compute visibility
    cloud_katz = compute_features(config,cloud_name,hd_points)
    
    if len(config.features)==0:
        cloud_features = np.ones((hd_points.shape[0], 3), dtype=np.float32)
    else:
        all_ls = []
        for f in config.features:
            if f=="intensity":
                all_ls.append(cloud_intensity)
            elif "visibility" in f:
                all_ls.append(cloud_katz)

        cloud_features = np.concatenate(all_ls,axis=1)
    cloud_features = cloud_features[low2high,:]
    
    
    sub_pc, _, _ = grid_subsampling(cloud_points,features=cloud_points, labels=np.ones(cloud_points.shape[0]).astype(np.int32), sampleDl=min(overlap*config.in_radius,overlap*2.)) # the subsampling value of 0.25*2. is fixed for radii > 2. m (i.e. >40% of shape diameter). For values below, use 0.25*config.in_radius
    patch_center_indices = tree.query(sub_pc,k=1,return_distance=False)
    
    return hd_points.squeeze(),NN.squeeze(),cloud_features.squeeze(),tree,patch_center_indices.squeeze()

def make_batches(a, batch_size):
    return np.split(a.squeeze(), np.arange(batch_size,len(a),batch_size))

def get_scene_seg_features(input_features_dim, features):
    rem = abs(3 - input_features_dim%3)%3

    if rem>0:
        ones = torch.ones((features.shape[0],rem)).type(torch.float32)
        features = torch.cat([ones,features], -1)

    return features.transpose(0, 1).contiguous()

def get_patch(config, point_ind, cur_cloud_tree, cur_features, cur_clouds_points_labels):
        """
        Returns:
            current_points: (N, 3), a point cloud.
            mask: (N, ), 0/1 mask to distinguish padding points.
            features: (input_features_dim, N), input points features.
            current_points_labels: (N), point label.
            input_inds: (N), the index of input points in point cloud.
        """
        points = np.array(cur_cloud_tree.data, copy=False)
        center_point = points[point_ind, :].reshape(1,3)#.reshape(1, -1)
        pick_point = center_point


        # Indices of points in input region
        query_inds = cur_cloud_tree.query_radius(pick_point,r=config.in_radius,
                                                            return_distance=True,
                                                            sort_results=True)[0][0]

        # Number collected
        cur_num_points = query_inds.shape[0]
        if config.num_points < cur_num_points:
            # choice = np.random.choice(cur_num_points, config.num_points)
            # input_inds = query_inds[choice]
            shuffle_choice = np.random.permutation(np.arange(config.num_points))
            input_inds = query_inds[:config.num_points][shuffle_choice]
            mask = torch.ones(config.num_points,).type(torch.int32)
        else:
            shuffle_choice = np.random.permutation(np.arange(cur_num_points))
            query_inds = query_inds[shuffle_choice]
            padding_choice = np.random.choice(cur_num_points, config.num_points - cur_num_points)
            input_inds = np.hstack([query_inds, query_inds[padding_choice]])
            mask = torch.zeros(config.num_points,).type(torch.int32)
            mask[:cur_num_points] = 1

        original_points = points[input_inds]
        current_points = (original_points - pick_point)


        current_points_labels = torch.from_numpy(cur_clouds_points_labels[input_inds].squeeze()).contiguous().type(torch.int64)

        current_features = cur_features[input_inds,:]
        current_features = torch.from_numpy(current_features).type(torch.float32)

        # adds ones at the beginning of feature vector
        features = get_scene_seg_features(current_features.shape[1], current_features)

        return [torch.from_numpy(current_points).float(), mask, features,
                       current_points_labels, torch.from_numpy(input_inds)]
    
def get_batch(config, patch_indices, cur_cloud_tree, cur_features, cur_clouds_points_labels):
    batch_size = patch_indices.shape[0]
    
    pts_ls = []
    mask_ls = []
    features_ls = []
    label_ls = []
    indices_ls = []
    for point_ind in patch_indices:
        [current_points, mask, features,
                       current_points_labels, input_inds] = get_patch(config, point_ind, cur_cloud_tree, cur_features, cur_clouds_points_labels)
        pts_ls.append(current_points[None,...])
        mask_ls.append(mask[None,...])
        features_ls.append(features[None,...])
        label_ls.append(current_points_labels[None,...])
        indices_ls.append(input_inds[None,...])
    points_batch = torch.cat(pts_ls,dim=0)
    mask_batch = torch.cat(mask_ls,dim=0)
    features_batch = torch.cat(features_ls,dim=0)
    label_batch = torch.cat(label_ls,dim=0)
    indices_batch = torch.cat(indices_ls,dim=0)
    return points_batch,mask_batch,features_batch,label_batch,indices_batch


def run_inference_on_cloud(config,model,batch_size,patch_center_indices,hd_pointcloud,cur_tree,cur_features,NN=None,device="cuda:0",verbose=True):
    if "cuda" not in device:
        raise ValueError("device={} is an incorrect value. The model has to be on a cuda device. Valid values: 'cuda:0', 'cuda:1', ..., 'cuda:N' where N is the number of available GPUs.".format(device))
    model = model.to(device)
    
    cur_labels = - np.ones((cur_features.shape[0],)).astype(np.int32)
    
    vote_logits_sum = np.zeros((config.num_classes, hd_pointcloud.shape[0]), dtype=np.float32)
    vote_counts = np.zeros((1, hd_pointcloud.shape[0]), dtype=np.float32) + 1e-6
    vote_logits = np.zeros((config.num_classes, hd_pointcloud.shape[0]), dtype=np.float32)

    batch_ls = make_batches(patch_center_indices,batch_size)
    num_batches = len(batch_ls)
    for i in range(num_batches):
        if verbose:
            print("\tInference on batch {:04d}/{:04d}".format(i+1,num_batches))
        patch_indices = batch_ls[i]
        points,mask,features,label,input_inds = get_batch(config, patch_indices, cur_tree, cur_features, cur_labels)

        with torch.no_grad():
            # prediction
            pred = model(points.to(device), mask.to(device), features.to(device))

        # collect
        bsz = points.shape[0]
        for ib in range(bsz):
            mask_i = mask[ib].cpu().numpy().astype(np.bool)
            logits = pred[ib].cpu().numpy()[:, mask_i]
            inds = input_inds[ib].cpu().numpy()[mask_i]
            vote_logits_sum[:, inds] = vote_logits_sum[:, inds] + logits
            vote_counts[:, inds] += 1
            vote_logits = vote_logits_sum / vote_counts
    
    probas = softmax(vote_logits, axis=0)
    probas_01 = probas[1,:].squeeze()
    count = vote_counts.squeeze()

    if NN is not None:
        NN = NN.squeeze()
        probas_01 = probas_01[NN]
        count = count[NN]
    
    return probas_01