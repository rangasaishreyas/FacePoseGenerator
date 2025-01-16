import numpy as np
import sklearn.metrics
import torchvision
import os
import pathlib
import random
import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from tqdm import tqdm 
import json 
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import re 
import CLIP_model
from torchvision.models.feature_extraction import create_feature_extractor

"""
Based on: 
https://github.com/clovaai/generative-evaluation-prdc
""" 

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--num-workers', type=int,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--dims', type=int, default= 768,
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('path', type=str, nargs=2,
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}



def compute_pairwise_distance(data_x, data_y=None):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(
        data_x, data_y, metric='euclidean', n_jobs=8)
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def compute_prdc(real_features, fake_features, nearest_k):
    """
    Computes precision, recall, density, and coverage given two manifolds.

    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """

    print('Num real: {} Num fake: {}'
          .format(real_features.shape[0], fake_features.shape[0]))

    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, nearest_k)
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        fake_features, nearest_k)
    distance_real_fake = compute_pairwise_distance(
        real_features, fake_features)

    precision = (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).any(axis=0).mean()

    recall = (
            distance_real_fake <
            np.expand_dims(fake_nearest_neighbour_distances, axis=0)
    ).any(axis=1).mean()

    density = (1. / float(nearest_k)) * (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean()

    coverage = (
            distance_real_fake.min(axis=1) <
            real_nearest_neighbour_distances
    ).mean()

    return dict(precision=precision, recall=recall,
                density=density, coverage=coverage)





class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, tfs=None):
        self.files = files
        #self.transforms = torchvision.models.VGG16_Weights.IMAGENET1K_V1.transforms()
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        #print("Transforms:", self.transforms)
    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        img = self.transforms(img)
        img = img.permute(1, 2, 0)
        #print(img)
        #exit()
        return img

def sample_features(feats, max_number_of_samples, seed):
    
    #set_all_seeds(seed)
    if len(feats) > max_number_of_samples and max_number_of_samples != 0: 
        print("Sample it")
        feats = random.sample(feats, max_number_of_samples)
        print("Samples num. of images:", len(feats))
    
    feats = np.array(feats)
    return feats 

def load_precomputed_features(path, sample_it, max_number_of_samples):
    print("Load existing features from:", path)
    feats = np.load(path).tolist()
    print("Num loaded feats:", len(feats))

    
    return feats

#########################################
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

##############################################
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

#########################################
def extract_features(path, model, batch_size, dims, device, presaved_feat_path, save_features, seed, max_number_of_samples, num_workers=1):
    
    sample_it = True
    
    

    presaved_feat_path_2 = f"{os.path.dirname(presaved_feat_path)}.npy" 
    
    if  os.path.exists(presaved_feat_path):
        feats = load_precomputed_features(presaved_feat_path, sample_it, max_number_of_samples)
        if sample_it: feats = sample_features(feats, max_number_of_samples, seed)
        return feats
    elif os.path.exists(presaved_feat_path_2):
        feats = load_precomputed_features(presaved_feat_path_2, sample_it, max_number_of_samples)
        if sample_it: feats = sample_features(feats, max_number_of_samples, seed) 
        return feats
    
    
    file_paths = []
    print(path)

    files = os.listdir(path)
    files.sort(key=natural_keys)

    file_paths = [os.path.join(path, img_name) for img_name in files]
    ##for img_name in files:
    #   file_paths.append(os.path.join(path, img_name))

    #files = sorted(files)
    print("Number of images:", len(file_paths))

    dataset = ImagePathDataset(file_paths)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    
    #print(model)

    feats = np.empty((len(file_paths), dims))
    #print(feats.shape)
    start_idx = 0
    for batch in tqdm(dataloader):
        batch = batch.numpy()#to(device)
        #print(batch.shape)
        #batch = batch / 255.0

        with torch.no_grad():
            pred_feats = model.embed(batch)# ['Output'] TODO 
            #print(pred_feats.shape)
            #exit()
            pred_feats = pred_feats.cpu().numpy()
            feats[start_idx: (start_idx + pred_feats.shape[0])] = pred_feats
            start_idx = start_idx + pred_feats.shape[0]
    
    if save_features:
        os.makedirs(os.path.dirname(presaved_feat_path), exist_ok=True)
        np.save(presaved_feat_path, arr=feats)

    
    feats = feats.tolist()
    if sample_it:
        feats = sample_features(feats, max_number_of_samples, seed)
        
    return feats




def calculate_scores(paths, batch_size, device, dims, feature_save_folder, save_features, seed, max_number_of_samples, num_workers=1):
    """Calculates Precision, Recall, Coverage and Density of two paths with the Imagenet model"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)
    
    # imagenet_model = torchvision.models.vgg16(weights='IMAGENET1K_V1', progress=True).to(device)#, #weights="IMAGENET1K_FEATURES", progress=True).to(device)
    
    embedding_model = CLIP_model.ClipEmbeddingModel()
    # embedding_model = 
    # print(imagenet_model)
    # return_nodes = {
    #     "classifier.3": "Output"
    # }
    # imagenet_model_extractor = create_feature_extractor(imagenet_model, return_nodes=return_nodes)
    #imagenet_model.classifier = imagenet_model.classifier[:-1] 
    
    #imagenet_model_extractor = imagenet_model 
    #imagenet_model_extractor.eval()
    
    presaved_real_feat_path = f"{os.path.join(feature_save_folder, paths[0].split('/')[-2])}.npy"
    print(presaved_real_feat_path)
    feats_real = extract_features(paths[0], embedding_model, batch_size, dims, device, presaved_real_feat_path, save_features, seed, max_number_of_samples, num_workers=1)
    
    presaved_synth_feat_path = f"{os.path.join(feature_save_folder, paths[1].split('/')[-2], paths[1].split('/')[-1])}.npy"
    feats_synth = extract_features(paths[1], embedding_model, batch_size, dims, device, presaved_synth_feat_path, save_features, seed, max_number_of_samples, num_workers=1)

    nearest_k = 5

    metrics = compute_prdc(real_features = feats_real, fake_features = feats_synth, nearest_k=nearest_k)

    return metrics



def main():
    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        num_avail_cpus = len(os.sched_getaffinity(0))
        num_workers = min(num_avail_cpus, 8)
    else:
        num_workers = args.num_workers

    save_features = True 
    seed = 42
    max_number_of_samples = 5000 #2500
    print("seed:", seed)
    set_all_seeds(seed)

    feature_save_folder = f"../../ID-Booth/Evaluation/CMMD/CMMD_NonFR_Features_12-2024"# PRDC_ID-Booth_FR_Features/12-2024_CLIP_seed{seed}"
    prdc_values = calculate_scores(args.path, args.batch_size, device, args.dims, feature_save_folder, save_features, seed, max_number_of_samples, num_workers)
    
    result = {"PRDC": prdc_values}
    print(result)

    output_folder = f"PRDC_ID-Booth_NonFR_CLIP_RESULTS/12-2024_CLIP_seed{seed}_{max_number_of_samples}/{args.path[0].split('/')[-2]}_vs_{args.path[1].split('/')[-2]}"
    os.makedirs(output_folder, exist_ok=True)

    output_filepath = os.path.join(output_folder, f"{args.path[1].split('/')[-1]}.json")
    print("Save results to:", output_filepath)
    with open(output_filepath, "w") as outfile:
        json.dump(result, outfile, indent=4)
        
if __name__ == '__main__':
    main()
