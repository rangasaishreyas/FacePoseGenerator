import os
import cv2
from tqdm import tqdm
import argparse
from os.path import join as ojoin
from torch.utils.data import Dataset, DataLoader
import re 
import json 
from PIL import Image 
import numpy as np
from utils.sorting_utils import natural_keys
# from utils.align_trans import norm_crop
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN
from skimage import transform as trans


mtcnn = MTCNN(
    select_largest=True, post_process=False, device="cuda:0"
)


def load_syn_paths(datadir, num_imgs=0):
    img_files = sorted(os.listdir(datadir))
    img_files = img_files if num_imgs == 0 else img_files[:num_imgs]
    return [ojoin(datadir, f_name) for f_name in img_files]


def load_real_paths(datadir, num_imgs=0):
    img_paths = []
    id_folders = sorted(os.listdir(datadir))
    num_imgs = num_imgs if num_imgs != 0 else len(id_folders)
    for id in id_folders[: num_imgs]:
        img_files = sorted(os.listdir(ojoin(datadir, id)))
        img_paths += [ojoin(datadir, id, f_name) for f_name in img_files]
    return img_paths


def is_folder_structure(datadir):
    """checks if datadir contains folders (like CASIA) or images (synthetic datasets)"""
    img_path = sorted(os.listdir(datadir))[0]
    img_path = ojoin(datadir, img_path)
    return os.path.isdir(img_path)


class InferenceDataset(Dataset):
    def __init__(self, datadir, num_imgs=0, folder_structure=False):
        """Initializes image paths"""
        self.folder_structure = folder_structure
        if self.folder_structure:
            self.img_paths = load_real_paths(datadir, num_imgs)
        else:
            self.img_paths = load_syn_paths(datadir, num_imgs)
        #print("Amount of images:", len(self.img_paths))

    def __getitem__(self, index):
        """Reads an image from a file and corresponding label and returns."""
        img_path = self.img_paths[index]
        img_file = os.path.split(img_path)[-1]
        if self.folder_structure:
            tmp = os.path.dirname(img_path)
            img_file = ojoin(os.path.basename(tmp), img_file)
        # TODO RGB is correct, not CV2 BGR  
        og_img = Image.open(self.img_paths[index])

        width, height = og_img.size
        img = Image.new(og_img.mode, (width*2, height*2), 0)
        img.paste(og_img, (width, height))
        
        img = np.array(img)

        return img, img_file

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.img_paths)


def align_images(in_folder, out_folder, batchsize, id_fold="", num_imgs=0):
    """MTCNN alignment for all images in in_folder and save to out_folder
    args:
            in_folder: folder path with images
            out_folder: where to save the aligned images
            batchsize: batch size
            num_imgs: amount of images to align - 0: align all images
    """
    os.makedirs(out_folder, exist_ok=True)
    is_folder = is_folder_structure(in_folder)
    train_dataset = InferenceDataset(
        in_folder, num_imgs=num_imgs, folder_structure=is_folder
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batchsize, shuffle=False, drop_last=False, num_workers=2
    )

    skipped_imgs = []
    id_number = id_fold.split("_")[1]

    for img_batch, img_names in (train_loader):
        img_batch = img_batch.to("cuda:0")
        boxes, probs, landmarks = mtcnn.detect(img_batch, landmarks=True)

        img_batch = img_batch.detach().cpu().numpy()

        for img, img_name, box, landmark in zip(img_batch, img_names, boxes, landmarks):
            if box is None or len(box) == 0: 
                skipped_imgs.append(img_name)
                continue

            if is_folder:
                id_dir = os.path.split(img_name)[0]
                os.makedirs(ojoin(out_folder, id_dir), exist_ok=True)

            facial5points = landmark[0]
            warped_face = norm_crop(
                img, landmark=facial5points, image_size=112
            )
            warped_face = Image.fromarray(warped_face)
            # img_name = "%05d.png" % (counter)
            warped_face.save(os.path.join(out_folder, f"{id_number}_{img_name}"))
            
            #cv2.imwrite(os.path.join(out_folder, f"{id_number}_{img_name}"), warped_face)
            # counter += 1
    #print(skipped_imgs)
    #print(f"Images with no Face: {len(skipped_imgs)}")
    return skipped_imgs


# lmk is prediction; src is template
def estimate_norm(lmk, image_size=112):
    """estimate the transformation matrix
    :param lmk: detected landmarks
    :param image_size: resulting image size (default=112)
    :return: transformation matrix M and index
    """
    assert lmk.shape == (5, 2)
    assert image_size == 112
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float("inf")
    src = arcface_eval_ref_points
    #    src = arcface_ref_points

    src = np.expand_dims(src, axis=0)

    lmk = np.float32(lmk)
    src = np.float32(src)
    

    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = np.float32(results.T)
        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


# norm_crop from Arcface repository (insightface/recognition/common/face_align.py)
def norm_crop(img, landmark, image_size=112):
    """transforms image to match the landmarks with reference landmarks
    :param landmark: detected landmarks
    :param image_size: resulting image size (default=112)
    :return: transformed image
    """
    M, pose_index = estimate_norm(
        landmark, image_size=image_size
    )
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


arcface_ref_points = np.array(
    [
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041],
    ],
    dtype=np.float32,
)

# https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py
# [:,0] += 8.0
arcface_eval_ref_points = arcface_ref_points
arcface_eval_ref_points[:,0] += 8.0
##############################################

def main():
    parser = argparse.ArgumentParser(description="MTCNN alignment")
    parser.add_argument(
        "--in_folder",
        type=str,
        default="/data/maklemt/synthetic_imgs/DiscoFaceGAN_large",
        help="folder with images",
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        default="/home/maklemt/DiscoFaceGAN_aligned2",
        help="folder to save aligned images",
    )
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument(
        "--num_imgs",
        type=int,
        default=0,
        help="amount of images to align; 0 for all images",
    )

    args = parser.parse_args()
    
    # which_folders = os.listdir("GENERATED_SAMPLES")
    which_folders = ["12-2024_SD21_LoRA4_alphaW0.1_Expr_Env"]#, "SD21_pretrained_combined", "SDXL_pretrained_base_prompt", "SDXL_pretrained_combined"]
    for which_folder in which_folders:
        args.in_folder = f"GENERATED_SAMPLES/{which_folder}"
        print(which_folder)
        args.out_folder = f"FR_DATASETS/{args.in_folder.split('/')[-1]}"#"Testing_Naser_aligned_samples"
        args.batchsize = 8

        model_folders = os.listdir(args.in_folder)

        missing_images_dict = dict()
        for model_fold in model_folders:
            if "COMPARISON_" in model_fold: continue 
            print(model_fold)
            missing_images_dict[model_fold] = dict()
            id_folders = os.listdir(os.path.join(args.in_folder, model_fold))
            id_folders.sort(key=natural_keys)

            for id_fold in tqdm(id_folders):
                current_in_folder = os.path.join(args.in_folder, model_fold, id_fold)
                current_out_folder = os.path.join(args.out_folder, model_fold)
                missing_images = align_images(
                    current_in_folder,
                    current_out_folder,
                    args.batchsize,
                    id_fold=id_fold,
                    num_imgs=args.num_imgs
                )
                missing_images_dict[model_fold][id_fold] = missing_images
            
        # Convert and write JSON object to file
        with open(f"{args.out_folder}/missing_images.json", "w") as outfile: 
            json.dump(missing_images_dict, outfile, indent=4)
        
        #break
if __name__ == "__main__":
    main()