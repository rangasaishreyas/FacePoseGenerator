import os 
import shutil 
from tqdm import tqdm 

root_folder = "../FR_DATASETS"
# With additional 10 or 21 synthetic images 
how_many_samples = 100
target_root = f"../FR_DATASETS_AUGMENTED_+{how_many_samples}_samples"

starting_dataset = f"{root_folder}/tufts_512_poses_1-7_all_imgs_jpg_per_ID/images"
# which_folders = ["12-2024_SD21_LoRA4_alphaWNone_FINAL_FacePortraitPhoto_Gender_Pose_BackgroundB"]
which_folders = ["12-2024_SD21_LoRA4_alphaWNone_FacePortrait_Photo_Gender_Pose_BackgroundB_100samples"]

for which_folder in which_folders:
    in_folder = os.path.join(root_folder, which_folder)
    model_folders = os.listdir(in_folder)

    for model_fold in tqdm(model_folders):
        
        print(model_fold)
        if "embeddings" in model_fold: continue 
        if ".json" in model_fold: continue 

        in_model = os.path.join(root_folder, which_folder, model_fold)
        
        # go across generated samples, copy them
        for img_name in os.listdir(in_model):
            # only copy how many samples 
            sample_number = int(img_name.split("_")[1])
            if sample_number >= how_many_samples:
                continue 

            src_img_path =  os.path.join(in_model, img_name)

            tar_fold = os.path.join(target_root, which_folder, model_fold)
            

            os.makedirs(tar_fold, exist_ok=True)

            tar_img_path =  os.path.join(tar_fold, img_name)

            shutil.copyfile(src_img_path, tar_img_path)

        # go across real samples, copy them to the same folder as well 
        for img_name in os.listdir(starting_dataset):

            src_img_path =  os.path.join(starting_dataset, img_name)
            tar_fold = os.path.join(target_root, which_folder, model_fold)
            
            tar_img_path =  os.path.join(tar_fold, img_name.replace(".jpg", ".png"))

            shutil.copyfile(src_img_path, tar_img_path)
        
