architecture = "resnet50"#"resnet50"#"resnet50"

# root_folder = "/shared/home/darian.tomasevic/ID-Booth/"
root_folder = "/home/darian/Desktop/Diffusion/ID-Booth/"
#dataset_folder = "../Generated_Split_Images_112x112/"

# dataset_folder = f"{root_folder}/FR_DATASETS/FR_DATASETS_SDXL/SDXL_DB_LoRA_Tufts_base_prompt_16_07_png"
#dataset_folder = f"{root_folder}/FR_DATASETS/FR_DATASETS_SDXL/SDXL_DB_LoRA_Tufts_combined_16_07_png"

#dataset_folder = f"{root_folder}/FR_DATASETS/FR_DATASETS_AUGMENTED_+21_samples/SDXL_DB_LoRA_Tufts_base_prompt_16_07_png"

# AlphawW0.1 
# dataset_folder = f"{root_folder}/FR_DATASETS/12-2024_SD21_LoRA4_alphaW0.1_Expr_Env"
# dataset_folder = f"{root_folder}/FR_DATASETS/12-2024_SD21_LoRA4_alphaW0.1_Age_Expr"
# dataset_folder = f"{root_folder}/FR_DATASETS/12-2024_SD21_LoRA4_alphaW0.1_Age"

# AlphaW None 
# dataset_folder = f"{root_folder}/FR_DATASETS/12-2024_SD21_LoRA4_alphaWNone_Expr_Env"
# dataset_folder = f"{root_folder}/FR_DATASETS/12-2024_SD21_LoRA4_alphaWNone_Age_Expr"
# dataset_folder = f"{root_folder}/FR_DATASETS/12-2024_SD21_LoRA4_alphaWNone_Age"


#dataset_folder = f"{root_folder}/FR_DATASETS/FR_DATASETS_AUGMENTED_+10_samples/SDXL_DB_LoRA_Tufts_combined_16_07_png"
# dataset_folder = f"{root_folder}/FR_DATASETS/FR_DATASETS_AUGMENTED_+10_samples/SDXL_DB_LoRA_Tufts_base_prompt_16_07_png"
folder_to_test = "12-2024_SD21_LoRA4_alphaW0.1_FINAL_FacePortraitPhoto_Gender_Pose_BackgroundB"
# folder_to_test = "12-2024_SD21_LoRA4_alphaW0.1_FINAL_FacePortraitPhoto_Gender_Pose_AgePhases_Expression_BackgroundB"

dataset_folder = f"{root_folder}/FR_DATASETS/{folder_to_test}"

# # TODO 
#dataset_folder = f"{root_folder}/FR_DATASETS/FR_DATASETS_SDXL/tufts_512_poses_1-7_all_imgs_jpg_per_ID"
#models = ["images"]

models = ["no_new_Loss", "identity_loss_TimestepWeight", "triplet_prior_loss_TimestepWeight"]


model = "TODO"
benchmark_folder = f"{root_folder}/FR_training/VALIDATION_DATASETS_from_webface"
augment = False 
stopping_condition_epochs = 6 #6 #6 # 10 # TODO was 6 
stop_only_after_epoch_schedule = False 


output_folder_name_start = "REC_EXP_01_2025_LFW_MultiRun_Verification1_CosFace"#_AllBench"

EMBEDDING_TYPE = [
    "."
]

embedding_type = EMBEDDING_TYPE[0]

width = 0
depth = 0

batch_size = 128 # 128  # 256
workers = 8  # 32
embedding_size = 512
learning_rate = 0.1
momentum = 0.9
weight_decay = 5e-4

global_step = 0  # to resume
start_epoch = 0

s = 64.0
m = 0.35
loss = "CosFace"#""
dropout_ratio = 0.4

augmentation = "ra_4_16"  # hf, ra_4_16

verification_frequency = 1
print_freq = 1 #50
val_path = "TODO"#"/data/Biometrics/database/faces_emore"  # "/data/fboutros/faces_emore"
val_targets = ["lfw"]
# val_targets = ["lfw", "agedb_30", "cfp_fp", "calfw", "cplfw"]


auto_schedule = True
num_epoch = 200
schedule = [22, 30, 35]


def lr_step_func(epoch):
    return (
        ((epoch + 1) / (4 + 1)) ** 2
        if epoch < -1
        else 0.1 ** len([m for m in schedule if m - 1 <= epoch])
    )


lr_func = lr_step_func

