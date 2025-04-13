architecture = "resnet50"#"resnet50"

# root_folder = "/shared/home/darian.tomasevic/ID-Booth/"
root_folder = ".."
folder_to_test = "FacePortrait_Photo_21_Gender_Pose_Background"
models = ["DreamBooth", "PortraitBooth", "ID-Booth"]

# folder_to_test = "tufts_512_poses_1-7_all_imgs_jpg_per_ID"
# models = ["images"]

dataset_folder = f"{root_folder}/FR_DATASETS/{folder_to_test}"


model = "TODO"
benchmark_folder = f"{root_folder}/FR_training/VALIDATION_DATASETS_from_webface"
augment = False 
stopping_condition_epochs = 0
verification_frequency = 1
output_folder_name_start = f"REC_EXP_01_2025_LFW_Verification{verification_frequency}"

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
loss = "AdaFace"
dropout_ratio = 0.4

augmentation = "ra_4_16"  # hf, ra_4_16

print_freq = 1 #50
val_path = "TODO"#"/data/Biometrics/database/faces_emore"  # "/data/fboutros/faces_emore"
val_targets = ["lfw", "agedb_30", "cfp_fp", "calfw", "cplfw"]#"lfw"]#, "agedb_30", "cfp_fp", "calfw", "cplfw"]

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

