pretrained_model_name_or_path =  "stabilityai/stable-diffusion-2-1-base" 

mixed_precision = "fp16"
logging_dir = "Logs"
report_to = "tensorboard"

revision = None 
variant = None 

resume_from_checkpoint = None 

# Source images
# source_folder = "/shared/home/darian.tomasevic/ID-Booth/FACE_DATASETS/tufts_512_poses_1-7_all_imgs_jpg_per_ID/images"
source_folder = "../tufts_512_poses_1-7_all_imgs_jpg_per_ID/images"
resolution = 512 
instance_prompt = "photo of sks person" # "face portrait photo of fid person"

# Prior preservation images
with_prior_preservation = True  
class_prompt = "photo of a person" #"face portrait photo of a person" #
# class_data_dir = "/shared/home/darian.tomasevic/ID-Booth/CLASS_IMAGES/SD21_Class_imgs_200/images"
class_data_dir = "../CLASS_IMAGES/SD21_Class_imgs_200/images"
num_class_images = 200 
prior_loss_weight = 1.0

validation_prompt = "photo of sks person with blue hair"
validation_negative_prompt = "" #"cartoon, cgi, render, illustration, painting, drawing, black and white, bad body proportions, landscape" 
validation_prompt_path = "FACE_DATASETS/Samples_for_validation/validation_prompt.pt"

num_validation_images = 4

dataloader_num_workers = 0
use_8bit_adam  = False 
enable_xformers_memory_efficient_attention = False 
allow_tf32 = True # "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training."
prior_generation_precision = "fp16" # Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Default to fp16 if a GPU is available, otherwise fp32.

local_rank = 1 
tokenizer_max_length = None 
tokenizer_name = None
text_encoder_use_attention_mask = False
class_labels_conditioning = None 
max_train_steps = None

seed = 0

# Training parameters to experiment with 
lora_rank = 4
train_batch_size = 1
gradient_accumulation_steps = 1
gradient_checkpointing = False 

num_train_epochs = 32 
validation_epochs = 8  
checkpointing_epochs = 8 
checkpoints_total_limit = None 

learning_rate = 1e-4 # TODO  
lr_scheduler = "cosine" 
lr_warmup_steps = 0 # TODO 500? 

adam_beta1 = 0.9 
adam_beta2 = 0.999
adam_weight_decay = 1e-2
adam_epsilon = 1e-08
max_grad_norm = 1.0

scale_lr = False   
lr_num_cycles = 1 # "Number of hard resets of the lr in cosine_with_restarts scheduler."
lr_power = 1.0 # Power factor of the polynomial scheduler

train_text_encoder = False
pre_compute_text_embeddings = False


losses_to_test = ["", "identity", "triplet_prior"] 
timestep_loss_weighting = True 
alpha_id_loss_weighting = 0.1

sample_batch_size = 1 # 4

output_folder = f"Trained_LoRA_Models/" #_NoPriorAblation" 
show_tqdm = True