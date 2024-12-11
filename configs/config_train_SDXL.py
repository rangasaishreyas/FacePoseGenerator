
mixed_precision = "fp16"

pretrained_model_name_or_path =  "stabilityai/stable-diffusion-xl-base-1.0"
pretrained_vae_model_name_or_path = "madebyollin/sdxl-vae-fp16-fix" # TODO was None

logging_dir = "Logs"


gradient_accumulation_steps = 1
gradient_checkpointing = True # TODO memory 

report_to = "tensorboard"

revision = None 
variant = None 

use_8bit_adam  = True  # TODO memory 
enable_xformers_memory_efficient_attention = False 
sample_batch_size = 1 # 4

scale_lr = False # TODO  

# "Number of hard resets of the lr in cosine_with_restarts scheduler."
lr_num_cycles = 1
# Power factor of the polynomial scheduler
lr_power = 1.0

dataloader_num_workers = 0
train_text_encoder = False

# can precompute text encodings! 

checkpoints_total_limit = None 
resume_from_checkpoint = None 

seed = 0

source_folder = "../tufts_512_poses_1-7_all_imgs_jpg_per_ID/images"
resolution = 1024 

instance_prompt = "photo of sks person" # "face portrait photo of fid person"

with_prior_preservation = True  
class_prompt = "photo of a person" #"face portrait photo of a person" #
class_data_dir = "../CLASS_IMAGES/SDXL_Class_imgs_200/images" # "./CLASS_IMAGES/epiCRealism_SD15/images"
num_class_images = 200 
prior_loss_weight = 1.0

validation_prompt = "photo of sks person with blue hair"#"photo of [ID] person, portrait"# photo of [ID] person with blue hair"  #"face portrait photo of fid person with blue hair" #  
validation_negative_prompt = "" #"cartoon, cgi, render, illustration, painting, drawing, black and white, bad body proportions, landscape" 

optimizer = "adamw"
adam_beta1 = 0.9 
adam_beta2 = 0.999
adam_weight_decay = 1e-2
adam_epsilon = 1e-08
max_grad_norm = 1.0

do_edm_style_training = False
snr_gamma = None
random_flip = False # TODO could also flip, but faces can be non-symetrical 

lr_warmup_steps = 0 # TODO 500? 
lr_scheduler = "cosine" 
num_validation_images = 4

# "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training."
allow_tf32 = True 
prior_generation_precision = "fp16" # Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Default to fp16 if a GPU is available, otherwise fp32.

local_rank = 1 
tokenizer_max_length = None 
tokenizer_name = None
text_encoder_use_attention_mask = False

class_labels_conditioning = None 
max_train_steps = None

# Parameters to experiment with 
train_batch_size = 1 
lora_rank = 4
use_dora = False 

learning_rate = 1e-4 # 1e-4 # TODO  
num_train_epochs = 5 # 10 #2 # TODO
checkpointing_epochs = 1
validation_epochs = 1 

losses_to_test = ["","identity", "triplet_prior"] # TODO #[ "", "syn_moco_identity"]

timestep_loss_weighting = True 

output_folder = f"OUTPUT_MODELS/12-2024_SDXL_LoRA{lora_rank}/" 
show_tqdm = True