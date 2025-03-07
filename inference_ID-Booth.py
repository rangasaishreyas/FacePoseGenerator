from diffusers import StableDiffusionPipeline
import torch
import os 
from torchvision.utils import save_image
from diffusers import DPMSolverMultistepScheduler
from diffusers import DDPMScheduler
import random 
from accelerate.utils import  set_seed
from tqdm import tqdm 
import re 
import json 
#from compel import Compel
from diffusers import AutoPipelineForText2Image
from itertools import product 
from utils.sorting_utils import natural_keys

backgrounds_list = ["","forest", "city street", "beach", "office", "bus", "laboratory", "factory", "construction site", "hospital", "night club"]
#backgrounds_list = ["in the forest", "in the city", "at the beach", "at the office", "in the bus", "in the laboratory", "at the factory", "at the construction site", "at the hospital", "at the night club"]
backgrounds_list = [f"{b} background"  if b != "" else "" for b in backgrounds_list]#

# backgrounds_list = [""] + backgrounds_list * 2

# expression_list = ["neutral", "happy", "sad", "angry", "shocked"]# "crying", "ashamed"]
# expression_list = [f"{e} expression"  if e != "" else "" for e in expression_list]# "sad", "frowning", "surprised", "angry"]
# expression_list = ["", "happy", "sad", "angry", "shocked"]

# face_alterations = ["", "curly hair", "long hair", "short hair", "face tattoos", "sunglasses"]
# face_alterations = [f"with {alter}" if alter != "" else alter for alter in face_alterations]

# clothing = ["", "turtleneck", "sweater", "t-shirt", "shirt", "suit", "hat and sunglasses", "glasses"]
# clothing = [f"wearing a {cloth}" if cloth != "" else cloth for cloth in clothing]

# ages = ["20", "30", "40", "50", "60"]
# ages = [f"{a} years old" for a in ages]
age_phases = ["", "young", "middle-aged", "old"]


num_samples_per_prompt = 1
num_prompts = 21 # 21 #21 #50 #21 #len(additions_list)
add_gender = True

add_pose = True
add_age = False # should be first in combination
add_background = True 
use_non_finetuned = True 


# all_prompt_combinations = list(product(backgrounds_list, expression_list))
# all_prompt_combinations = list(backgrounds_list) #list(product(ages, expression_list))#, backgrounds_list))
if add_age and add_background: 
    all_prompt_combinations = list(product(age_phases, backgrounds_list))
elif add_background: 
    if num_prompts == 100:
        all_prompt_combinations = list(backgrounds_list[1:] * 10)
    else:
        all_prompt_combinations = list([""] + backgrounds_list[1:] * 2)
    
elif add_age: 
    all_prompt_combinations = list(age_phases * 6)

print(all_prompt_combinations)

device = "cuda:0"
seed = 0 
guidance_scale = 5.0
num_inference_steps = 30 #30

which_model_folder = "12-2024_SD21_LoRA4_alphaWNone"  #0.1 # None
folder_of_models = f"OUTPUT_MODELS/{which_model_folder}" #0.1
checkpoint =  "checkpoint-31-6400"  # "checkpoint-19-4000" #9-2000 #"checkpoint-12-2600"# "checkpoint-12-2600" 
models_to_test = ["no_new_Loss", "identity_loss_TimestepWeight", "triplet_prior_loss_TimestepWeight"]

folder_output = f"GENERATED_SAMPLES/{which_model_folder}_FacePortrait_Photo_21"  # _NonFinetuned
if add_gender: folder_output += "_Gender"
if add_pose: folder_output+= "_Pose"
if add_age: folder_output+= "_Age"
if add_background: folder_output += "_Background"

architectures = ["stabilityai/stable-diffusion-2-1-base"]
model_architecture = architectures[0]
arch = model_architecture.split("/")[1]

set_seed(seed)

width, height = 512, 512

ids = os.listdir(os.path.join(folder_of_models, models_to_test[0]))
ids = [i for i in ids if ".json" not in i]
ids.sort(key=natural_keys)
# ids = ids[:2]

print(ids)
if add_gender:
    with open("tufts_gender_dict.json", "r") as fp:
        gender_dict = json.load(fp)

# ids = ids[:1]

negative_prompt = "cartoon, cgi, render, illustration, painting, drawing, black and white, bad body proportions, landscape" # "cartoon, cgi, render, illustration, painting, drawing, black and white, bad body proportions" #"blurry, fake skin, plastic skin, cartoon, grayscale, painting, monochrome"
# original_prompt =  "photo of sks person"#, aligned close-up portrait""#,  50 years old, sunglasses, forest"#, face tattoos"#, 10 years old"
original_prompt = f"face portrait photo of sks person"

# generate seeds for each prompt number 
num_seeds = len(ids) 

prompt = ""

for id_number, which_id in enumerate(ids):
    print("\n", which_id) 

    if add_gender: 
        gender = gender_dict[which_id]
        if gender == "M": gender = "male"
        elif gender == "F": gender = "female"
    
    
    all_prompts_for_id = random.sample(all_prompt_combinations, num_prompts)
    
    comparison_image_list = [] 
    for model_name in models_to_test:

        full_model_path = os.path.join(folder_of_models, model_name, which_id, checkpoint)

        output_dir = os.path.join(folder_output, model_name)#"GENERATED_SAMPLES/FINAL_No_ID_loss_TEST"
        print("Load:", full_model_path)
        
        
        pipe = StableDiffusionPipeline.from_pretrained(model_architecture, torch_dtype=torch.float16).to(device)
        
        #print("original scheduler:", pipe.scheduler.config)
        pipe.scheduler = DDPMScheduler.from_pretrained(model_architecture, subfolder="scheduler")
        #print("New scheduler:", pipe.scheduler.config)
        
        if not use_non_finetuned:
            pipe.load_lora_weights(full_model_path)
        pipe.set_progress_bar_config(disable=True)

        #compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
        #prompt_embeds = compel_proc(f"{prompt}.and()")
        
        os.makedirs(output_dir, exist_ok=True)
        generator = torch.Generator(device=device).manual_seed(id_number) 

        for i, num_prompt in enumerate(tqdm(range(num_prompts))): 
            prompt_additions = all_prompts_for_id[i]
            prompt = original_prompt
            if add_age:
                age_insert = ""
                if isinstance(prompt_additions, str): age_insert = prompt_additions
                else: 
                    age_insert = prompt_additions[0] 
                    prompt_additions = prompt_additions[1:]
                if age_insert != "": prompt = prompt.replace(" sks person", f" {age_insert} sks person")
                

            if add_gender: prompt = prompt.replace(" sks person", f" {gender} sks person")
            if add_pose and random.choice([True, False]): prompt = prompt.replace("portrait", "side-portrait")

            if add_background:
                if isinstance(prompt_additions, str): 
                    prompt += f", {prompt_additions}"  
                else:
                    for addition in prompt_additions:
                        if addition != "":
                            prompt += f", {addition}"

            # generate samples
            for j in range(num_samples_per_prompt):  
                output = pipe(prompt=prompt, negative_prompt=negative_prompt,  output_type="np", generator=generator, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, width=width, height=height)
                output = torch.Tensor(output.images)
                comparison_image_list.append(output)
                output = torch.permute(output, (0, 3, 1, 2))
                path_to_output = f"{output_dir}/{which_id}_{checkpoint}_{arch}"
                os.makedirs(path_to_output, exist_ok=True)
                save_image(output, fp=f"{path_to_output}/{i}_{j}_{prompt}.png")
        
    images = torch.cat(comparison_image_list)
    images = torch.permute(images, (0, 3, 1, 2)) # permute dimensions to be (x, 3, 512, 512)
    #print(images.shape)
    #exit()
    print("Saving comparison image") 
    comparison_folder = "COMPARISON"

    comparison_folder = os.path.join(folder_output, comparison_folder)
    os.makedirs(comparison_folder, exist_ok=True)
    save_path = f"{comparison_folder}/{which_id}_{checkpoint}_{arch}_{guidance_scale}.jpg"
    print(save_path)
    save_image(images, fp=save_path, nrow=num_prompts*num_samples_per_prompt, padding=0)
    #break
