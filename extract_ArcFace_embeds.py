
import numpy as np 
import torch 
import json 
import os 

from Arcface_files.ArcFace_functions import prepare_locked_ArcFace_model
from facenet_pytorch import MTCNN
from PIL import Image 
from torchvision import transforms
from tqdm import tqdm 

def prepare_for_arcface_model_torch(img): 
    img = img.permute(2,0,1)
    img = transforms.functional.resize(img, (112, 112))
    img = img.float()
    img = ((img / 255) - 0.5) / 0.5 
    img = img[None, :, :, :]
    return img


origin_path = "FACE_DATASET"
device = "cuda:0"

arcface_model = prepare_locked_ArcFace_model()
arcface_model.to(device=device)

mtcnn_model = MTCNN(image_size=112,device=device, margin=0)

files_without_faces = dict()
files_without_faces["files_without_faces"] = []

folders = os.listdir(os.path.join(origin_path, "images"))

for folder in tqdm(folders):
    folder_path = os.path.join(origin_path, "images", folder)
    
    output_path = folder_path.replace("images", "ArcFace_embeds")
    os.makedirs(output_path, exist_ok=True)
    img_files = os.listdir(folder_path)
    
    images = []
    for img_name in (img_files):        
        img_path = os.path.join(folder_path,img_name)
        image = torch.from_numpy(np.array(Image.open(img_path))).to(device)
        images.append(image)
    
    images = torch.stack(images, dim=0)

    # image must be (x, x, 3) .. RGB
    # detect face bounding box
    bboxs, probs = mtcnn_model.detect(images, landmarks=False)

    images_cropped = []
    for image, bbox in zip(images, bboxs): 
        if bbox is None: 
            files_without_faces["files_without_faces"].append(img_path)
            continue 

        # crop to face
        bbox = bbox[0].astype(int)
        initial_size = image.shape[0]
        img_cropped = image[max(0,bbox[1]): min(bbox[3], initial_size ),
                            max(0, bbox[0]): min(bbox[2], initial_size)] 
        
        # transform to (1, 3, x, x)
        img_cropped = prepare_for_arcface_model_torch(img_cropped)
        images_cropped.append(img_cropped)

    images_cropped = torch.stack(images_cropped, dim=0)
    
    face_embeds = arcface_model(img_cropped)
    
    embed_file_name = folder + ".pt"
    torch.save(face_embeds, os.path.join(output_path, embed_file_name)) 
    
json_pth = f"{origin_path}/files_without_faces.json"

print(json_pth)

with open(json_pth, 'w') as fp:
    json.dump(files_without_faces, fp)