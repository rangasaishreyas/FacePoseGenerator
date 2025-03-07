import argparse
import os
import sys

#from evaluation.QualityModel import QualityModel
import random 

import cv2
import numpy as np
from sklearn.preprocessing import normalize

from iresnet import iresnet100, iresnet50
#from evaluation.FaceModel import FaceModel
import torch
from tqdm import tqdm 
import re 



class FaceModel():
    def __init__(self,model_prefix, model_epoch, ctx_id=7 , backbone="iresnet50"):
        self.gpu_id=ctx_id
        self.image_size = (112, 112)
        self.model_prefix=model_prefix
        self.model_epoch=model_epoch
        self.model=self._get_model(ctx=ctx_id,image_size=self.image_size,prefix=self.model_prefix,epoch=self.model_epoch,layer='fc1', backbone=backbone)
    def _get_model(self, ctx, image_size, prefix, epoch, layer):
        pass

    def _getFeatureBlob(self,input_blob):
        pass
    
    def get_feature(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (112, 112))
        a = np.transpose(image, (2, 0, 1))
        input_blob = np.expand_dims(a, axis=0)
        emb=self._getFeatureBlob(input_blob)
        emb = normalize(emb.reshape(1, -1))
        return emb

    def get_batch_feature(self, image_path_list, batch_size=16):
        print("Get batch feature")
        count = 0
        num_batch =  int(len(image_path_list) / batch_size)
        features = []
        quality_score=[]
        for i in tqdm(range(0, len(image_path_list), batch_size)):

            if count < num_batch:
                tmp_list = image_path_list[i : i+batch_size]
            else:
                tmp_list = image_path_list[i :]
            count += 1

            images = []
            for image_path in tmp_list:
                # print(image_path)
                image = cv2.imread(image_path)
                #print(image)
                image = cv2.resize(image, (112, 112))
                a = np.transpose(image, (2, 0, 1))
                images.append(a)
            input_blob = np.array(images)

            emb, qs = self._getFeatureBlob(input_blob)
            quality_score.append(qs)
            features.append(emb)
            #print("batch"+str(i))
        features = np.vstack(features)
        quality_score=np.vstack(quality_score)
        features = normalize(features)
        
        return features, quality_score



class QualityModel(FaceModel):
    def __init__(self, model_prefix, model_epoch, gpu_id, backbone):
        super(QualityModel, self).__init__(model_prefix, model_epoch, gpu_id, backbone)

    def _get_model(self, ctx, image_size, prefix, epoch, layer, backbone):
        weight = torch.load(os.path.join(prefix,epoch+"backbone.pth"))
        if (backbone=="iresnet50"):
            backbone = iresnet50(num_features=512, qs=1, use_se=False).to(f"cuda:{ctx}")
        else:
            backbone = iresnet100(num_features=512, qs=1, use_se=False).to(f"cuda:{ctx}")

        backbone.load_state_dict(weight)
        model = torch.nn.DataParallel(backbone, device_ids=[ctx])
        model.eval()
        return model

    @torch.no_grad()
    def _getFeatureBlob(self,input_blob):
        imgs = torch.Tensor(input_blob).cuda()
        imgs.div_(255).sub_(0.5).div_(0.5)
        feat, qs = self.model(imgs)
        return feat.cpu().numpy(), qs.cpu().numpy()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Root dir for evaluation dataset')
    parser.add_argument('--data_folder', type=str, default='./data',
                        help='Root dir for evaluation dataset')
    parser.add_argument('--pairs', type=str, default='pairs.txt',
                        help='lfw pairs.')
    parser.add_argument('--datasets', type=str, default='XQLFW',
                        help='list of evaluation datasets (,)  e.g.  XQLFW, lfw,calfw,agedb_30,cfp_fp,cplfw,IJBC.')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU id.')
    parser.add_argument('--model_path', type=str, default="/home/fboutros/LearnableMargin/output/ResNet50-COSQSArcFace_SmothL1",
                        help='path to pretrained evaluation.')
    parser.add_argument('--model_id', type=str, default="32572",
                        help='digit number in backbone file name')
    parser.add_argument('--backbone', type=str, default="iresnet50",
                        help=' iresnet100 or iresnet50 ')
    parser.add_argument('--score_file_name', type=str, default="quality_r50.txt",
                        help='score file name, the file will be store in the same data dir')
    

    #

    return parser.parse_args(argv)

def read_image_list(image_list_file, image_dir=''):
    image_lists = []
    with open(image_list_file) as f:
        absolute_list = f.readlines()
        for l in tqdm(absolute_list):
            image_lists.append(os.path.join(image_dir, l.rstrip()))

    print("Number of images:", len(image_lists))

    return image_lists, absolute_list



def atoi(text):
        return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def main(param):

    seed = 7

    experiment_folder = param.data_dir
    data_folder = param.data_folder # "samples_without_masks_5000_arcface_0.6"# "tufts_256_poses_1-7_aligned/validate/images"
    datasets = param.datasets.split(',')
    face_model = QualityModel(param.model_path,param.model_id, param.gpu_id, param.backbone)
    
    all_datasets_path = os.path.join(experiment_folder, data_folder) 
    all_datasets = os.listdir(all_datasets_path)

    # all_datasets = ["no_new_Loss_NoPrior"]
    # TODO ... 2500 random images or 5000 images (since some have 95 samples vs more ... but do we even need more samples ... maybe?) 

    number_of_images_for_comparison = 10000 # TODO ... just do all? (or 2000)
    sample_images_instead_of_using_all = True

    with torch.no_grad():
        for dataset in all_datasets:
            print("Do dataset:", dataset)
            if ".json" in dataset or "embed" in dataset: continue  

            #for specter in ["VIS", "NIR"]:
            image_dataset_path = os.path.join(all_datasets_path, dataset)
            id_list = os.listdir(image_dataset_path)
            image_list = []
                
            path_folder = os.path.join(image_dataset_path)
            if ".txt" in path_folder: 
                print("Skip txt file")
                continue 

            img_list = os.listdir(path_folder)
            img_list.sort(key=natural_keys)

            for img_name in img_list: 
                img_path = os.path.join(path_folder, img_name)
                image_list.append(img_path)

            print("Number of images in total, before sampling:", len(image_list))
            
            if sample_images_instead_of_using_all:
                print("Sample it")
                if len(image_list) > number_of_images_for_comparison: 
                    random.seed(seed)
                    image_list = random.sample(image_list, number_of_images_for_comparison)
                    print("Number of images, after sample:", len(image_list))

                else: 
                    print("Not enough samples in this dataset! Use all instead.")
                    # continue 
            
            embedding, quality = face_model.get_batch_feature(image_list, batch_size=16)

            path_to_output_folder = os.path.join("RESULTS_ID-Booth_FR_CRFIQA_12-2024", os.path.basename(experiment_folder))#experiment_folder, data_folder + "_" + param.score_file_name )
            os.makedirs(path_to_output_folder, exist_ok=True)
            
            path_to_scores = os.path.join(path_to_output_folder,  f"{dataset}.txt")

            print("Write scores to", path_to_scores)
            quality_score=open(path_to_scores,"w")
            for i in range(len(quality)):
                quality_score.write(image_list[i].rstrip()+ " "+str(quality[i][0])+ "\n")
            
            print("==" * 30)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))