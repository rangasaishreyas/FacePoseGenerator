#!/usr/bin/env python
import argparse
import os
import logging as logging
from accelerate.logging import get_logger
import shutil
import torch
import torch.optim
import torch.nn.functional as F

import torchvision.transforms as transforms


# from utils.dataset import DataLoaderX
from utils.utils_logging import AverageMeter #, init_logging
from utils.utils_callbacks import (
    CallBackLogging,
    CallBackVerification,
    CallBackModelCheckpointOld,
)
from utils.losses import ArcFace, CosFace, AdaFace
from backbones.iresnet import iresnet100, iresnet50, iresnet34, iresnet18
import json 


# from utils.dataset import CustomImageFolder
# from limited_image_folder_dataset import LimitedWidthAndDepthImageFolder
from tqdm.auto import tqdm 
from accelerate import Accelerator

from accelerate.utils import set_seed
import sys
import config.test_FR_config as cfg

def init_logging(logger, output_dir):
    logfile= "training.log"
    logger.logger.handlers = []
    logger.logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter("(%(asctime)s): %(message)s", datefmt='%Y-%m-%d %H:%M')
    
    handler_file = logging.FileHandler(os.path.join(output_dir, logfile))
    handler_file.setFormatter(formatter)
    logger.logger.addHandler(handler_file)

    handler_stream = logging.StreamHandler(sys.stdout)
    handler_stream.setFormatter(formatter)
    logger.logger.addHandler(handler_stream)
    
   

def main(args):    
    

    accelerator = Accelerator()
    accelerator.print(args)
    #accelerator.print("===" * 30)

    for model in cfg.models: 
        
        #seed = 42
        seed = int(args.run_index)
        set_seed(seed)

        cfg.model = model 

        if args.embedding_type != "None":
            cfg.embedding_type = args.embedding_type

        if args.architecture != "None":
            cfg.architecture = args.architecture

        if args.width != 0:
            cfg.width = args.width

        if args.depth != 0:
            cfg.depth = args.depth

        cfg.val_path = cfg.benchmark_folder 

        #cfg.val_targets = os.listdir(cfg.val_path) # ["lfw"] 
        #cfg.val_targets = [bench.split(".bin")[0] for bench in cfg.val_targets] # ["lfw",  "calfw"]#[bench.split(".bin")[0] for bench in cfg.val_targets] #["lfw", "calfw"]# # ["lfw"] 
    

        # rec = os.path.join(cfg.dataset_folder, cfg.model, cfg.embedding_type)  # training dataset
        # model_dir = os.path.join(cfg.root_folder, "Face_recognition_training", f"{cfg.output_folder_name_start}_{cfg.loss}_{cfg.dataset_folder.split('/')[-2]}_AUG{cfg.augment}/{cfg.dataset_folder.split('/')[-1]}", cfg.architecture, cfg.model) #cfg.embedding_type + f"_w{cfg.width}_d{cfg.depth}")
        model_dir = os.path.join(cfg.root_folder, "Face_recognition_training", f"{cfg.output_folder_name_start}/{cfg.architecture}_AUG{cfg.augment}/{args.run_index}_{cfg.dataset_folder.split('/')[-1]}", cfg.model) 
        
        # root_model_folder = f"EXPERIMENTS_RECOGNITION_{cfg.model.split('/')[2].split('_')[3]}"
        #model_dir = os.path.join(root_model_folder, cfg.architecture, os.path.join(cfg.model.split("/")[-2], cfg.model.split("/")[-1]))
        results_dir = os.path.join(cfg.root_folder, "Face_recognition_training", f"REC_RESULTS/{cfg.architecture}_AUG{cfg.augment}/{args.run_index}_{cfg.dataset_folder.split('/')[-1]}") #cfg.embedding_type + f"_w{cfg.width}_d{cfg.depth}")
        os.makedirs(results_dir, exist_ok=True)
        
        file_name_to_save_to = f"{cfg.model}.json"
        

        logger = get_logger("__Train__")
        init_logging(logger, results_dir)
        
        logger.info(f"Model: {cfg.model}")
        logger.info(f"All models: {cfg.models}")
        logger.info(f"Output: {results_dir}")
        
        logger.info(f"Train on dataset: {cfg.dataset_folder}")
        logger.info(f"Val targets: {cfg.val_targets}")
        logger.info(f"Output dir: {results_dir}")
        logger.info("===" * 30)
        

        if os.path.exists(os.path.join(results_dir, file_name_to_save_to)): 
            logger.info("These results already exist, so skip them!")
            logger.info("in:", os.path.join(results_dir, file_name_to_save_to))
            logger.info("===" * 30)
            #exit()
            break 
        
        # if cfg.augment:
        #     transform = get_conventional_aug_policy(cfg.augmentation)

        ###############################################
        ####### Create Model + resume Training ########
        ###############################################
        logger.info("=> creating model '{}'".format(cfg.architecture))

        if cfg.architecture == "resnet18":
            model_class = iresnet18
        elif cfg.architecture == "resnet50":
            model_class = iresnet50
        else:
            raise NotImplementedError

        model = model_class(num_features=cfg.embedding_size, dropout=cfg.dropout_ratio).to(accelerator.device)

        backbone_pth = os.path.join(
            model_dir, "best_backbone.pth"# #980"best_backbone.pth" #"646_backbone.pth"#
        )

        model.load_state_dict(
            torch.load(backbone_pth, map_location=torch.device(accelerator.device))
        )
        
            
        if accelerator.is_main_process:
            logging.info("backbone resume loaded successfully!")
        #except (FileNotFoundError, KeyError, IndexError, RuntimeError):
        #    logging.info("load backbone resume init, failed!")
        
        # Send everything through accelerator.prepare
        # model = accelerator.prepare(model)
        model.eval()

        ###############################################
        ################# Callbacks ###################
        ###############################################

        # callback_verification = CallBackVerification(1, accelerator.process_index, cfg.val_targets, cfg.val_path, img_size=[112, 112], logger=logger)
        # callback_checkpoint = CallBackModelCheckpointOld(accelerator.process_index, results_dir)

    
        global_step = 0
        avg_acc = torch.zeros(1).to(accelerator.device)

        best_benchmark_accs = dict()

        for val_target in cfg.val_targets: 
            best_benchmark_accs[val_target] = []



        print("=="*30) 
        print("Do verification benchmarks:")
        list_of_accs = []
        print(cfg.val_targets)
        for val_target in cfg.val_targets:

            callback_verification = CallBackVerification(
                1, accelerator.process_index, [val_target], cfg.val_path, img_size=[112, 112], logger=logger
                )

            ver_accs = callback_verification(global_step, model)
            print("Ver accs", ver_accs)
            best_benchmark_accs[val_target] = callback_verification.highest_acc_list
            list_of_accs.append(ver_accs[0])
            print("=="*30) 
        
        ver_accs = list_of_accs
        print(best_benchmark_accs)
        
        avg_acc = sum([v[0] / len(best_benchmark_accs.values()) for v in best_benchmark_accs.values()])
        #avg_acc[0] = sum(ver_accs[:5]) / len(ver_accs)
        print("Avg accs:", avg_acc)
        best_benchmark_accs["Average"] = [avg_acc]

        saving_to_path = os.path.join(results_dir, file_name_to_save_to)
        print("Saving to json file:", saving_to_path)
        
        with open(saving_to_path, "w") as outfile:
            json.dump(best_benchmark_accs, outfile, indent=4)

        print("==" * 30)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    parser.add_argument("--resume", type=int, default=0, help="resume training")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")

    parser.add_argument("--model", type=str, default="None", help="name of generative difusion model, e.g. unet-cond-ca-bs512-150K")
    parser.add_argument("--architecture", type=str, default="None", help="name of architecture, e.g. resnet18 or resnet50")
    parser.add_argument("--embedding_type", type=str, default="None", help="name of embedding type")

    parser.add_argument("--width", type=int, default=0, help="number of ids")
    parser.add_argument("--depth", type=int, default=0, help="samples per ids")
    parser.add_argument("--run_index", type=str, default="0", help="samples per ids")


    args = parser.parse_args()
    #print(args)
    main(args)
