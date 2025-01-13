#!/usr/bin/env python
import argparse
import os
import logging as logging
from accelerate.logging import get_logger
import shutil
import torch
from torch.nn import CrossEntropyLoss
import torch.backends.cudnn as cudnn
# import torch.distributed as dist
import torch.optim
#from torch.utils.data import ConcatDataset
import torch.utils.data.distributed
import torch.nn.functional as F

from utils.augmentation import get_conventional_aug_policy
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
# from torch.nn.parallel.distributed import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_

# from utils.dataset import DataLoaderX
from utils.utils_logging import AverageMeter #, init_logging
from utils.utils_callbacks import (
    CallBackLogging,
    CallBackVerification,
    CallBackModelCheckpointOld,
)
from utils.losses import ArcFace, CosFace, AdaFace
from backbones.iresnet import iresnet100, iresnet50, iresnet34, iresnet18


# from utils.dataset import CustomImageFolder
# from limited_image_folder_dataset import LimitedWidthAndDepthImageFolder

from tqdm.auto import tqdm 
from accelerate import Accelerator

from accelerate.utils import set_seed
import sys
from utils.dataset import ArcBiFaceGANDataset
import config.FR_config as cfg

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

        rec = os.path.join(cfg.dataset_folder, cfg.model, cfg.embedding_type)  # training dataset
        output_dir = os.path.join(cfg.root_folder, "Face_recognition_training", f"{cfg.output_folder_name_start}/{cfg.architecture}_AUG{cfg.augment}/{args.run_index}_{cfg.dataset_folder.split('/')[-1]}", cfg.model) #cfg.embedding_type + f"_w{cfg.width}_d{cfg.depth}")
        os.makedirs(output_dir, exist_ok=True)
        
        logger = get_logger("__Train__")
        init_logging(logger, output_dir)
        
        logger.info(f"Model: {cfg.model}")
        logger.info(f"All models: {cfg.models}")
        logger.info(f"Output: {output_dir}")
        
        logger.info(f"Train on dataset: {cfg.dataset_folder}")
        logger.info(f"Val targets: {cfg.val_targets}")
        logger.info(f"Output dir: {output_dir}")
        logger.info("===" * 30)
        

        if os.path.exists(os.path.join(output_dir, "best_backbone.pth")): 
            logger.info("This has already been trained so skip it!")
            logger.info("===" * 30)
            #exit()
            break 

        #logging.basicConfig(level=logging.INFO)
        
        

        # copy config to output folder
        shutil.copyfile(
            r"./Face_recognition_training/config/FR_config.py", os.path.join(output_dir, "config.py")
        )

        ###############################################
        ################ Data Loading #################
        ###############################################
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        )
        
        if cfg.augment:
            transform = get_conventional_aug_policy(cfg.augmentation)

        # accelerator.print("Basic dataset")
        # trainset = ImageFolder(rec, transform=transform)
        trainset = ArcBiFaceGANDataset(rec, transform=transform)

        num_classes = trainset.classes

        logger.info(f"Length of dataset: {len(trainset)}")
        logger.info(f"Number of classes: {num_classes}")

        train_loader = torch.utils.data.DataLoader(
            dataset=trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )
        
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

        model = model_class(num_features=cfg.embedding_size, dropout=cfg.dropout_ratio)#.to(accelerator.device)
        
        start_epoch = 0
        header = None

        if cfg.loss == "ArcFace":
            header = ArcFace(in_features=cfg.embedding_size, out_features=num_classes, s=cfg.s, m=cfg.m,)
        elif cfg.loss == "CosFace":
            header = CosFace(in_features=cfg.embedding_size, out_features=num_classes, s=cfg.s, m=cfg.m)
        elif cfg.loss == "AdaFace":
            header = AdaFace(embedding_size=cfg.embedding_size, classnum=num_classes)
        else:
            accelerator.print("Header not implemented")

        if args.resume:
            try:
                header_pth = os.path.join(output_dir, str(cfg.global_step) + "header.pth")
                header.load_state_dict(torch.load(header_pth, map_location=torch.device(accelerator.device)))

                if accelerator.is_main_process:
                    logger.info("header resume loaded successfully!")
            except (FileNotFoundError, KeyError, IndexError, RuntimeError):
                logger.info("header resume init, failed!")

        

        ###############################################
        ######### loss function + optimizer ###########
        ###############################################
        criterion = CrossEntropyLoss()

        opt_backbone = torch.optim.SGD(
            params=[{"params": model.parameters()}],
            lr=cfg.learning_rate / 512 * cfg.batch_size * accelerator.num_processes,
            momentum=0.9,
            weight_decay=cfg.weight_decay,
        )
        opt_header = torch.optim.SGD(
            params=[{"params": header.parameters()}],
            lr=cfg.learning_rate / 512 * cfg.batch_size * accelerator.num_processes,
            momentum=0.9,
            weight_decay=cfg.weight_decay,
        )

        scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(
            optimizer=opt_backbone, lr_lambda=cfg.lr_func
        )
        scheduler_header = torch.optim.lr_scheduler.LambdaLR(
            optimizer=opt_header, lr_lambda=cfg.lr_func
        )

        if cfg.auto_schedule:
            scheduler_backbone = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt_backbone, mode="max", factor=0.1, patience=4
            )
            scheduler_header = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt_header, mode="max", factor=0.1, patience=4
            )


        # Send everything through accelerator.prepare
        train_loader, model, header, opt_backbone, opt_header, scheduler_backbone, scheduler_header = accelerator.prepare(
            train_loader, model, header, opt_backbone, opt_header, scheduler_backbone, scheduler_header
        )

        model.train()
        header.train()

        ###############################################
        ################# Callbacks ###################
        ###############################################
        total_step = int(len(trainset) / cfg.batch_size / accelerator.num_processes * cfg.num_epoch)
        if accelerator.is_main_process:
            logger.info("Total Step is: %d" % total_step)
        callback_logging = CallBackLogging(
            cfg.print_freq, accelerator.process_index, total_step, cfg.batch_size, accelerator.num_processes, logger=logger
        )

        
        callback_verification = CallBackVerification(1, accelerator.process_index, cfg.val_targets, cfg.val_path, img_size=[112, 112], logger=logger)

        callback_checkpoint = CallBackModelCheckpointOld(accelerator.process_index, output_dir)

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        global_step = 0
        e_acc = torch.zeros(2).to(accelerator.device)  # (epoch, maxAcc)
        avg_acc = torch.zeros(1).to(accelerator.device)

        best_benchmark_accs = dict()

        for val_target in cfg.val_targets: 
            best_benchmark_accs[val_target] = []

        ###############################################
        ################## Training ###################
        ###############################################
        reached_best = False 

        logger.info("Begin training")
        for epoch in range(start_epoch, cfg.num_epoch):
            
            # Show only one progress bar
            progress_bar = tqdm(
                range(0, int(len(trainset) / (cfg.batch_size * accelerator.num_processes) )),
                desc="Steps",
                disable=True,#not accelerator.is_local_main_process,
            )
            
            # train for one epoch
            for i, (images, labels) in enumerate(train_loader):

                features = F.normalize(model(images))
                
                if cfg.loss == "AdaFace":
                    norm = torch.norm(features, 2, 1, True)
                    output = torch.div(features, norm)
                    thetas = header(output, norm, labels)
                else:
                    thetas = header(features, labels)

                loss_v = criterion(thetas, labels)
                
                accelerator.backward(loss_v)
                
                clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
                clip_grad_norm_(header.parameters(), max_norm=5, norm_type=2)

                opt_backbone.step()
                opt_header.step()

                opt_backbone.zero_grad()
                opt_header.zero_grad()
                            
                losses.update(loss_v.item(), 1)
                acc = multi_class_acc(thetas, labels)
                top1.update(acc)
                top5.update(0) 
                
                # initialize logging
                if global_step == 0: 
                    callback_logging(global_step, losses, top1, top5, epoch)
                
                
                progress_bar.update(1) #cfg.batch_size * accelerator.num_processes)
                global_step += 1
                # break 

            progress_bar.close()

            #print("Do callback logging")
            callback_logging(global_step, losses, top1, top5, epoch)
            callback_checkpoint(global_step, model, header)

            logger.info(f"End of epoch {epoch}. Perform validation:" )

            if accelerator.is_main_process:
                model.eval()
                header.eval()

                if len(cfg.val_targets) != 0:
                    #list_of_accs = []
                    #logger.info(f"Verification benchmarks: {cfg.val_targets}")

                    ver_accs = callback_verification(global_step, model)
                    avg_acc[0] = sum(ver_accs) / len(ver_accs)
                    # ver_accs = [round(ver_acc, 4) for ver_acc in ver_accs ]
                    logger.info(f"Avg. acc: {avg_acc[0]:.4f}, Ver. accs: {ver_accs}")
                    logger.info("---" * 30)
                    if cfg.auto_schedule:
                        scheduler_backbone.step(avg_acc)
                        scheduler_header.step(avg_acc)
                    else:
                        scheduler_backbone.step()
                        scheduler_header.step()
                    
                    # update max accuracy
                    if accelerator.process_index == 0 and avg_acc[0] > e_acc[1]:
                        e_acc[0] = epoch
                        e_acc[1] = avg_acc[0]
                        if cfg.auto_schedule:
                            callback_checkpoint(global_step, model, header)
                            callback_checkpoint("best", model, header, saving_best=True)

                    if cfg.auto_schedule and cfg.stopping_condition_epochs != 0 and e_acc[0] <= epoch - cfg.stopping_condition_epochs:
                        if not cfg.stop_only_after_epoch_schedule or (cfg.stop_only_after_epoch_schedule and epoch > cfg.schedule[-1]):
                            # callback_checkpoint("best", model, header, saving_best=True)
                            logger.info(f"Avg validation accuracy on BENCHMARKS did not improve for {cfg.stopping_condition_epochs} epochs. Terminating at epoch {epoch} with best average {e_acc[1]} at epoch {e_acc[0]}")
                            #exit()
                            reached_best = True 
                            #logger.info("TODO not breaking just in case")
                            break
                
                model.train()
                header.train()

            top1.reset()
            top5.reset()
            
            if reached_best: 
                break 
            
            #accelerator.print("==" * 30)
        

@torch.no_grad()
def multi_class_acc(pred, labels):
    a_max = torch.argmax(pred, dim=1)
    acc = (a_max == labels).sum().item() / labels.size(0)
    acc = round(acc * 100, 2)
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    parser.add_argument("--resume", type=int, default=0, help="resume training")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")

    parser.add_argument("--model", type=str, default="None", help="name of generative difusion model, e.g. unet-cond-ca-bs512-150K")
    parser.add_argument("--architecture", type=str, default="None", help="name of architecture, e.g. resnet18 or resnet50")
    parser.add_argument("--embedding_type", type=str, default="None", help="name of embedding type")

    parser.add_argument("--width", type=int, default=0, help="number of ids")
    parser.add_argument("--depth", type=int, default=0, help="samples per ids")
    parser.add_argument("--run_index", type=str, default="0", help="Consecutive number of the current run.")

    args = parser.parse_args()
    #print(args)
    main(args)
