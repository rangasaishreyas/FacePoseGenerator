import numbers
import os
import queue as Queue
import random
import threading

import mxnet as mx
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
from PIL import Image


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank,
                                                 non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))
    
    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)


class FaceDatasetFolder(Dataset):
    def __init__(self, root_dir, local_rank):
        super(FaceDatasetFolder, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = os.path.join( root_dir,"synthetic")
        self.local_rank = local_rank
        self.imgidx, self.labels=self.scan(self.root_dir)
    def scan(self,root):
        imgidex=[]
        labels=[]
        lb=0
        list_dir=os.listdir(root)
        #list_dir.sort()
        for img in list_dir:
                imgidex.append(os.path.join(root,img))
                labels.append(lb)
        return imgidex,labels
    def readImage(self,path):
        return cv2.imread(os.path.join(self.root_dir,path))

    def __getitem__(self, index):
        path = self.imgidx[index]
        img=self.readImage(path)
        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.long)
        sample = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)
    



class CustomImageFolder(Dataset):
    def __init__(self, root_dir, transform, limited_dataset):
        super(CustomImageFolder, self).__init__()
        #self.transform = transforms.Compose(
             #[transforms.ToPILImage(),
             #transforms.Resize((112,112)),
             #transforms.RandomHorizontalFlip(),
             #transforms.ToTensor(),
             #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             #])
        self.transform = transform

        self.limited_dataset = limited_dataset
        self.root_dir = root_dir
        self.image_list = os.listdir(self.root_dir)
        self.classes = int(np.array([int(file.split("_")[0]) for file in self.image_list]).max()) + 1
        
        
        if self.limited_dataset: 
            self.image_list = self.image_list[:10000]
        


    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        
        label = int(img_name.split("_")[0])
        label = torch.tensor(label, dtype=torch.long)

        if self.transform is not None:
            image = self.transform(image)

        #image = self.transform(image)
        #print("Label:", label, img_name)
        #print("Shape", image.shape)
        #print("label:" , label)
        return image, label
    

class DreamBooth_dataset(Dataset):
    def __init__(self, root_dir, transform):
        super(DreamBooth_dataset, self).__init__()
        #self.transform = transforms.Compose(
             #[transforms.ToPILImage(),
             #transforms.Resize((112,112)),
             #transforms.RandomHorizontalFlip(),
             #transforms.ToTensor(),
             #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             #])
        self.transform = transform

        self.root_dir = root_dir
        self.image_list = os.listdir(self.root_dir)
        self.classes = int(np.array([int(file.split("_")[0]) for file in self.image_list]).max()) + 1
        print("Root:", self.root_dir)
        print("Num images:", len(self.image_list))

  
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        
        label = int(img_name.split("_")[0])
        label = torch.tensor(label, dtype=torch.long)

        if self.transform is not None:
            image = self.transform(image)

        #image = self.transform(image)
        #print("Label:", label, img_name)
        #print("Shape", image.shape)
        #print("label:" , label)
        return image, label
    


class ArcBiFaceGANDataset(Dataset):
    def __init__(self, root_dir, transform):
        super(ArcBiFaceGANDataset, self).__init__()
        #self.transform = transforms.Compose(
             #[transforms.ToPILImage(),
             #transforms.Resize((112,112)),
             #transforms.RandomHorizontalFlip(),
             #transforms.ToTensor(),
             #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             #])
        self.transform = transform

        self.root_dir = root_dir
        self.image_list = os.listdir(self.root_dir)
        self.classes = int(np.array([int(file.split("_")[0]) for file in self.image_list]).max()) + 1
        print("Root:", self.root_dir)
        print("Num images:", len(self.image_list))

  
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        
        label = int(img_name.split("_")[0])
        label = torch.tensor(label, dtype=torch.long)

        if self.transform is not None:
            image = self.transform(image)

        #image = self.transform(image)
        #print("Label:", label, img_name)
        #print("Shape", image.shape)
        #print("label:" , label)
        return image, label
    
    

class ArcBiFaceGANDataset_VISNIR(Dataset):
    def __init__(self, root_dir, transform):
        super(ArcBiFaceGANDataset_VISNIR, self).__init__()
        self.transform = transform

        self.root_dir = root_dir
        self.NIR_root_dir = self.root_dir.replace("VIS", "NIR")
        self.image_list = os.listdir(self.root_dir)
        self.NIR_image_list = os.listdir(self.NIR_root_dir)
        
        self.classes = int(np.array([int(file.split("_")[0]) for file in self.image_list]).max()) + 1
        print("Root:", self.root_dir)
        print("Num images:", len(self.image_list))
        print("Num NIR images:", len(self.NIR_image_list))
  
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        NIR_img_name = self.NIR_image_list[idx]
        NIR_img_path = os.path.join(self.NIR_root_dir, NIR_img_name)
        NIR_image = Image.open(NIR_img_path).convert('L')
        
        image = np.asarray(image)
        NIR_image = np.asarray(NIR_image) #transforms.ToTensor(NIR_image)
        
        #print(image.shape)
        #print(NIR_image.shape)
        NIR_image = NIR_image[..., np.newaxis]
        #print(NIR_image.shape)
        
        image = np.concatenate([image,NIR_image], axis=2)
        #print(image.shape)
        image = Image.fromarray(image)
        #print(image)
        #exit()
        label = int(img_name.split("_")[0])
        label = torch.tensor(label, dtype=torch.long)

        if self.transform is not None:
            image = self.transform(image)


        return image, label
    
