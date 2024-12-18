import os
import argparse
import numpy as np
from matplotlib import rcParams
import sys
import inspect
from tqdm import tqdm
from multiprocessing import Process, Queue

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import re 
import random 
import numpy as np 

from create_boundary_data import extract_features
from utils.utils import save_emb_2_id, pairwise_cos_sim

rcParams.update({"figure.autolayout": True})


device_to_use = "cuda:1"

#########################################
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)

import itertools

def split_gen_imp(embeddings, args, min_samples=3):
    """form genuine and impostor pairs
    args:
        gen_embs and imp_embs: list of embeddings per identity
    return:
        numpy array of genuine pairs, numpy array of impostor pairs
    """
    gens1, gens2, impos1, impos2 = [], [], [], []
    
    # unpack all ... 
    
    #print(len(gen_embs[0]))
    #indices = [i for i in range(len(gen_embs))]
    #print(indices)
    #gen_inds = random.sample(set(itertools.product(indices, indices)), 5)
    #print(gen_inds)

    #impos1 = random.sample(set(itertools.product(gen_embs, imp_embs)), 5)
    #print(len(gens1))
    #print(gens1)
    #embeddings = embeddings
    #print(embeddings[0][0][0])
    # TODO 
    #print("Shuffle embeddings:")
    #random.shuffle(embeddings) 
    #print(embeddings)
    num_ids = len(embeddings)

    limit_for_real_ids = 95 

    print("Create genuine and impostor pairs... num_ids:", num_ids)

    if "COMBINED" in args.datadir: 
        min_samples = 12
        samples_skip = 1 #10 

    elif num_ids == 1000: 
        min_samples = 4
        samples_skip = 16 #8 
    
    elif num_ids == 500: 
        min_samples = 5
        samples_skip = 12 

    if "ID_SPLIT" in args.datadir:
        min_samples = 6
        samples_skip = 9 

    elif num_ids == 95: 
        min_samples = 9
        samples_skip = 8 

    for p in tqdm(range(limit_for_real_ids)):#num_ids)):
        
        how_many_to_add = 0
        print("Ref (genuine) id:", p)
        id_embs = embeddings[p]
        for i in range(len(id_embs)):
            emb1 = id_embs[i]
            for j in range(i + 1, len(id_embs)):
                emb2 = id_embs[j]
                gens1.append(emb1)
                gens2.append(emb2)
                how_many_to_add+=1 
        num_id_embs = len(id_embs)
        num_id_samples = min(num_id_embs, min_samples) #4)
        #print(how_many_to_add)
        #print(num_id_samples)
        #num_id_samples = int(how_many_to_add / num_id_embs )
        #print(num_id_samples)
        
        
        #print("Do impostors")
        #print("P:", p)
        #print("num_ids:", num_ids)
        for ref_idx in range(limit_for_real_ids, num_ids, samples_skip):
            #print(ref_idx)
            print("Ref (imposter) id:", ref_idx)
            ref_id_embs = embeddings[ref_idx]
            num_ref_embs = len(ref_id_embs)
            num_ref_samples = min(num_ref_embs, min_samples)# 4)
            img1_idx = np.random.choice(num_id_embs, num_id_samples, replace=False)
            #print("new impostor:", img1_idx)
            for i in img1_idx:
                emb1 = id_embs[i]
                img2_idx = np.random.choice(
                    num_ref_embs, num_ref_samples, replace=False
                )
                for j in img2_idx:
                    emb2 = ref_id_embs[j]
                    impos1.append(emb1)
                    impos2.append(emb2)
                    #print("append")

    print("Num id samples", num_id_samples)
    print("Genuine shape:", len(gens1), len(gens2))
    print("Impostor shape:", len(impos1), len(impos2))
    return gens1, gens2, impos1, impos2


def genereate_embeddings(args):
    """infer and save embeddings"""
    img_path = os.path.join(args.datadir, "VIS")
    emb_path = os.path.join(args.datadir, "embeddings")
    
    #dataname = args.datadir.split(os.path.sep)[-1]
    #save_dir = os.path.join("analysis_data", dataname + "_both_classes")
    #emb_path = os.path.join(save_dir, "embeddings")
    #emb_path = os.path.join(args.datadir, "embeddings")

    os.makedirs(emb_path, exist_ok=True)
    num_ids = args.num_ids# len(os.listdir(img_path))
    print("Ids in Img_path", num_ids)
    if args.num_ids != 0:
        num_ids = args.num_ids

    embs = os.listdir(emb_path)
    print("Existing embeds", embs)

    if len(embs) > 0: 
        print("Using existing embeds, skipping extraction")
        return 
    #if num_ids != 0 and len(os.listdir(emb_path)) >= num_ids:#-10:
    #    print("Too many")
    #    return
    
    print("Extract features:")
    embs, img_paths = extract_features(
        img_path, args.batchsize, 0, args.fr_path, num_ids=num_ids, device=device_to_use
    )
    #print("Embs", embs.shape)
    save_emb_2_id(embs, img_paths, emb_path)


def split_list(l, num):
    """splits given list or numpy array into num amount of chunks
    args:
        l: list or numpy array
        num: amount of chunks
    return:
        list of lists or numpy arrays
    """
    chunk = len(l) // num
    out = []
    for i in range(num):
        start_idx = i * chunk
        if i == num - 1:
            out.append(l[start_idx:])
        else:
            out.append(l[start_idx : start_idx + chunk])
    return out


def cos_sim(emb1, emb2, queue):
    sims = pairwise_cos_sim(emb1, emb2)
    queue.put(sims)


def calculate_cos_sim(emb1, emb2, nthreads=10):
    """calculates the pairwise cosine similarity between both embeddings
    in multiple parallel processes.
    args:
        emb1: list of embeddings
        emb2: list of embeddings
        nthreads: number of parallel threads
    return:
        list of pairwise cosine similarities
    """
    q = Queue()
    processes = []
    emb_list1 = split_list(emb1, nthreads)
    emb_list2 = split_list(emb2, nthreads)
    cos_sims = []
    for embs1, embs2 in zip(emb_list1, emb_list2):
        p = Process(
            target=cos_sim,
            args=(embs1, embs2, q),
        )
        processes.append(p)
        p.start()

    for p in processes:
        ret = q.get()  # will block
        cos_sims.append(ret)
    for p in processes:
        p.join()
    return np.hstack(cos_sims)

#################################################
def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

#################################################
def load_embeddings(path, num_ids=0, num_imgs=0):
    """load embeddings from files in directory path
    args:
        path: path to directory containing embedding files
        num_ids: number of identities loaded; 0=all
        num_imgs: number of images per identity loaded; 0=all
    return:
        list of embeddings per identity [identities, embs_per_id, emb_dim]
    """
    
    files = sorted(os.listdir(path))
    print("Files in embeddings:", files)

    sort_nicely(files)
    print("Files sorted in embeddings:", files)

    if num_ids != 0:
        files = files[:num_ids]
    embeddings = []
    for f in files:
        print(f)
        e = np.load(os.path.join(path, f))
        if num_imgs != 0:
            e = e[:num_imgs]
            #print(len(e))
        embeddings.append(e)
    return embeddings


#################################################
def create_genuine_and_impostor_files(args):
    
    
    emb_path = os.path.join(args.datadir, "embeddings")
    print("Load embeddings from:", emb_path)

    # split after 95 
    embeddings = load_embeddings(emb_path, args.num_ids, args.num_imgs)

    #print(embeddings)
    #gen_embs = embeddings[:95]
    #imp_embs = embeddings[95:]

    gens1, gens2, impos1, impos2  = split_gen_imp(embeddings, args)
    #print(len(gen_embs))
    #print(len(imp_embs))
    
    # TODO here create a combination of ... gens and reals 
    #gens1, gens2, impos1, impos2 = split_gen_imp(embeddings, args)

    
    print("Calculating genuine cosine similarity...")
    gen_sims = calculate_cos_sim(gens1, gens2, nthreads=5)
    print("Calculating impostor cosine similarity...")
    #print(impos1.shape())
    impo_sims = calculate_cos_sim(impos1, impos2, nthreads=10)

    #dataname = args.datadir.split(os.path.sep)[1:]
    dataname = args.datadir.split(os.path.sep)[2:]
    dataname = "__".join(dataname)
    print(dataname)
    
    save_dir = os.path.join(args.outdir, dataname)
    os.makedirs(save_dir, exist_ok=True)
    genuine_file = os.path.join(save_dir, "genuines.txt")
    impostor_file = os.path.join(save_dir, "impostors.txt")
    np.savetxt(genuine_file, gen_sims)
    np.savetxt(impostor_file, impo_sims)


#################################################
def main(args):
    genereate_embeddings(args)
    create_genuine_and_impostor_files(args)
#################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Study of datasets")
    parser.add_argument(
        "--datadir",
        type=str,
        default="/data/synthetic_imgs/ExFaceGAN_SG3",
        help="path to directory containing the image folder",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="DATA_FOR_ANALYSIS",
        help="Path for output directory",
    )
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--num_ids", type=int, default=0)
    parser.add_argument("--num_imgs", type=int, default=0)
    parser.add_argument(
        "--fr_path",
        default="path/to/pre-trained/FR/model.pth",
    )
    args = parser.parse_args()

    seed = 7 #0 #42  #100
    set_all_seeds(seed)
    main(args)
