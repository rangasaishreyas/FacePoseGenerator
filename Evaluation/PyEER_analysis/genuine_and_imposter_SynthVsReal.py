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

import random 
import numpy as np 

from create_boundary_data import extract_features
from utils.utils import save_emb_2_id, pairwise_cos_sim

rcParams.update({"figure.autolayout": True})


device_to_use = "cuda:0"

#########################################
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)



def split_gen_imp(embeddings_synth, embeddings_real, args, min_samples=3):
    """split embeddings in genuine and impostor pairs
    args:
        embeddings: list of embeddings per identity
    return:
        numpy array of genuine pairs, numpy array of impostor pairs
    """
    gens1, gens2, impos1, impos2 = [], [], [], []
    # embeddings = embeddings

    #print(embeddings[0][0][0])
    # TODO 
    print("Shuffle embeddings:")
    tmp = list(zip(embeddings_synth, embeddings_real))
    random.shuffle(tmp)
    embeddings_synth, embeddings_real = zip(*tmp)
    # res1 and res2 come out as tuples, and so must be converted to lists.
    embeddings_synth, embeddings_real = list(embeddings_synth), list(embeddings_real)
    #print(embeddings)
    num_ids = len(embeddings_synth)
    print("Create genuine and impostor pairs... num_ids:", num_ids)
    
    min_samples =  8 #1000 # 9
    samples_skip = 17 #1 #16

    for p in tqdm(range(num_ids)):        
        how_many_to_add = 0
        id_embs = embeddings_synth[p]
        id_real_embs = embeddings_real[p]
        
        for i in range(len(id_embs)):
            emb1 = id_embs[i]
            for j in range(i + 1, len(id_real_embs)):
                emb2 = id_real_embs[j]
                gens1.append(emb1)
                gens2.append(emb2)
                how_many_to_add+=1 
        num_id_embs = len(id_embs)
        num_id_samples = min(num_id_embs, min_samples) #4)
        
        #print("Do impostors")
        #print("P:", p)
        #print("num_ids:", num_ids)
        for ref_idx in range(p + 1, num_ids, samples_skip):
            #print(ref_idx)
            ref_id_embs = embeddings_real[ref_idx]
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


def generate_embeddings(args):
    """infer and save embeddings"""

    for img_dir in [args.datadir, args.realdir]:
        img_path = os.path.join(img_dir)
        emb_path = os.path.join(os.path.dirname(img_dir), "embeddings", os.path.basename(img_dir))

        os.makedirs(emb_path, exist_ok=True)
        num_ids = args.num_ids# len(os.listdir(img_path))
        print("Ids in Img_path", num_ids)
        if args.num_ids != 0:
            num_ids = args.num_ids

        embs = os.listdir(emb_path)
        print("Existing embeds", embs)

        if len(embs) > 0: 
            print("Using existing embeds, skipping extraction")
            continue 
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
    if num_ids != 0:
        files = files[:num_ids]
    embeddings = []
    for f in files:
        e = np.load(os.path.join(path, f))
        if num_imgs != 0:
            e = e[:num_imgs]
            #print(len(e))
        embeddings.append(e)
    return embeddings


#################################################
def create_genuine_and_impostor_files(args):

    folder = os.path.dirname(args.datadir)
    dataname = os.path.basename(args.datadir)#.split(os.path.sep)[1:]
    #path_names = args.datadir.split(os.path.sep)[1:]
    print(dataname)

    emb_path = os.path.join(folder, "embeddings", dataname)
    print("Load embeddings from:", emb_path)

    emb_real_path = os.path.join(os.path.dirname(args.realdir), "embeddings", os.path.basename(args.realdir))
    print("Load real embeddings from:", emb_real_path)
    embeddings_synth = load_embeddings(emb_path, args.num_ids, args.num_imgs)
    embeddings_real = load_embeddings(emb_real_path, args.num_ids, args.num_imgs)
    # print(len(embeddings))
    # exit()
    gens1, gens2, impos1, impos2 = split_gen_imp(embeddings_synth, embeddings_real, args)
    # exit()
    
    print("Calculating genuine cosine similarity...")
    gen_sims = calculate_cos_sim(gens1, gens2, nthreads=5)
    print("Calculating impostor cosine similarity...")
    #print(impos1.shape())
    impo_sims = calculate_cos_sim(impos1, impos2, nthreads=10)

    #dataname = args.datadir.split(os.path.sep)[1:]
    #dataname = os.path.join(args.datadir[-1], #args.datadir.split(os.path.sep)[2:]
    #dataname = "__".join(dataname)
    
    
    save_dir = os.path.join(args.outdir, dataname)
    os.makedirs(save_dir, exist_ok=True)
    print("Save to:", save_dir)
    genuine_file = os.path.join(save_dir, "genuines.txt")
    impostor_file = os.path.join(save_dir, "impostors.txt")
    np.savetxt(genuine_file, gen_sims)
    np.savetxt(impostor_file, impo_sims)
#################################################
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

#################################################
def run_create_gen_imp_files_SynthVsReal(datadir, realdir, fr_path, outdir):

    num_ids = 0 
    batchsize = 16
    num_imgs = 0
    seed = 0 #42 #0 #42  #100
    set_all_seeds(seed)
    
    args = Namespace(datadir = datadir,
                    realdir = realdir,
                    fr_path = fr_path,
                    outdir = outdir,
                    num_ids = num_ids,
                    batchsize = batchsize, 
                    num_imgs = num_imgs,
                    seed = seed)
    
    generate_embeddings(args)
    create_genuine_and_impostor_files(args)
#################################################