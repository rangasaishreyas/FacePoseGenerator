import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import sys
import inspect
from tqdm import tqdm
from multiprocessing import Process, Queue
import pandas as pd

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import seaborn as sns 

rcParams.update({"figure.autolayout": True})



############################
# create figures with only nice distribution plots first
def plot_score_histogram(ax, df, eer):
    TU_DESIGN_COLORS = {
        'Genuine': "#64a0d9",#"#009D81",
        'Imposter': "#d99d64", #"#0083CC",
        'random': "#721085", #"#FDCA00",
        'eer': "#EC6500"
    }
    sns.histplot(ax=ax,
                    data=df, x="scores",
                    hue="label", palette=TU_DESIGN_COLORS,
                    stat="probability", kde=True, bins=100, binrange=(-1, 1))

    ax.axvline(x=eer, c=TU_DESIGN_COLORS['eer'])

    
    labels = ['Genuine', 'Imposter'] 
    handles = [mpatches.Patch(color=TU_DESIGN_COLORS[label], label=label) for label in labels]
    ax.legend(handles=handles, labels=labels, loc="upper left", title="")
    
    ax.set_title(f"TODO")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Probability")
    
    ax.set_ylim(0, 0.075)

############################

def plot_gen_imp_distribution(datadir, num_ids=0, num_imgs=0):
    """plots genuine impostor distribution and saves genuine and impostor scores
    args:
        datadir: directory containing embeddings folder
        num_ids: number of identities
        num_imgs: number of images per identity
    """
    dataname = args.datadir.split(os.path.sep)[-1]
    save_dir = os.path.join("data_plots", dataname + "_both_classes")
    genuine_file = os.path.join(datadir, "genuines.txt")
    impostor_file = os.path.join(datadir, "impostors.txt")
    gen_sims = np.loadtxt(genuine_file)
    impo_sims = np.loadtxt(impostor_file)
    print("Plot histogram...")
    savename = os.path.join(save_dir, "distribution_" + dataname + ".png")
    plt.hist(gen_sims, bins=200, alpha=0.5, label="Genuine Similarities", density=True)
    plt.hist(
        impo_sims, bins=200, alpha=0.5, label="Impostor Similarities", density=True
    )
    plt.xlabel("Cosine Similarity", size=14)
    plt.ylabel("Probability Density", size=14)
    plt.legend(loc="upper left")
    plt.savefig(savename)
    plt.close()
    print("Histogram saved")

    print("Plot histogram new...")

    #df = pd.DataFrame({'Genuine': gen_sims, 'Imposter': impo_sims})
    #x = ['Genuine', 'Imposter']
    #df_long = df.melt('x')
    print(np.array(gen_sims))
    

    gen_sims = np.array(gen_sims)
    impo_sims = np.array(impo_sims)
    
    df_gen = pd.DataFrame({'Genuine': gen_sims})
    df_impo = pd.DataFrame({'Imposter': impo_sims}) 

    

    fig = plt.figure(figsize=(8, 5))
    plot_score_histogram(plt.gca(), df_gen, eer)
    plt.tight_layout()
    #plt.savefig(save_path, dpi=256)
    #plt.close(fig)
    
    #sns.histplot(data=df_gen, stat="probability", bins=100, binrange=(-1.1, 1.1), kde=True, palette=TU_DESIGN_COLORS)
    #sns.histplot(data=df_impo, stat="probability", bins=100, binrange=(-1.1, 1.1), kde=True, palette=TU_DESIGN_COLORS)
    
    """
    sns.histplot(ax=ax,
                     data=model_dfs[model_name][frm_name][plot_type], x="scores",
                     hue="label", palette=TU_DESIGN_COLORS,
                     stat="probability", kde=True, bins=100, binrange=(-1, 1))
    """
    savename = os.path.join(save_dir, "NEW_distribution_" + dataname + ".png")
    #plt.hist(gen_sims / sum(gen_sims), bins=200, alpha=0.5, label="Genuine Similarities")#, density=True)
    #plt.hist(
    #    impo_sims / sum(impo_sims), bins=200, alpha=0.5, label="Impostor Similarities"#, density=True
    #)
    #plt.xlabel("Cosine Similarity", size=14)
    #plt.ylabel("Probability", size=14)
    #plt.legend(loc="upper left")
    #plt.savefig(savename)
    plt.close()
    print("Histogram saved")


def main(args):
    plot_gen_imp_distribution(
        args.datadir,
        args.num_ids,
        args.num_imgs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Study of datasets")
    parser.add_argument(
        "--datadir",
        type=str,
        default="/data/synthetic_imgs/ExFaceGAN_SG3",
        help="path to directory containing the image folder",
    )
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--num_ids", type=int, default=5000)
    parser.add_argument("--num_imgs", type=int, default=100)
    parser.add_argument("--eer", type=float, default=0)
    parser.add_argument(
        "--fr_path",
        default="path/to/pre-trained/FR/model.pth",
    )
    args = parser.parse_args()
    main(args)
