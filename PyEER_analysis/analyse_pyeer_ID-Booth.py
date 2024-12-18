import os
# num_ids = 0
# num_imgs = 32 
from pyeer.eer_info import get_eer_stats
import os 
import pandas as pd
import json 

import seaborn as sns 
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import sys
import inspect
#from multiprocessing import Process, Queue
import pandas as pd
from create_genuine_and_impostor_files import run_create_gen_imp_files


#dataset_folder = "../GENERATED_SAMPLES_Turbo/27-10_SD21_LoRA32_LR0.0001_GS0.0_Steps4"
#dataset_folder = "../Generated_Split_Images_112x112"
main_folder = "/shared/home/darian.tomasevic/ID-Booth/FR_DATASETS/"
dataset_folders = ["12-2024_SD21_LoRA4_alphaWNone_Expr_Env", "12-2024_SD21_LoRA4_alphaWNone_Age_Expr", "12-2024_SD21_LoRA4_alphaWNone_Age", 
                   "12-2024_SD21_LoRA4_alphaW0.1_Expr_Env", "12-2024_SD21_LoRA4_alphaW0.1_Age_Expr", "12-2024_SD21_LoRA4_alphaW0.1_Age"]

subfolders = ["no_new_Loss", "identity_loss_TimestepWeight", "triplet_prior_loss_TimestepWeight"]


# create pyeer report
report_which_metrics = [
    "eer",
    "eer_th",
    #"auc",
    "fmr100",
    "fmr1000",
    "gmean",
    "gstd",
    "imean",
    "istd",
    "fdr",
]

#########################
def compute_fdr(stats):
    return (stats["gmean"] - stats["imean"]) ** 2 / (stats["gstd"] ** 2 + stats["istd"] ** 2)

#########################

############################
# create figures with only nice distribution plots first
def plot_score_histogram(ax, df, stats, which_stat):
    TU_DESIGN_COLORS = {
        'Genuine': "#64a0d9",#"#009D81",
        'Imposter': "#d99d64", #"#0083CC",
        'random': "#721085", #"#FDCA00",
        'eer': "#E0221F"
    }
    sns.histplot(ax=ax,
                    data=df,
                    x="scores",
                    hue="label",
                    #palette=TU_DESIGN_COLORS, 
                    alpha=0.5,
                    stat=which_stat, kde=True, bins=100) # binrange=(-1, 1))

    ax.axvline(x=stats["eer_th"], c=TU_DESIGN_COLORS['eer'], linestyle="--")

    labels = [f'Genuine', f'Imposter'] 
    handles = [mpatches.Patch(color=TU_DESIGN_COLORS[label], label=label) for label in labels]

    genuine_info = f"${round(stats['gmean'], 3)} \pm {round(stats['gstd'], 3)}$" 
    imposter_info = f"${round(stats['imean'], 3)} \pm {round(stats['istd'], 3)}$"
    
    labels = [f'Genuine ({genuine_info})', f'Imposter ({imposter_info})']     
    ax.legend(handles=handles, labels=labels, loc="upper left", title="")

    ax.set_title(subfolder.replace("/", "__"), size=10)
    ax.set_xlabel("Cosine Similarity", size=14)
    ax.set_ylabel("Probability", size=14)

    #plt.legend(loc="upper left")
    #ax.set_ylim(0, 0.075)
############################


for dataset_folder in dataset_folders:
    dataset_folder = os.path.join(main_folder, dataset_folder)

    output_folder = os.path.join("RESULTS", os.path.basename(dataset_folder))

    for subfolder in subfolders:
        data_folder = os.path.join(dataset_folder, subfolder)
        which_FR_model = "backbones/ArcFace_r100_ms1mv3_backbone.pth"

        run_create_gen_imp_files(datadir=data_folder, fr_path=which_FR_model, outdir=output_folder)

        print(subfolder)
        folder = os.path.join(output_folder,  subfolder.replace("/","__"))
        print("Folder", folder)
        gscore_file = os.path.join(folder, "genuines.txt")
        iscore_file = os.path.join(folder, "impostors.txt")

        genuine_scores = pd.read_csv(gscore_file, header=None)[0].to_list()
        impostor_scores = pd.read_csv(iscore_file, header=None)[0].to_list()

        # Calculating stats
        stats = get_eer_stats(genuine_scores, impostor_scores)
        stats = stats._asdict()
        print(stats.keys())

        stats["fdr"] = compute_fdr(stats)
        
        saving_dict = dict()
        for metric in report_which_metrics:
            print(f"{metric}: {stats[metric]}")
            saving_dict[metric] = stats[metric]


        #with open(os.path.join(folder, "PyEER_report_" + subfolder.replace("/", "__"))+".json", "w") as outfile:
        with open(os.path.join(folder, "PyEER_report.json"), "w") as outfile:
            json.dump(saving_dict, outfile, indent=4)
            print("==" * 30)

        currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        parentdir = os.path.dirname(currentdir)
        sys.path.insert(0, parentdir)

        datadir =  os.path.join(output_folder, subfolder.replace("/", "__"))#"Synth_100_subset_preprocess_both_classes"
        print(datadir)
        save_dir = os.path.join(datadir)

        gen_sims = list(np.loadtxt(os.path.join(datadir, "genuines.txt")))
        impo_sims = list(np.loadtxt(os.path.join(datadir, "impostors.txt")))

        # Plot 
        df = pd.DataFrame()
        df['scores'] = gen_sims + impo_sims
        df['label'] = ['Genuine'] * len(gen_sims) + ['Imposter'] * len(impo_sims)

        fig = plt.figure()#figsize=(8, 8))
        plot_score_histogram(plt.gca(), df, stats, which_stat="probability")
        plt.tight_layout()
        savename = os.path.join(save_dir, "distribution_" + subfolder.replace("/", "__") + ".png")
        print("Saving to:", savename)
        plt.savefig(savename, dpi=256)
        plt.close(fig)

        print("====" * 30)