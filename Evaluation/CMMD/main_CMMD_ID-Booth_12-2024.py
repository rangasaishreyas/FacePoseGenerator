# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The main entry point for the CMMD calculation."""
import os 
from absl import app
from absl import flags
import distance
import embedding
import io_util
import numpy as np
import json 
import random 
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 32, "Batch size for embedding generation.")
_MAX_COUNT = flags.DEFINE_integer("max_count", -1, "Maximum number of images to read from each directory.")
_REF_EMBED_FILE = flags.DEFINE_string(
    "ref_embed_file", None, "Path to the pre-computed embedding file for the reference images."
)


def compute_cmmd(ref_dir, eval_dir, feature_save_folder,  max_number_of_samples, sample_it, seed, ref_embed_file=None, batch_size=32, max_count=-1):
    """Calculates the CMMD distance between reference and eval image sets.

    Args:
      ref_dir: Path to the directory containing reference images.
      eval_dir: Path to the directory containing images to be evaluated.
      ref_embed_file: Path to the pre-computed embedding file for the reference images.
      batch_size: Batch size used in the CLIP embedding calculation.
      max_count: Maximum number of images to use from each directory. A
        non-positive value reads all images available except for the images
        dropped due to batching.

    Returns:
      The CMMD value between the image sets.
    """
    # presaved_real_feat_path = f"{os.path.join(feature_save_folder, ref_dir.split('/')[-2])}.npy"
    presaved_real_feat_path = f"{os.path.join(feature_save_folder, ref_dir.split('/')[-2], ref_dir.split('/')[-1])}.npy"
    
    presaved_synth_feat_path = f"{os.path.join(feature_save_folder, eval_dir.split('/')[-2], eval_dir.split('/')[-1])}.npy"
    
    
    if ref_dir and ref_embed_file:
        raise ValueError("`ref_dir` and `ref_embed_file` both cannot be set at the same time.")
    embedding_model = embedding.ClipEmbeddingModel()
    print("Desired image size:", embedding_model.input_image_size)
    if ref_embed_file is not None:
        ref_embs = np.load(ref_embed_file).astype("float32")
    else:
        ref_embs = io_util.compute_embeddings_for_dir(ref_dir, embedding_model, batch_size, presaved_real_feat_path, 
                                                      max_number_of_samples, sample_it = sample_it, seed = seed, max_count= max_count).astype("float32")

    
    eval_embs = io_util.compute_embeddings_for_dir(eval_dir, embedding_model, batch_size, presaved_synth_feat_path, 
                                                   max_number_of_samples, sample_it = sample_it, seed = seed, max_count= max_count).astype("float32")
    val = distance.mmd(ref_embs, eval_embs)

    return val.numpy()


def main(argv):
    feature_save_folder = "CMMD_FR_Features_12-2024"
    if len(argv) != 3:
        raise app.UsageError("Too few/too many command-line arguments.")
    _, dir1, dir2 = argv
    
    max_number_of_samples = 2500
    sample_it = True 
    seed = 42 
    print("Sample it:", sample_it)
    print("MAX", _MAX_COUNT.value)
    score = float(compute_cmmd(dir1, dir2, feature_save_folder,  max_number_of_samples, sample_it, seed, _REF_EMBED_FILE.value, _BATCH_SIZE.value, _MAX_COUNT.value))
    print(
        "The CMMD value is: "
        f" {score:.3f}"
    )
    
    output_folder = f"CMMD_FR_Results_12-2024_Seed{seed}_2500/{dir1.split('/')[-2]}_vs_{dir2.split('/')[-2]}"
    os.makedirs(output_folder, exist_ok=True)
    
    output_filename = f"{dir2.split('/')[-1]}.json"
    #print("SAVE_TO:", output_filename)

    score_dict = {"CMMD": score}
    with open(os.path.join(output_folder, output_filename), "w") as outfile:
        json.dump(score_dict, outfile, indent=4)
        #print("==" * 30)

if __name__ == "__main__":
    app.run(main)
