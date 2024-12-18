#!/usr/bin/env bash
#DATASET="ExFaceGAN_SG3"
DATASET="Synth_100_subset_preprocess_both_classes"

geteerinf -p "data_plots/"$DATASET -i "impostors.txt" -g "genuines.txt" -sp "data_plots/"$DATASET -e $DATASET
