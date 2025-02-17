# ID-Booth: Identity-consistent Face Generation with Diffusion Models

<div align="center">
  Darian Tomašević, Fadi Boutros, Naser Damer, Peter Peer, Vitomir Štruc
  <br>
  <br>
  <a href='https://arxiv.org/abs/2403.11641'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
  <br>
  <br>
</div>  
<div align="center">
        
</div>
This is the official implementation of the ID-Booth framework, which:

&emsp;🔥 generates in-the-wild images of consenting identities captured in a constrained environment <br>
&emsp;🔥 uses a triplet identity loss to fine-tune Stable Diffusion for identity-consistent yet diverse image generation <br>
&emsp;🔥 can augment small-scale datasets to improve their suitability for training face recognition models  <br>


## <div align="center"> Results </div>
<div align="center">
  <p>
    <img width="80%" src="./assets/preview_samples.jpg">
  </p>
</div>

## <div align="center"> Framework </div>
<div align="center">
  <p>
    <img width="80%" src="./assets/preview_framework.jpg">
  </p>
</div>

## <div align="center"> Setup </div>

```bash
conda create -n id-booth python=3.10
conda activate id-booth

# Install requirements
pip install -r requirements.txt
TODO
```


## Training 
use identity embeddings extracted with pretrained ArcFace recognition model (TODO weights)

```
TODO
```

## Usage

Load trained LoRA weights with [diffusers](https://huggingface.co/docs/diffusers/index):
```python
TODO
```



## Citation

If you use code or results from this repository, please cite the ID-Booth paper:

```
TODO
```

## Acknowledgements

Supported in parts by the Slovenian Research and Innovation Agency ARIS through the Research Programmes P2-0250(B) "Metrology and Biometric Systems" and P2--0214 (A) “Computer Vision”, the ARIS Project J2-2501(A) "DeepBeauty" and the ARIS Young Researcher Program.

<img src="./docs/ARIS_logo_eng_resized.jpg" alt="ARIS_logo_eng_resized" width="400"/>



