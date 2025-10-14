# DiT360: High-Fidelity Panoramic Image Generation via Hybrid Training

<p align="center">
  <a href='https://arxiv.org/abs/2510.11712'><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
  <a href='https://fenghora.github.io/DiT360-Page/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=insta360&logoColor=white' alt='Project Page'></a>
  <a href='https://huggingface.co/spaces/Insta360-Research/DiT360'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live_Demo-blue'></a>
  <a href='https://huggingface.co/datasets/Insta360-Research/Matterport3D_polished'><img src='https://img.shields.io/badge/%F0%9F%93%88%20Hugging%20Face-Dataset-yellow'></a>
</p>

![teaser](assets/teaser.jpg)

**DiT360** is a framework for high-quality panoramic image generation, leveraging both **perspective** and **panoramic** data in a hybrid training scheme.
It adopts a two-level strategy—**image-level cross-domain guidance** and **token-level hybrid supervision**—to enhance perceptual realism and geometric fidelity.

## 🌟 Features

<!-- <p align="center">
  <img src="assets/result.gif" width="90%">
</p> -->
![result](assets/result.gif)

- **High Perceptual Realism**: It produces panoramic images with high resolution and clear details for lifelike visual quality.
- **Precise Geometric Fidelity**: It correctly models multi-scale distortions in panoramic images with smooth, continuous edges.
- **Versatile Applications**: It enables robust handling of generated assets across multiple tasks, seamlessly supporting both inpainting and outpainting.


## ⏩ Updates
**15/10/2025**
- Release the training code.

**14/10/2025**
- Release our paper on arXiv.

**11/10/2025**
- Release the refined Matterport3D dataset.
  
**10/10/2025**
- Release the pretrained model and inference code.

## 🔨 Installation

Clone the repo first:

```Bash
git clone https://github.com/Insta360-Research-Team/DiT360.git
cd DiT360
```

(Optional) Create a fresh conda env:

```Bash
conda create -n dit360 python=3.12
conda activate dit360
```

Install necessary packages (torch > 2):

```Bash
# pytorch (select correct CUDA version, we test our code on torch==2.6.0 and torchvision==0.21.0)
pip install torch==2.6.0 torchvision==0.21.0

# other dependencies
pip install -r requirements.txt
```

## 🖼️ Dataset

We have uploaded the dataset to **Hugging Face**. For more details, please visit [Insta360-Research/Matterport3D_polished](https://huggingface.co/datasets/Insta360-Research/Matterport3D_polished).

For a quick start, you can try:

```python
from datasets import load_dataset

ds = load_dataset("Insta360-Research/Matterport3D_polished")

# check the data
print(ds["train"][0])
```
If you encounter any issues, please refer to the official Hugging Face documentation.

## 📒 Inference

```Bash
python inference.py
```

## 🤝 Acknowledgement

We appreciate the open source of the following projects:

* [diffusers](https://github.com/huggingface/diffusers)

## Citation
```
@misc{dit360,
      title={DiT360: High-Fidelity Panoramic Image Generation via Hybrid Training}, 
      author={Haoran Feng and Dizhe Zhang and Xiangtai Li and Bo Du and Lu Qi},
      year={2025},
      eprint={2510.11712},
      archivePrefix={arXiv},
}
```
If you find our dataset useful, please also include a citation for Matterport3D:
```
@article{Matterport3D,
  title={Matterport3D: Learning from RGB-D Data in Indoor Environments},
  author={Chang, Angel and Dai, Angela and Funkhouser, Thomas and Halber, Maciej and Niessner, Matthias and Savva, Manolis and Song, Shuran and Zeng, Andy and Zhang, Yinda},
  journal={International Conference on 3D Vision (3DV)},
  year={2017}
}
```
