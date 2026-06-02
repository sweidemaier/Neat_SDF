# NeatSDF
![We compute a global neural network approximation of the signed distance function (SDF) by taking into account a simultaneously
learned neural network phase field representing the medial axis, i.e. the SDF gradient discontinuities.](teaser.jpg)
This repository contains the official code for the paper:

**"Medial Axis Aware Learning of Signed Distance Functions"**  
📄 [Paper](https://doi.org/10.48550/arXiv.2604.16512)

---

## 🔥 Overview

**NeatSDF** is a neural framework that reconstructs Signed Distance Functions (SDFs) from **unoriented point clouds** using a novel higher-order variational loss that explicitly accounts for medial-axis structure (jump set of the SDF gradient).

## 🧠 Abstract

We propose a novel variational method to compute a highly accurate global signed distance function (SDF) to a given point cloud. To this end, the jump set of the gradient of the SDF, which coincides with the medial axis of the surface, is explicitly taken into account through a higher-order variational formulation that enforces linear growth along the gradient direction away from this discontinuity set. The eikonal equation and the zero-level set of the SDF are enforced as constraints. To make this variational problem computationally tractable, a phase field approximation of Ambrosio-Tortorelli type is employed. The associated phase field function implicitly describes the medial axis. The method is implemented for surfaces represented by unoriented point clouds using neural network approximations of both the SDF and the phase field. Experiments demonstrate the method's accuracy both in the near field and globally. Quantitative and qualitative comparisons with other approaches show the advantages of the proposed method. 

---

## 🛠 Installation
This repository provides a Anaconda environment, and requires NVIDIA GPU to run the optimization routine. The code is tested with the following main dependencies: cudatoolkit 11.0, pytorch 2.4.1, torchaudio 2.4.1, torchvision 0.20.0, scikit-learn 1.3.2, trimesh 4.5.3, ubuntu 20.04. 
The whole environment can be set-up using the following commands:

```bash
conda env create -f NeatSDF.yaml
conda activate NeatSDF_env
```
## 🚀 Usage
**Note:** This release supports only 3D point clouds. The 2D version described in the paper will be released soon.

To run the method, execute the following command:
```
python run.py
```
This will start the training process using the example bunny point cloud provided in the pointclouds folder. To use your own data, see the section "Input Data" below.

The config file allows you to adjust various settings, including data paths and hyperparameters. 

The results are saved in `/NeatSDF/logs/SDF<ptcld_name>eps<eps_value><current_date>`.

## 📂 Input Data

NeatSDF accepts point clouds in `.csv` and `.txt` format. Input point coordinates are expected to be normalized to the domain `[-1,1]^d`, where `d` denotes the spatial dimension of the data. Place your point cloud data in the `/NeatSDF/pointclouds` folder to process them with NeatSDF. Multiple examples can be processed in a single run by placing all point cloud files into this folder. The method automatically detects supported file formats and processes the examples sequentially.

---
## ✍️ Citation
If you use this code or ideas from the paper, please cite:
``` bibtex
@article{weidemaier2026medialaxisawarelearning,
      title={Medial Axis Aware Learning of Signed Distance Functions}, 
      author={Samuel Weidemaier and Christoph Norden-Smoch and Martin Rumpf},
      year={2026},
      eprint={2604.16512},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.16512}, 
}
```
Questions or suggestions? Reach out at weidemai@ins.uni-bonn.de


