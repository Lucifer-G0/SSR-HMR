
# SSR-HMR: Skeleton-Aware Sparse Node-Based Real-Time Human Motion Reconstruction

The code for this project is currently in a pre-release state and will be publicly available upon the formal acceptance of the associated paper in the journal. We are committed to adhering to the open-source spirit of the academic community and ensuring the integrity and reproducibility of the code.

**SSR-HMR** is a real-time human motion reconstruction system based on sparse nodes and skeleton perception. The system can efficiently reconstruct human movement from sparse sensor data, which is suitable for virtual reality (VR), augmented reality (AR), motion capture and other fields.

![Pipeline](visualization/Pipeline.jpg)

![nine_person](visualization/nine_person.png)
Pose reconstruction for users with different body proportions. Each row corresponds to a different action, and each column represents a different user. Different users are visually distinguished, while the blue trajectories represent the real motion capture data from the dataset
![vr-qualitative](visualization/vr-qualitative.png)
Qualitative comparison of the jumping jack motion sequence reconstructed by different models. The first row shows the visual results of the motion reconstruction for each model, while the second row compares the reconstruction results with the real human motion (represented in blue).

## Demo Video
![AMASS_dance](visualization/AMASS_dance.mp4)
![xsens_nineperson](visualization/xsens_nineperson.mp4)

## Highlights
- Novel SSR-HMR method for real-time, accurate full-body motion reconstruction.
- Lightweight spatiotemporal graph module for precise motion from sparse inputs.
- Torso pose refinement module reduces torso and head orientation drift.
- Hierarchical skeletal structure enhances end-effector positioning accuracy.
- Sub-centimeter accuracy (MPJPE 10 mm, MPEEPE 5 mm) at 267 FPS on CPU

## Project Structure

```
SSR-HMR/
├── data/                # Dataset
├── models/              # Pre-trained models
├── net/                 # Model definitions\utility functions
├── train.py             # Example code
├── eval.py              # Example code
├── requirements.txt     # Dependency list
├── LICENSE              # License
└── README.md            # Project introduction
```

## Quickstart

### Requirements

- Python 3.8+
- PyTorch 1.10+
- numpy==1.25.1
- upc-pymotion==0.1.8

### Run example

1. clone repository：

```bash
git clone https://github.com/Lucifer-G0/SSR-HMR.git
cd SSR-HMR
```
2. Training
```bash
python train.py [data_path(directory of npz files)]
```

3. Testing
pretrained model is stored in models
```bash
python eval.py [data_path(directory of npz files)] [model_epoch(default:1277)]
```

### Dataset

Download different datasets from the following list as needed.

- [dataset](https://zenodo.org/record/8427980/files/data.zip?download=1) for Xsens from [SparsePoser ](https://github.com/UPC-ViRVIG/SparsePoser)

- datasets from [AMASS](https://amass.is.tue.mpg.de/index.html)
  - DanceDB
  - HUMAN4D
  - SOMA
  - others

  
## Citation

If you use this project in your research, please cite the following paper:

```bibtex
@article{yourpaper,
  title={SSR-HMR: Skeleton-Aware Sparse Node-Based Real-Time Human Motion Reconstruction},
  author={Linhai Li, Co-authors},
  journal={Journal Name},
  year={2025}
}
```

## Contact

For any questions, please contact: [linhai.student@foxmail.com](mailto:linhai.student@foxmail.com)