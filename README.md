
# SSR-HMR: Skeleton-Aware Sparse Node-Based Real-Time Human Motion Reconstruction

The code for this project is currently in a pre-release state and will be publicly available upon the formal acceptance of the associated paper in the journal. We are committed to adhering to the open-source spirit of the academic community and ensuring the integrity and reproducibility of the code.

**SSR-HMR** is a real-time human motion reconstruction system based on sparse nodes and skeleton perception. The system can efficiently reconstruct human movement from sparse sensor data, which is suitable for virtual reality (VR), augmented reality (AR), motion capture and other fields.

![Pipeline](Pipeline.pdf)

![nine_person](nine_person.png)
Pose reconstruction for users with different body proportions. Each row corresponds to a different action, and each column represents a different user. Different users are visually distinguished, while the blue trajectories represent the real motion capture data from the dataset
![vr-qualitative](vr-qualitative.png)
Qualitative comparison of the jumping jack motion sequence reconstructed by different models. The first row shows the visual results of the motion reconstruction for each model, while the second row compares the reconstruction results with the real human motion (represented in blue).

## Demo Video
![AMASS_dance](AMASS_dance.mp4)
![xsens_nineperson](xsens_nineperson.mp4)

## Highlights
- Novel SSR-HMR method for real-time, accurate full-body motion reconstruction.
- Lightweight spatiotemporal graph module for precise motion from sparse inputs.
- Torso pose refinement module reduces torso and head orientation drift.
- Hierarchical skeletal structure enhances end-effector positioning accuracy.
- Sub-centimeter accuracy (MPJPE 10 mm, MPEEPE 5 mm) at 267 FPS on CPU


## 应用场景

- **虚拟现实（VR）**：实时捕捉用户动作，提供沉浸式体验。
- **增强现实（AR）**：将虚拟角色与现实环境结合，实现自然交互。
- **运动捕捉**：用于电影制作、游戏开发等领域的动作捕捉。
- **医疗康复**：实时监测患者的运动状态，辅助康复训练。

## 快速开始

### 安装依赖

确保你已经安装了以下依赖：

- Python 3.8+
- PyTorch 1.10+
- NumPy
- OpenCV

你可以通过以下命令安装依赖：

```bash
pip install -r requirements.txt
```

### 运行示例

1. 克隆本仓库：

```bash
git clone https://github.com/yourusername/SSR-HMR.git
cd SSR-HMR
```

2. 运行示例代码：

```bash
python demo.py
```

### 数据集

我们提供了一个示例数据集 `sample_data/`，你可以使用它来测试系统。你也可以使用自己的数据集，只需按照相同的格式组织数据。

## 项目结构

```
SSR-HMR/
├── data/                # 数据集
├── models/              # 预训练模型
├── net/               # 模型定义\工具函数
├── train.py              # 示例代码
├── eval.py              # 示例代码
├── requirements.txt     # 依赖列表
├── LICENSE              # 许可证
└── README.md            # 项目介绍
```

## 贡献

我们欢迎任何形式的贡献！如果你有任何改进建议或发现了问题，请提交 Issue 或 Pull Request。


## 引用

如果你在研究中使用了本项目，请引用以下文献：

```bibtex
@article{yourpaper,
  title={SSR-HMR: Skeleton-Aware Sparse Node-Based Real-Time Human Motion Reconstruction},
  author={Linhai Li, Co-authors},
  journal={Journal Name},
  year={2025}
}
```

## 联系方式

如有任何问题，请联系：[linhai.student@foxmail.com](mailto:linhai.student@foxmail.com)

