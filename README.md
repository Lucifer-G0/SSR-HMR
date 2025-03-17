
# SSR-HMR: Skeleton-Aware Sparse Node-Based Real-Time Human Motion Reconstruction

The code for this project is currently in a pre-release state and will be publicly available upon the formal acceptance of the associated paper in the journal. We are committed to adhering to the open-source spirit of the academic community and ensuring the integrity and reproducibility of the code.

**SSR-HMR** 是一个基于稀疏节点和骨架感知的实时人体运动重建系统。该系统能够从稀疏的传感器数据中高效地重建人体的运动，适用于虚拟现实（VR）、增强现实（AR）、运动捕捉等领域。

![Pipeline](Pipeline.pdf)
![nine_person](nine_person.png)
![vr-qualitative](vr-qualitative.png)

## 演示视频
![Demo Video](AMASS_dance.mp4)
![Demo Video](xsens_nineperson.mp4)

## 项目亮点

- **实时性**：系统能够在毫秒级时间内完成运动重建，适用于实时应用。
- **稀疏节点**：仅需少量传感器节点即可实现高精度运动重建，减少硬件成本。
- **骨架感知**：通过骨架约束，确保重建的运动符合人体运动学，避免不自然的姿态。
- **轻量级**：算法经过优化，能够在低功耗设备上运行。

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

