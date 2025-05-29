# Mask R-CNN 目标检测与实例分割演示

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-ee4c2c.svg)](https://pytorch.org/)
[![Detectron2](https://img.shields.io/badge/Detectron2-0.5+-blueviolet.svg)](https://github.com/facebookresearch/detectron2)

本项目旨在演示如何使用 **Detectron2** 框架加载预训练的 **Mask R-CNN** 模型，并对自定义图像进行目标检测和实例分割。我们特别关注模型在第一阶段生成的提议框 (Proposal Box) 与最终精确的预测结果之间的对比，以及模型在实例分割任务上的表现。

## 目录

- [项目简介](#项目简介)
- [实验内容与模型](#实验内容与模型)
  - [模型：Mask R-CNN](#模型mask-r-cnn)
  - [数据集](#数据集)
  - [关于预训练模型](#关于预训练模型)
- [环境要求与依赖](#环境要求与依赖)
- [如何运行演示](#如何运行演示)
  - [运行 Python 脚本](#运行-python-脚本)
  - [查看 Jupyter Notebook 报告](#查看-jupyter-notebook-报告)

## 项目简介

本项目通过 Python 脚本 (`way.py`) 和 Jupyter Notebook (`report.ipynb`) 展示了 Mask R-CNN 的核心功能。主要包括：

-   加载基于 ResNet-50-FPN 的预训练 Mask R-CNN 模型。
-   对指定的图像（例如 `new01.jpg`, `new02.jpg`, `new03.jpg`）进行推理。
-   可视化 Mask R-CNN 第一阶段生成的区域提议框 (经过NMS处理)。
-   可视化最终的目标检测结果 (边界框、类别标签、置信度得分) 和实例分割掩码。

## 实验内容与模型

### 模型：Mask R-CNN

我们使用了 **Mask R-CNN (Mask Region-based Convolutional Neural Network)** 模型。该模型是目标检测领域的强大框架，它扩展了 Faster R-CNN，不仅能够识别图像中的对象并定位它们（通过边界框），还能为每个检测到的对象实例生成像素级的分割掩码。这使得模型能够精确地描绘出对象的轮廓。

本项目中使用的 Mask R-CNN 模型采用 ResNet-50 和 FPN (Feature Pyramid Network) 作为其主干卷积网络。

### 数据集

这里演示代码主要针对一组用户提供的本地图像文件进行操作（在 `way.py` 中默认为 `new01.jpg`, `new02.jpg`, `new03.jpg`）。

### 关于预训练模型

由于时间和计算资源的限制，本项目**直接采用了在 COCO 数据集上预训练好的 Mask R-CNN 模型**。这意味着我们利用了模型在大量多样化图像上学习到的通用特征，而没有在本机上进行额外的训练或微调。这使得我们可以快速部署和测试模型。

## 环境要求与依赖

确保您的环境中安装了以下主要库和工具：

-   **Python** (>= 3.7)
-   **PyTorch** (>= 1.8, 推荐与 Detectron2 兼容的版本)
-   **torchvision** (与 PyTorch 版本对应)
-   **Detectron2** (>= 0.5, 请根据官网指引安装)
-   **OpenCV (cv2)**
-   **NumPy**
-   **Matplotlib**
-   **Jupyter Notebook / Jupyter Lab** (用于查看 `.ipynb` 报告)


*(请注意：Detectron2 的安装强烈建议遵循其官方文档，因为它依赖于特定版本的 PyTorch 和 CUDA (如果使用GPU)。)*

## 如何运行演示

### 运行 Python 脚本

1.  **准备图像**: 将您想要分析的图像 (例如 `new01.jpg`, `new02.jpg`, `new03.jpg`) 放置在与 `way.py` 脚本相同的目录下。
2.  **运行脚本**:
    ```bash
    python way.py
    ```
    脚本会加载模型，处理指定的图像，并在屏幕上显示可视化结果（通过 Matplotlib 弹窗）。每一次可视化后可能需要手动关闭绘图窗口才能继续处理下一张图片或完成程序。

### 查看 Jupyter Notebook 报告

1.  **启动 Jupyter**:
    ```bash
    jupyter lab
    # 或者
    # jupyter notebook
    ```
2.  在 Jupyter 文件浏览器中打开 `report.ipynb` 文件。
3.  按顺序运行 Notebook 中的单元格，即可查看详细的步骤说明、代码执行过程以及嵌入的可视化结果。
