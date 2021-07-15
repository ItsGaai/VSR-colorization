#!/usr/bin/env python
# coding: utf-8

# # 1.**安装PaddleGAN**
# 
# PaddleGAN的安装目前支持Clone GitHub和Gitee两种方式.

# In[ ]:


# 当前目录在: /home/aistudio/, 该目录即：左边文件和文件夹所在的目录
# 克隆最新的PaddleGAN仓库到当前目录
# !git clone https://github.com/PaddlePaddle/PaddleGAN.git
# github下载慢，从gitee clone：
get_ipython().system('git clone https://gitee.com/paddlepaddle/PaddleGAN.git')
get_ipython().run_line_magic('cd', 'PaddleGAN/')
get_ipython().system('pip install -v -e .')


# # 2.**PaddleGAN 中使用的模型介绍**
# ## 2.1补帧模型 DAIN
# DAIN的全称是Depth-Aware Video Frame Interpolation，即深度感知视频帧插值，DAIN模型通过探索深度的信息来显式检测遮挡。
# 
# 在这篇研究中，研究人员提出了一种通过探索深度信息来检测遮挡的方法。
# ![](https://ai-studio-static-online.cdn.bcebos.com/f2c916c965e24c259c9a15ac8361bf8a3d667ef587874181825eb10d2b93b0cd)
# 上图是DAIN的体系架构：给定两个时刻的输入帧，先估计光流和深度图，然后使用建议的深度感知流投影层生成中间流。
# 
# 之后，模型基于光流和局部插值内核对输入帧、深度图和上下文特征进行扭曲，合成输出帧。
# 
# 这种模型紧凑、高效且完全可微分。定量和定性的结果表明，DAIN在各种数据集上均优于最新的帧插值方法。
# 
# 简单来说，作者开发了一个深度感知光流投影层来合成中间流，中间流对较远的对象进行采样。此外，学习分层功能以从相邻像素收集上下文信息。
# 
# 【1】论文地址：[https://arxiv.org/pdf/1904.00830.pdf](http://)
# 
# *"Depth-Aware Video Frame Interpolation"*
# 
# 【2】项目地址：[https://github.com/baowenbo/DAIN*](http://)
# 
# ![](./images/dain_network.png)
# 
# ```
# ppgan.apps.DAINPredictor(
#                         output_path='output',
#                         weight_path=None,
#                         time_step=None,
#                         use_gpu=True,
#                         remove_duplicates=False)
# ```
# #### 参数
# 
# - `output_path (str，可选的)`: 输出的文件夹路径，默认值：`output`.
# - `weight_path (None，可选的)`: 载入的权重路径，如果没有设置，则从云端下载默认的权重到本地。默认值：`None`。
# - `time_step (int)`: 补帧的时间系数，如果设置为0.5，则原先为每秒30帧的视频，补帧后变为每秒60帧。
# - `remove_duplicates (bool，可选的)`: 是否删除重复帧，默认值：`False`.
# 
# ## 2.2上色模型 DeOldifyPredictor
# DeOldify采用自注意力机制的生成对抗网络，生成器是一个U-NET结构的网络。在图像的上色方面有着较好的效果。
# 
# DeOldify使用了一种名为NoGAN的新型GAN训练方法，用来解决在使用由一个鉴别器和一个生成器组成的正常对抗性网络架构进行训练时出现的主要问题。典型地，GAN训练同时训练鉴别器和生成器，生成器一开始是完全随机的，随着时间的推移，它会欺骗鉴别器，鉴别器试图辨别出图像是生成的还是真实的。NoGan提供了与通常的GAN训练相同的好处，同时花费更少的时间来训练GAN架构(通常计算时间相当长)。相反，它对生成器进行了预先训练，使其利用常规损失函数，变得更强大、更快、更可靠；大部分的训练时间是用更直接、快速和可靠的传统方法分别预训练生成器和鉴别器。**这里的一个关键观点是，那些更 "传统 "的方法通常可以得到你所需要的大部分结果，而GAN可以用来缩小现实性方面的差距。**
# 
# 其步骤如下：
# 
# *Step1.以传统的方式只用特征损失（feature loss）训练生成器。*
#  
# *Step2.接下来，从中生成图像，并作为一个基本的二进制分类器训练鉴别器区分这些输出和真实图像。*
#  
# *Step3.最后，在GAN设置中一起训练生成器和鉴别器。*
# 
# 【1】暂无论文
# 
# 【2】项目地址：[https://github.com/jantic/DeOldify](http://)
# 
# ![](./images/deoldify_network.png)
# 
# ```
# ppgan.apps.DeOldifyPredictor(output='output', weight_path=None, render_factor=32)
# ```
# #### 参数
# 
# - `output_path (str，可选的)`: 输出的文件夹路径，默认值：`output`.
# - `weight_path (None，可选的)`: 载入的权重路径，如果没有设置，则从云端下载默认的权重到本地。默认值：`None`。
# - `render_factor (int)`: 会将该参数乘以16后作为输入帧的resize的值，如果该值设置为32，
#                          则输入帧会resize到(32 * 16, 32 * 16)的尺寸再输入到网络中。
# 
# ## 2.3上色模型 DeepRemasterPredictor
# DeepRemaster 模型基于时空卷积神经网络和自注意力机制。并且能够根据输入的任意数量的参考帧对图片进行上色。
# ![](./images/remaster_network.png)
# 
# ```
# ppgan.apps.DeepRemasterPredictor(
#                                 output='output',
#                                 weight_path=None,
#                                 colorization=False,
#                                 reference_dir=None,
#                                 mindim=360):
# ```
# #### 参数
# 
# - `output_path (str，可选的)`: 输出的文件夹路径，默认值：`output`.
# - `weight_path (None，可选的)`: 载入的权重路径，如果没有设置，则从云端下载默认的权重到本地。默认值：`None`。
# - `colorization (bool)`: 是否对输入视频上色，如果选项设置为 `True` ，则参考帧的文件夹路径也必须要设置。默认值：`False`。
# - `reference_dir (bool)`: 参考帧的文件夹路径。默认值：`None`。
# - `mindim (bool)`: 输入帧重新resize后的短边的大小。默认值：360。
# 
# ## 2.4超分辨率模型 RealSRPredictor
# RealSR模型通过估计各种模糊内核以及实际噪声分布，为现实世界的图像设计一种新颖的真实图片降采样框架。基于该降采样框架，可以获取与真实世界图像共享同一域的低分辨率图像。并且提出了一个旨在提高感知度的真实世界超分辨率模型。对合成噪声数据和真实世界图像进行的大量实验表明，该模型能够有效降低了噪声并提高了视觉质量。
# 
# > 在CVPR-NTIRE-2020真实图像超分比赛中以明显优势获得双赛道冠军。
# 
# **算法创新设计**,与已有的超分辨率方法相比，RealSR的创新主要体现在三个方面：
# 
# 1. RealSR采用了自主设计的新型图片退化方法，通过分析真实图片中的模糊和噪声，模拟真实图片的退化过程。
# 
# 2. 不需要成对的训练数据，利用无标记的数据即可进行训练。
# 
# 3. 可以处理低分辨率图像中的模糊噪声问题，得到更加清晰干净的高分辨结果。
# 
# 【1】论文地址：[https://arxiv.org/pdf/1904.00523.pdf](http://)  
# 
# *"Toward Real-World Single Image Super-Resolution: A New Benchmark and A New Model"*
# 
# 【2】项目地址：[https://github.com/Tencent/Real-SR](http://)
# 
# ![](./images/realsr_network.png)
# 
# ```
# ppgan.apps.RealSRPredictor(output='output', weight_path=None)
# ```
# #### 参数
# 
# - `output_path (str，可选的)`: 输出的文件夹路径，默认值：`output`.
# - `weight_path (None，可选的)`: 载入的权重路径，如果没有设置，则从云端下载默认的权重到本地。默认值：`None`。
# 
# ## 2.5超分辨率模型 EDVRPredictor
# EDVR模型提出了一个新颖的视频具有增强可变形卷积的还原框架：第一，为了处理大动作而设计的一个金字塔，级联和可变形（PCD）对齐模块，使用可变形卷积以从粗到精的方式在特征级别完成对齐；第二，提出时空注意力机制（TSA）融合模块，在时间和空间上都融合了注意机制，用以增强复原的功能。
# 
# > 在CVPR 2019 Workshop NTIRE 2019 视频恢复比赛中，来自商汤科技、港中文、南洋理工、深圳先进技术研究院的联合研究团队使用EDVR获得了全部四个赛道的所有冠军！
# 
# **算法创新设计**：
# 
# 1. 图像对齐（Alignment）。
# 
# 视频相邻帧存在一定的抖动，必须先对齐才能进一步处理融合。以往这可以使用光流算法处理，但本文中作者发明了一种新的网络模块PCD对齐模块，使用Deformable卷积进行视频的对齐，整个过程可以端到端训练。
# 
# 2. 时空信息融合（Fusion）。
# 
# 挖掘时域（视频前后帧）和空域（同一帧内部）的信息融合。本文中作者发明了一种时空注意力模型进行信息融合。
# 
# EDVR算法架构：
# ![](https://ai-studio-static-online.cdn.bcebos.com/19459eaceee24a628ae4a8378be4b5a44e31edde186f47bf9053806e3348cec1)
# 
# 其中PCD 对齐模块，使用金字塔结构级联的Deformable卷积构建，如图：![](https://ai-studio-static-online.cdn.bcebos.com/b826003b2f6f4c94b561bb757bddf55f31c48f3986c54bc893f886f85ad4b131)
# 
# 时空注意力融合模型TSA如图：![](https://ai-studio-static-online.cdn.bcebos.com/5855ebe382d24cb8a97e2becc50b2d3714a479f3739443828ece14f31f3fcd6b)
# 
# 
# 【1】论文地址：[https://arxiv.org/pdf/1905.02716.pdf](http://)
# 
# *"EDVR: Video Restoration with Enhanced Deformable Convolutional Networks"*
# 
# 【2】项目地址：[https://github.com/xinntao/EDVR](http://)
# 
# ![](./images/edvr_network.png)
# 
# ```
# ppgan.apps.EDVRPredictor(output='output', weight_path=None)
# ```
# #### 参数
# 
# - `output_path (str，可选的)`: 输出的文件夹路径，默认值：`output`.
# - `weight_path (None，可选的)`: 载入的权重路径，如果没有设置，则从云端下载默认的权重到本地。默认值：`None`。

# # **3.使用 PaddleGAN 进行视频修复**

# ## 3.1import-导入可视化需要的包

# In[ ]:


import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import warnings
warnings.filterwarnings("ignore")
import paddle
print("本项目Paddle版本号："+ paddle.__version__)


# ## 3.2定义函数用于展示视频

# In[ ]:


# 定义函数用于展示视频
def display(driving, fps, size=(8, 6)):
    fig = plt.figure(figsize=size)

    ims = []
    for i in range(len(driving)):
        cols = []
        cols.append(driving[i])

        im = plt.imshow(np.concatenate(cols, axis=1), animated=True)
        plt.axis('off')
        ims.append([im])

    video = animation.ArtistAnimation(fig, ims, interval=1000.0/fps, repeat_delay=1000)

    plt.close()
    return video


# ## 3.3用于处理的原始视频展示

# In[ ]:


video_path = '/home/aistudio/Peking_5s.mp4'    # 需要处理的视频的路径
video_frames = imageio.mimread(video_path, memtest=False)
cap = cv2.VideoCapture(video_path)    # 打开视频文件
fps = cap.get(cv2.CAP_PROP_FPS)    # 获得视频的原分辨率
HTML(display(video_frames, fps).to_html5_video())    # Html5 video展示需要处理的原始黑白视频


# ## 3.4调用模型，视频处理过程

# In[23]:


# 使用插帧(DAIN), 上色(DeOldify), 超分(EDVR, RealSR)模型对该视频进行修复
"""
 input参数表示输入的视频路径
 proccess_order 表示使用的模型和顺序（目前支持）
 output表示处理后的视频的存放文件夹
"""
get_ipython().run_line_magic('cd', '/home/aistudio/PaddleGAN/applications/')
get_ipython().system('python tools/video-enhance.py --input /home/aistudio/Peking_5s.mp4                                --process_order  DAIN DeOldify EDVR                                --output output_dir')


# ## 3.5处理后的视频展示

# In[24]:


# 处理好的视频路径如下, 注：如果视频太大耗时久又可能会报错，最好下载到本地来看。
output_video_path = '/home/aistudio/PaddleGAN/applications/output_dir/EDVR/Peking_5s_deoldify_out_edvr_out.mp4'
# 加载过长视频会造成内存溢出，可以在网页上展示处理后的19秒的视频
# output_video_path = '/home/aistudio/moderntimes_output19.mp4'
video_frames = imageio.mimread(output_video_path, memtest=False)
cap = cv2.VideoCapture(output_video_path)    # 打开处理后的视频文件 
fps = cap.get(cv2.CAP_PROP_FPS)    # 获得视频的原分辨率
HTML(display(video_frames, fps).to_html5_video())    # 展示处理后的视频


# ## 3.6音频处理

# In[ ]:


# 完整版Peking_5s.mp4，添加了音频，需要下载到本地播放
# 以上过程没有考虑视频的音频，这部分代码用于音频的添加
video_frames = imageio.mimread(output_video_path2, memtest=False)
cap = cv2.VideoCapture(output_video_path2)    
fps = cap.get(cv2.CAP_PROP_FPS)    # 获得视频的原分辨率
HTML(display(video_frames, fps).to_html5_video())

