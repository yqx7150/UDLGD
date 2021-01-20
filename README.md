# UDLGD

**Paper**: UDLGD: Unsupervised Deep Learning in Gradient Domain for Multi-contrast MRI Reconstruction

**Authors**: Tao Deng, Yu Guan, Shanshan Wang, Dong Liang, Qiegen liu

Date : 1/2021  
Version : 1.0  
The code and the algorithm are for non-comercial use only.  
Copyright 2020, Department of Electronic Information Engineering, Nanchang University.  

Deep learning, particularly unsupervised learning, has demonstrated tremendous potential to significantly speed up image reconstruction with reduced measurements recently. 
This paper proposes an efficient unsupervised deep learning algorithm in gradient domain (UDLGD) for reconstructing multi-contrast images of the same anatomical cross section from partially sampled K-space data. The present UDLGD consists of two consequent stages. At the prior learning stage, score-based gener-ative model is utilized to train gradient domain prior information from single-contrast image dataset. After the prior is learned, the data-consistency, gradient image and group sparsity are alternatively updated at the iterative reconstruction stage. Experimental results in synthetic and in-vivo MRI data demonstrated that the proposed reconstruction method can achieve lower reconstruction errors and better preserve image struc-tures than competing methods.

## Test
```bash
% UDLGD 
python3.5 separate_SIAT.py --model ncsn --runner siat_multicontrast_compare_TSE_sag_random_R4 --config anneal_lr005_gradient4.yml --doc SIAT1_1dataaug4ch_lr005gradient4 --test --image_folder result_MultiContrast_Sag_4_random_R4
% UDLGD_GS
python3.5 separate_SIAT.py --model ncsn --runner siat_multicontrast_compare_TSE_sag_random_R4 --config anneal_lr005_gradient4.yml --doc SIAT1_1dataaug4ch_lr005gradient4 --test --image_folder result_MultiContrast_Sag_4_random_R4_GS --GS
```

## Graphical representation
 <div align="center"><img src="https://github.com/yqx7150/UDLGD/blob/fig1.png">  </div>
 
Multi-contrast MR images and horizontal and vertical gradients structure information of the corresponding multi-contrast MR images. Here the value range of T1 and T2 images are different, while the corresponding gradient images are near to the same.
 <div align="center"><img src="https://github.com/yqx7150/UDLGD/blob/fig2.png"> </div>

Unsupervised deep learning in gradient domain (UDLGD) for multi-contrast MRI reconstruction framework. UDLGD iterates between two alternative stages: Top: Training stage to learn the gradient distribution via denoising score matching. Bottom: Reconstruction to progressively remove aliasing and recover fine details via Langevin dynamics and group sparsity.   and   stand for extracting the real and imaging part, respectively. The gradient encoding means reconstructing the images from horizontal and vertical gradient estimates, and the gradient decoding means deriving the gradients from spatial images. It is worth noting that UDLGD is learned from single-modal dataset, while used for multi-contrast image reconstruction.

Reconstruction Results by Various Methods at 2D Random Undersampling, acceleration factor = 4.
<div align="center"><img src="https://github.com/yqx7150/UDLGD/blob/TSE_sag_random.png"> </div>

Visual comparison of TSE_sag (256 × 256) reconstructions using same sampling schemes with acceleration  . a: TSE scans at Nyquist rate sampling. b: Random sampling pattern. cd: BCS reconstruction and its absolute error. ef: GSMRI reconstruction and its absolute error. gh: FCSA-MT reconstruction and its absolute error. ij: UDLGD reconstruction and its absolute error. kl: UDLGD-GS reconstruction and its absolute error.

## Table
<div align="center"><img src="https://github.com/yqx7150/UDLGD/blob/table.png"> </div>
Summary of quantitative reconstruction results (PSNR/SNR/RMSE) on the TSE datasets using the FIVE algorithms after retrospective under-sampling with various patterns and acceleration factors. The size of test images is 256×256.

## Checkpoints
We provide pretrained checkpoints. You can download pretrained models from [Baidu Drive](https://pan.baidu.com/s/1PF0uxHE0fuPbAjory2yMOg). 
key number is "011k" 

## Test Data
In file './Sag_Multicontrast', PD, T1 and T2-weighted sagittal brain datasets with size of 256x256 were acquired on a 3T scanner(SIEMENS MAGNETOM Trio), and the datasets were provided by Chinese Academy of Sciences.

## Other Related Projects
  * Multi-Contrast MR Reconstruction with Enhanced Denoising Autoencoder Prior Learning 
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/9098334)  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EDAEPRec)   [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

  * Multi-Channel and Multi-Model-Based Autoencoding Prior for Grayscale Image Restoration  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8782831)  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/MEDAEP)   [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

  * High-dimensional Embedding Network Derived Prior for Compressive Sensing MRI Reconstruction  
 [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S1361841520300815?via%3Dihub)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EDMSPRec)
 
  * Denoising Auto-encoding Priors in Undecimated Wavelet Domain for MR Image Reconstruction  
[<font size=5>**[Paper]**</font>](https://arxiv.org/ftp/arxiv/papers/1909/1909.01108.pdf)  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/WDAEPRec)

  * REDAEP: Robust and Enhanced Denoising Autoencoding Prior for Sparse-View CT Reconstruction  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/9076295)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/REDAEP)

  * Iterative Reconstruction for Low-Dose CT using Deep Gradient Priors of Generative Model  
[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2009.12760)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EASEL)

* Progressive Colorization via Interative Generative Models  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/9258392)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/iGM)

