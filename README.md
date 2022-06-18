# Semantic Segmentation using PyTorch
- State-of-the-Art Semantic Segmentation model

## Semantic Segmentation
    - MIT ADE20K dataset in PyTorch, scene parsing dataset 
        -- http://sceneparsing.csail.mit.edu/
        
    - ADE20K is the largest open source dataset for semantic segmentation and scene parsing, released by MIT Computer Vision team. 
    
    - All pretrained models can be found at: http://sceneparsing.csail.mit.edu/model/pytorch
    
- Syncronized Batch Normalization on PyTorch
    - This module computes the mean and standard-deviation across all devices during training. 
    - We empirically find that a reasonable large batch size is important for segmentation. 

- The implementation is easy to use as:
  - It is pure-python, no C++ extra extension libs.
  - It is completely compatible with PyTorch's implementation. 
  - Specifically, it uses unbiased variance to update the moving average, and use sqrt(max(var, eps)) instead of sqrt(var + eps).
  - It is efficient, only 20% to 30% slower than UnsyncBN.
  
- Dynamic scales of input for training with multiple GPUs
- 
- Supported models:
    - We split our models into encoder and decoder, where encoders are usually modified directly from classification networks, and decoders consist of final convolutions and upsampling. 
    - We have provided some pre-configured models in the config folder.

- Encoder:
    - MobileNetV2dilated
    - ResNet18/ResNet18dilated
    - ResNet50/ResNet50dilated
    - ResNet101/ResNet101dilated
    - HRNetV2 (W48)

- Decoder:
    - C1 (one convolution module)
    - C1_deepsup (C1 + deep supervision trick)
    - PPM (Pyramid Pooling Module, see PSPNet paper for details.)
    - PPM_deepsup (PPM + deep supervision trick)
    - UPerNet (Pyramid Pooling + FPN head, see UperNet for details.)

-  The code is developed under the following configurations:
    - Hardware: >=4 GPUs for training, >=1 GPU for testing (set [--gpus GPUS] accordingly)
    - Software: Ubuntu 16.04.3 LTS, CUDA>=8.0, Python>=3.5, PyTorch>=0.4.0
    - Dependencies: numpy, scipy, opencv, yacs, tqdm
    
