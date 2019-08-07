# SSIM_Loss-using-Tensroflow
This is my implementation of SSIM_Loss using tensorflow. I modify the existed code [here](https://github.com/iteapoy/SSIM-Loss) to handle image batches. I also refer to the [pytorch version](https://github.com/Po-Hsun-Su/pytorch-ssim) and other meterials.

## Dependencies
Tensorflow (tested with 1.2.1)
Numpy
Scikit-image (only for test)

## How to use
Copy "SSIM_loss_class.py" in your project, then

from SSIM_loss_class.py import SSIM
loss_ssim = SSIM(k1=.., k2=.., window_size=..)
loss = loss_ssim.ssim_loss(image1, image2)

Please refer to the "SSIM_loss_class.py" for more details. You can just run it with python

## Attention
1. I modify the existed tensorflow version to handl image batches case. Especially, I use tf.nn.depthwise_conv2d to perform the same conv operations as in pytorch version, which use grouped conv2d. 
2. I introduce the skimage.measure.compare_ssim to valid my code. I find that different padding strategies will influence the final results. In this version, I use "VALID", and the result is 0.6357603 compared with 0.634312 obtained from skimage. If using "SAME", the result is about 0.6567436. However, I only valid the code on provided two images. Anyway, I think it can be used as a stable loss function.

## To Do
1. Extend the code to MM-SSIM
2. Please let me known if there exists any problems.
