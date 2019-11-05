# Deep-Learning

## Understanding-semantic-segmentation-with-unet

* Receptive field or context
* Convolution and pooling operations down sample the image, i.e. convert a high resolution image to a low resolution image
* Max Pooling operation helps to understand “WHAT” is there in the image by increasing the receptive field. However it tends to lose the information of “WHERE” the objects are.
* In semantic segmentation it is not just important to know “WHAT” is present in the image but it is equally important to know “WHERE” it is present. Hence we need a way to up sample the image from low resolution to high resolution which will help us restore the “WHERE” information.
* Transposed Convolution is the most preferred choice to perform up sampling, which basically learns parameters through back propagation to convert a low resolution image to a high resolution image.

## With Pytorch

* https://github.com/udacity/deep-learning-v2-pytorch


## Activation Functions

* Why RELU- https://stats.stackexchange.com/questions/226923/why-do-we-use-relu-in-neural-networks-and-how-do-we-use-it

## Categorical Variables

* https://towardsdatascience.com/deep-embeddings-for-categorical-variables-cat2vec-b05c8ab63ac0

## Batch Normalization in Neural Networks

* https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c
