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

## Weight Initialization

* https://datascience.stackexchange.com/questions/30989/what-are-the-cases-where-it-is-fine-to-initialize-all-weights-to-zero
* https://datascience.stackexchange.com/questions/17987/how-should-the-bias-be-initialized-and-regularized/18145#18145
* https://datascience.stackexchange.com/questions/13061/when-to-use-he-or-glorot-normal-initialization-over-uniform-init-and-what-are

## IOT Application
- https://arxiv.org/pdf/1712.04301.pdf

## Embeddings
- https://blogs.rstudio.com/tensorflow/posts/2018-09-26-embeddings-recommender/

## Deep Pi Car
- https://towardsdatascience.com/tagged/deep-pi-car

## GPU on EC2
- https://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/

## Convolutions
- https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728

## EfficientNet
- https://github.com/lukemelas/EfficientNet-PyTorch

## Mix-Data
- https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/

## BERT
- https://towardsdatascience.com/bert-for-dummies-step-by-step-tutorial-fb90890ffe03
- https://www.kaggle.com/c/alaska2-image-steganalysis/discussion/168510

## Machine Learning Deployment
- https://www.udemy.com/course/testing-and-monitoring-machine-learning-model-deployments/

## Interpretating Deep Learning
- https://arxiv.org/pdf/2210.05189.pdf
- https://github.com/SelfExplainML/PiML-Toolbox

## Tabular Deep Learning
### Deep Neural Networks and Tabular Data: A Survey (2021-10)
- Paper - https://arxiv.org/abs/2110.01889
- Code - https://github.com/kathrinse/TabSurvey

#### Revisting Deep Learning For Tabular Data
- https://github.com/Yura52/rtdl
- https://arxiv.org/pdf/2106.11959.pdf
