# Deep-Learning

**What Deep Learning?**

Ans:- Method to learn the representations of data using successive layers.

**How Deep Learning works?**

Ans :- It has many network layers learning the representation of data. Each layer is different than original image and increasingly informative about the final results. We can think of it as passing through many filters and finally getting purified results.

**What is the meaning of Learning?**

Ans:- We have weight assigned to each network neuron. We optimize over the loss function to estimate the weights and that is called learning. We feed data many times leading to changing of weights therefore leading to Learning of task.Below is the image:-

![alt text](https://github.com/avinashbarnwal/Deep-Learning/blob/master/image/feedbackloop.png)

**How do we initiate the weights of network?**

Ans :- Initially, we assign random values to weigths of network, leading to random transformations of original data and also this results to high loss score.

**How you would show the toy problem?**

Ans:- We consider handwriting recognition as toy problem.Open this [link](https://github.com/avinashbarnwal/Deep-Learning/blob/master/code/toy_problem.ipynb)

## Understanding-semantic-segmentation-with-unet

* Receptive field or context
* Convolution and pooling operations down sample the image, i.e. convert a high resolution image to a low resolution image
* Max Pooling operation helps to understand “WHAT” is there in the image by increasing the receptive field. However it tends to lose the information of “WHERE” the objects are.
* In semantic segmentation it is not just important to know “WHAT” is present in the image but it is equally important to know “WHERE” it is present. Hence we need a way to up sample the image from low resolution to high resolution which will help us restore the “WHERE” information.
* Transposed Convolution is the most preferred choice to perform up sampling, which basically learns parameters through back propagation to convert a low resolution image to a high resolution image.
