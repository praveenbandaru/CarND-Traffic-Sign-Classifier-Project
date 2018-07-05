# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/sampleperclass.png "Sample Image per Class"
[image2]: ./examples/noofimagesineachclass.png "No. of Images in each Class"
[image3]: ./examples/equalizecompare.png "Equalization Comparison"
[image4]: ./examples/normalizecompare.png "Normalization Comparison"
[image5]: ./examples/augment.png "Augment example"
[image6]: ./examples/augmentcountchart.png "No. of Images in each Class after augmentation"
[image7]: ./test_images/Children%20crossing.jpg "Traffic Sign 1"
[image8]: ./test_images/No%20entry.jpg "Traffic Sign 2"
[image9]: ./test_images/Right-of-way%20at%20the%20next%20intersection.jpg "Traffic Sign 3"
[image10]: ./test_images/Road%20work.jpg "Traffic Sign 4"
[image11]: ./test_images/Roundabout%20mandatory.jpg "Traffic Sign 5"
[image12]: ./test_images/Speed%20limit%20(50kmh).jpg "Traffic Sign 6"
[image13]: ./test_images/Stop.jpg "Traffic Sign 7"
[image14]: ./test_images/Turn%20right%20ahead.jpg "Traffic Sign 8"
[image15]: ./examples/prediction.png "Prediction"
[image16]: ./examples/top51.png "Top 5 Probabilities for Image1"
[image17]: ./examples/top52.png "Top 5 Probabilities for Image2"
[image18]: ./examples/top53.png "Top 5 Probabilities for Image3"
[image19]: ./examples/top54.png "Top 5 Probabilities for Image4"
[image20]: ./examples/top55.png "Top 5 Probabilities for Image5"
[image21]: ./examples/top56.png "Top 5 Probabilities for Image6"
[image22]: ./examples/top57.png "Top 5 Probabilities for Image7"
[image23]: ./examples/top58.png "Top 5 Probabilities for Image8"
[image24]: ./examples/sampleafterpreprocessing.png "Sample Image per Class after preprocessing"
[image25]: ./examples/visual.png "Visualize Featuremaps"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.

First, I plotted a sample image from each class.
![alt text][image1]

Next, I plotted a bar chart showing how the data is distributed across different classes. It shows the total count of images from each class.
![alt text][image2]

Here we can clearly see that some classes were having less number of images when compared to others.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

My preprocessing pipeline consisted of the following steps:
1. Grayscale Conversion.
2. Histogram Equalization.
3. Normalization.

As a first step, I converted the color image to grayscale using OpenCV library. This was done to reduce the complexity and to speed up the pre-processing.

Next, I applied histogram equalization using OpenCV library. I preferred the CLAHE (Contrast Limited Adaptive Histogram Equalization) method to equalize the images. This helped in preserving the foreground details while increasing the contrast of the bakground.

Here is an example of a traffic sign image before and after grayscaling plus histogram equalization. We can clearly see that CLAHE version performs better in preserving the foreground details.

![alt text][image3]

As a last step, I normalized the image data because data normalization is an important step which ensures that each input parameter (pixel, in this case) has a similar data distribution. This makes convergence faster while training the network.  I have tried different normalization medthods and picked the one which had zero mean and unit variance.

Here is the comparison of the different normalization techniques that were tried:

![alt text][image4]

Here, I plotted a sample image from each class after preprocessing.
![alt text][image24]

I decided to generate additional data because it helped in increasing the training accuracy of the neural network by allowing it to train on an extended dataset when the existing numbers of some classes were relatively low. I augmented the existing data-set with perturbed versions of the existing images. This is done to expose the neural network to a wide variety of variations. This makes it less likely that the neural network recognizes unwanted characteristics in the data-set.

To add more data to the the data set, I used the following techniques:
- Scaling
- Translation
- Rotation

Here is an example where a set of augmented images were produced from a single orignal image:

![alt text][image5]

I had made sure that each class in the dataset contains atleast 1400 images by augmenting the classes which had lesser number of images.

![alt text][image6]

The size of the training set was increased from **34799** to **63880**.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x64							|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 				|
| Flatten	      	| outputs 1600 				|
| Dropout	      	| 0.5 keep probability 				|
| Fully connected		| Input = 1600. Output = 120									|
| RELU					|												|
| Dropout	      	| 0.5 keep probability 				|
| Fully connected		| Input = 120. Output = 84									|
| RELU					|												|
| Dropout	      	| 0.5 keep probability 				|
| Fully connected		| Input = 84. Output = 43									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used AdamOptimizer, a batch size of 128, 50 epochs, a learning rate of 0.001 and dropout rate of 0.5 while training. I have tried changing the optimizer to GradientDescentOptimizer but it perfomed much worse.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.976
* test set accuracy of 0.963

The code for calculating accuracies on my final model is located in the 36th cell of the Ipython notebook.

I have started with the LeNet neural network which I have trained in the LeNet lab as part of the course work and made minor changes to the input and output layers. It is as follows:

1. Input: 32x32x3 for color and 32x32x1 for grayscale.
2. Layer 1: Convolutional. The output shape should be 28x28x6.
3. Activation. Relu.
4. Pooling. The output shape should be 14x14x6.
5. Layer 2: Convolutional. The output shape should be 10x10x16.
6. Activation. Relu.
7. Pooling. The output shape should be 5x5x16.
8. Flatten. 400
9. Layer 3: Fully Connected. This should have 120 outputs.
10. Activation. Relu.
11. Layer 4: Fully Connected. This should have 84 outputs.
12. Activation. Relu.
13. Layer 5: Fully Connected (Logits). This should have 43 outputs.
14. Output: 43.

This gave me an accuracy of 89% out of the box using above architecture. I initally started with RGB images and couldn't get better accuracy so I then moved to grayscale images. Then I applied histogram equalization and normalization and could get the accuracy to 91% to 92%. I've augmented the training data set and it increased the accuracy to 94% to 95% but my test accuracy was not high as expected. So then I introduced dropout layer for fully connected layers and it reduced the validation accuracy a bit. I have played with different learning rates and dropout probabilities. I have even tried to apply L2 regularization but found no improvements. Finally I increased the number of filters in the convolutional layers and it worked as a charm and produced excellent results. I got around 100% training accuracy, 98% validation accuracy and 96.5% testing accuracy. The presence of dropout layers prevented the network from overfitting.

The LeNet architecture was chosen because it is simple and the training dataset consisted of 32x32 images as its input and it performed really well on the MNIST dataset in the LeNet lab. With minimal modifications I was able to get a better testing accuracy using LeNet.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are the German traffic signs that I found on the web:

![alt text][image7] ![alt text][image8]![alt text][image9] ![alt text][image10] ![alt text][image11] ![alt text][image12] ![alt text][image13] ![alt text][image14] 

All of the images got plenty of features and should be easily classified. I'm suspecting little difficulty on the last image due to less difference in contrast between it's foreground and background.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image15] 

The model was able to correctly guess all the traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 96.3%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 40th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a *Children crossing* sign (probability of 99.99%), and the image does contain a *Children crossing*  sign. The top five soft max probabilities were

![alt text][image16] 

For the second image, the model is exactly sure that this is a *No entry* sign (probability of 100%), and the image does contain a *No entry*  sign. The top five soft max probabilities were

![alt text][image17] 

For the third image, the model is exactly sure that this is a *Right-of-way at the next intersection* sign (probability of 100%), and the image does contain a *Right-of-way at the next intersection*  sign. The top five soft max probabilities were

![alt text][image18] 

For the fourth image, the model is relatively sure that this is a *Road work* sign (probability of 99.79%), and the image does contain a *Road work*  sign. The top five soft max probabilities were

![alt text][image19] 

For the fifth image, the model is exactly sure that this is a *Roundabout mandatory* sign (probability of 100%), and the image does contain a *Roundabout mandatory*  sign. The top five soft max probabilities were

![alt text][image20] 

For the sixth image, the model is exactly sure that this is a *Speed limit (50km/h)* sign (probability of 100%), and the image does contain a *Speed limit (50km/h)*  sign. The top five soft max probabilities were

![alt text][image21] 

For the seventh image, the model is exactly sure that this is a *Stop* sign (probability of 100%), and the image does contain a *Stop*  sign. The top five soft max probabilities were

![alt text][image22] 

For the eighth image, the model is relatively sure that this is a *Turn right ahead* sign (probability of 99.99%), and the image does contain a *Turn right ahead*  sign. The top five soft max probabilities were

![alt text][image23] 


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Considering the below image as input:

![alt text][image17]

The visulaization of the output of the activations of the first convolutional layer is as follows:

![alt text][image25] 

We can clearly see that different feature maps look for different features. The ninth featuremap is activated by inside horizontal edges while the tenth featuremap is activated by outside curves.

I couldn't quite understand the output of the activations of the second convolutional layer.
