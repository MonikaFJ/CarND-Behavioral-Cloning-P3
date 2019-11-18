# **Behavioral Cloning**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[simple]: ./examples/image_center.jpg "Simple track"
[left]: ./examples/image_left.jpg "Driving left"
[fliped]: ./examples/fliped.png "Flipped image"
[dataset_initial]: ./examples/dataset_visualization.png "Initial dataset"
[dataset_augmented]: ./examples/dataset_visualization_augmented.png "Augmented dataset visualization"
[advanced]: ./examples/advanced.jpg "Advanced track"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My model is based on network presented in [Nvidia End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/). I introduced, however, few changes to prevent overfitting and handle the difference in the input size and quality of the training data.

Nvidia has bigger dataset, with more complicated images (real live vs. simulation) and higher image resolution.

Finding the best network architecture was an iterative work. I was changing the size and removing layers and observing the behavior during training and during run in the simulator.

It has fast become quite obvious, that changes in the network are not enough to obtain good results when training only on  data that I was initially using. That's why I've added data augmentation and new data, before I decided on final network architecture.

#### 2. Final Model Architecture

Final model consist of 4 convolution layers with different filter, kernel and stride sizes (see description below) each flowed by 2x2 max pooling layers and 3 fully connected layers with relu activation function (that helps captured nonlinearity). Two first fully connected layers has an additional dropout layer that was randomly skipping 20% of neurons during training phase to avoid overfitting.

          _________________________________________________________________
        Layer (type)                 Output Shape              Param #
        =================================================================
        lambda_1 (Lambda)            (None, 160, 320, 3)       0
        _________________________________________________________________
        cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0
        _________________________________________________________________
        conv2d_1 (Conv2D)            (None, 86, 318, 3)        138
        _________________________________________________________________
        max_pooling2d_1 (MaxPooling2 (None, 43, 159, 3)        0
        _________________________________________________________________
        conv2d_2 (Conv2D)            (None, 39, 157, 24)       1104
        _________________________________________________________________
        max_pooling2d_2 (MaxPooling2 (None, 19, 78, 24)        0
        _________________________________________________________________
        conv2d_3 (Conv2D)            (None, 15, 76, 36)        12996
        _________________________________________________________________
        max_pooling2d_3 (MaxPooling2 (None, 7, 38, 36)         0
        _________________________________________________________________
        flatten_1 (Flatten)          (None, 9576)              0
        _________________________________________________________________
        dense_1 (Dense)              (None, 100)               957700
        _________________________________________________________________
        dropout_1 (Dropout)          (None, 100)               0
        _________________________________________________________________
        dense_2 (Dense)              (None, 50)                5050
        _________________________________________________________________
        dropout_2 (Dropout)          (None, 50)                0
        _________________________________________________________________
        dense_3 (Dense)              (None, 10)                510
        _________________________________________________________________
        dense_4 (Dense)              (None, 1)                 11
        =================================================================
        Total params: 977,509
        Trainable params: 977,509
        Non-trainable params: 0


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded following situations:

  * Driving forward line in the middle
  * Driving backward line in the middle
  * Driving forward recovering with driving on the edges
  * Driving backward recovering with driving on the edges

Then I repeated this process on track two in order to get more data points.


From right to left: simple track, driving in the middle of the lane; simple track driving close to the left side of the track; advanced track.

![Simple track, car driving forward][simple]
![Simple track, car driving on the left][left]
![Advanced track][advanced]


After I loaded the data I realized that it was highly biased with driving straight (s. That's why from each frame when the car was not driving straight I've created 5 new frames by adding left and right camera images (angle measurement was created by subtracting or adding offset 0.25 to steering angle) and flipping images for left, center and right frames (angle measurement was created by multiplying the steering angle for the frame by -1)

![Initial dataset from simple track][dataset_initial]
Testset represenation before augmentation. On the horizontal axis number of samples and on the vertical axis the steering angle

![Augmented dataset from simple track][dataset_augmented]
Testset represenation after augmentation. On the horizontal axis number of samples and on the vertical axis the steering angle

![Flipped image][fliped]
Example of flipped image.

After the collection process, I had 6570 train samples and 1643 validation samples for simple track and 13756 train samples and 3439 validation samples for advanced track. Then, in Keras Lambda layer, I've normalized this data. To reduce the size of the network I cut the top 50 px and the bottom 30 px of each image using Cropping2D Keras layer. This parts of the image anyway doesn't contain any valuable information.

For training the model that is running on simplified track I could not used all the collected data. After training the model with all runs on the advanced track I've noticed that the car is trying to follow one of the edge lines which results in collisions with road elements. It was probably caused by the different road design in advanced data: the road there has a line in the middle, so the model trained to follow it.

![Simple track design][simple]
![Advanced track design][advanced]

The model was trained and validated on different randomly shuffled data sets to ensure that the model was not overfitting (split 80/20). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

It was learning fast and after 8-12 epochs it starts overfitting (training accuracy was decreasing when evaluation accuracy was increasing). That's why the 'EarlyStopping' parameter was added to model fitting. It was terminating the training as soon as accuracy on training set stop decreasing.

The model trained only on data from simple track was able to follow the simple track correctly (see video.mp4), but was failing quickly on the advanced track.

The model trained on data from simple and advanced tracks was following the edge of the road in the simple track, which results in collision, however, it was able to navigate correctly through most of the advanced track. It is possible that with the better training data it would finish the track, the best data that I could record was quite far away from the most optimal line.
