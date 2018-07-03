# Traffic Sign Classification

This project involved building a classification model to recognize German traffic signs.

## Dataset Exploration

The training set included 34,799 different images, which covered 43 different signs. All of them were preprocessed to be 32x32 pixels, with 3 color channels (RGB). 4,410 imags were used for validation, with 12,630 reserved for testing after the model was fully tuned.

By printing out random images from the training set, it was clear that a 32x32 resolution is a lot fuzzier than current photographic standards. This is tricky in the sense that the images are a bit hard to discern at times, but with traffic signs being intentionally designed to not have a lot of minute details, it helps the classification algorithm to not have as many nodes in the network.

Also, the frequency of signs in the training set was quite imbalanced - some had as many as 2000 examples in the training set while others had 200.

## Model Architecture

I normalized all pixel values to zero-centered distributions between -1 and 1 to reduce the likelihood of numerical instabilities.

I started with the LeNet architecture then added layers in the middle to add additional complexity and depth.  It was mainly a process of trial-and-error aimed at increasing accuracy in the validation data, which eventually reached **95.6%**. the final architecture was as follows:

#### Convolutional Layer
6 output channels, with a filter width of 5. Valid padding
Input: 32x32x3
Output: 28x28x6

#### Activation Layer
Simple ReLu layer to introduce non-linearities to the Network

#### Pooling layer
2x2 maximum pooling filter that reduces the breadth of the network to allow for computational efficiency
Input: 28x28x6
Output: 14x14x6

#### Convolutional Layer
16 output channels, with a filter width of 5. Valid padding
Input: 14x14x6
Output: 10x10x16

#### Activation Layer
Simple ReLu layer to introduce non-linearities to the Network

#### 1x1 Convolutional layer
24 output channels, as a cheap way of adding depth to the Network
Input: 10x10x16
Output: 10x10x24

#### Activation Layer
Simple ReLu layer to introduce non-linearities to the Network

#### Convolutional Layer
32 output channels, with a filter width of 5. Valid padding
Input: 10x10x24
Output: 6x6x32

#### Activation Layer
Simple ReLu layer to introduce non-linearities to the Network

#### Flattening
Flattening into a 1-D array for use by a fully-connected layers
Input: 6x6x32
Output: 1152 nodes

#### Fully Connected layer
Input: 1152 nodes
Output: 300 nodes

#### Activation
Simply Relu layer

#### Another Fully Connected layer
Introducing fully connected layers piecemeal with Relus introduces nonlinearities and a potentially richer model
Input: 300 nodes
Output: 100 nodes

#### Activation
Simply Relu layer

#### Final fully connected classification layer
Input: 100 nodes
Output: 43 nodes

I used an Tensorflow's Adam Optimizer, with a learning rate of 0.001. A batch size of 256 items, iterated over 40 epochs. All weights were initialized with a mean of zero and a standard deviation of 0.1

## Testing on New Images
I procured 5 images of German Traffic Signs, with the help of a classmate through the Slack channel for this project. The signs I chose were:  **Bicycle crossing, Right of way, Ice/snow warning, 120Kph speed limit, Road work**. Though I did my best to pick images that were centered on the sign, and where the signs were mostly encompassing the entire image, some of them had a bit more empty space than the training set data, which might have posed some problems.

I reshaped the images to 32x32 size using OpenCV's `resize` function, and was a bit struck by how blurry they were, but in fact they are not discernibly blurrier than the other training examples.

After applying the recently-fitted weights and biases to the model and finding the max values of the logits returned by my network, only 1 of 5 traffic signs, the Right-of-Way warning, was correctly predicted (20% accuracy). This is less than I hoped for, and even more surprising was that the Right-of-way sign was the one with the least certainty, based on the softmax probabilities returned by the model. The probability of the right-of-way sign being identified as such was 76%, while for instance, the "bicycle crossing" sign was misidentified as a "turn left ahead" sign with near 100% accuracy. There is obviously more to do to train this model and think this underscores the importance of preventing against overfitting through techniques such as dropout - a lesson I will keep in mind in the future
