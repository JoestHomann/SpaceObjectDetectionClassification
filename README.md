# SpaceObjectDetectionClassification



Space Object Detection and Classification
Building a Space Object Detection and Classification Model for Space
## Objects
## Authors: Joest Homann, Christopher Kern, Felix Kolb
## Course: Applied Machine Learning
## February 7, 2026

GitHub
The complete source code for this project is can be accessed via the link below:
https://github.com/JoestHomann/SpaceObjectDetectionClassification
i

## Contents
## 1  Introduction1
1.1  Motivation and Project Definition . . . . . . . . . . . . . . . . . . . . . . . . . . .   1
1.2  Dataset  . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   1
## 2  Architecture2
2.1  System Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   2
2.2  Loss Calculation  . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   3
2.3  Hyperparameter Optimization and Results . . . . . . . . . . . . . . . . . . . . . .   5
3  Conclusion and Outlook7
3.1  Strengths of our model . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   7
3.2  Limitations of our model . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   8
3.3  Outlook . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   8
## 4  Appendix10
4.1  Training and Inference Commands  . . . . . . . . . . . . . . . . . . . . . . . . . .  10
4.2  Tensorboard metrics of the last run . . . . . . . . . . . . . . . . . . . . . . . . . .  11
ii

List of Figures
1.1  Object Classes in the Spark 2022 Stream 1 dataset . . . . . . . . . . . . . . . . .   1
2.1  Basic architecture of our model . . . . . . . . . . . . . . . . . . . . . . . . . . . .   2
2.2  Training data from all runs. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   5
2.3  Validation data from all runs. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   6
3.1  8x8 grid with center, class, and bounding box predictions  . . . . . . . . . . . . .   7
3.2  Confusion matrix showing the class accuracy  . . . . . . . . . . . . . . . . . . . .   8
4.1  File-Stucture  . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  10
4.2  Tensorboard training metrics of the last run . . . . . . . . . . . . . . . . . . . . .  11
4.3  Tensorboard validation metrics of the last run . . . . . . . . . . . . . . . . . . . .  11
iii

List of Tables
2.1  Trained hyperparameters and value sets  . . . . . . . . . . . . . . . . . . . . . . .   5
iv

## 1  Introduction
1.1  Motivation and Project Definition
Computer vision has generally been of great interest in the machine learning community. Object
detection models like Ultralytics YOLO can be used for real-time detection of objects. In the
context of space exploration, this could support autonomous or semi-autonomous satellite dock-
ing maneuvers, reducing reliance on continuous ground station communication. To gain a more
in-depth understanding of such models, this report shows the development of an object detection
and classification model based on ResNet18. The model is used to predict bounding boxes and
classifications on space imagery of different types of satellites and space debris.
## 1.2  Dataset
The dataset used is the SPARK 2022 Stream 1 dataset, created by the University of Luxembourg
[1]. It consists of 110,000 high-resolution images of different types of satellites and space debris.
There are 10 different types of satellites and one type of space debris, as shown in image 1.1.
The dataset is split up into training, evaluation, and testing data with a ratio of 60/20/20. The
dataset is a synthetic dataset, consisting of rendered images using the Unity 3D engine [1]. This
means that even when a model performs well on this dataset, functionality is not guaranteed on
real-world data.
Figure 1.1: Object Classes in the Spark 2022 Stream 1 dataset
## 1

## 2  Architecture
To enable structured work on the model, it was split up into multiple Python files. An overview
can be seen in fig. 4.1. All files are interconnected, the train_main.py and infer_main.py can
be accessed via terminal command lines. Example command lines can be seen in section 4.1.
The model takes in training and validation data in the YOLO format.
## 2.1  System Overview
In order to detect objects, the features of the image must be extracted first. To achieve this,
we used ResNet18 as the model’s backbone.  We used a version of ResNet18 that is already
pretrained for feature extraction to reduce the training time of our model.  By training the
model, the backbone gets retrained to our data, so using a pretrained network simply speeds
up convergence of the training. Before feeding an image into ResNet18, it is resized to 320x320
pixels to reduce computational time, as can be seen in figure 2.1. ResNet18 transforms the image
into a 10x10x512 feature space, which our custom heads use to extract information from.
Figure 2.1: Basic architecture of our model
All heads are 1x1 convolutional kernels.  The center head predicts the probability, that the
center of the object lies within this cell for all 10x10 feature grid cells. The cell with the highest
confidence score is identified for the following heads to work on. Both the class head and the
box head use only this grid cell to predict the class and the box dimensions. In a way, the center
head is an attention module that tells the class and box heads where to look in the feature space,
while the class and box heads do the actual predictions that the model is supposed to make. The
## 2

class head predicts a probability for all 11 classes given in the dataset, and the box head predicts
the dimensions of the bounding box by predicting center coordinates as well as the height and
width of each box.
## 2.2  Loss Calculation
From the architecture it can be deduced that there are three critical parts to our training and
therefore to our loss calculation. The class loss, the center loss, and the box loss. Firstly, there
is the classification of the detected object, which is represented by an 11-entry vector filled with
the model’s logit output for each class. To compute a comparable loss, this data needs to be
normalized to get consistent loss values. Therefore, the softmax function is applied to all entries.
Softmax normalizes data to an interval between [0, 1]. As can be seen in the equation
softmax(z)
c
## =
e
z
c
## P
## C
k=1
e
z
k
## ,(2.1)
the softmax function is an exponential normalization function that assigns the highest probability
to the highest logit value but scales exponentially. After normalization, a negative log-likelihood
is applied to the entry that corresponds to the true class.
## L
## CE
(z,c) =− log
##  
softmax(z)
c
## 
## (2.2)
The second part of the loss evaluation is the calculation of the center loss. To compute this loss,
ground-truth grid cells from the dataset. Then the model can check whether the computed grid
cell matches the true center. This is computed via the Binary Cross Entropy, which you can see
in the following:
## L
## BCE
## (p
i,j
## ,y
i,j
## ) =− (y
i,j
log(p
i,j
) + (1− y
i,j
) log(1− p
i,j
## )).(2.3)
Here, the p
i,j
represents the calculated probability that the center point lies in the cell
i,i
. This y
i,j
is the ground-truth value, meaning the exact cell has the value one, and all the background cells
should have the value zero. Therefore, the loss is minimized when the probability for background
cells is high and the probability for the true center cell is high as well. If the true center point is
close to the border of the next cell, the model starts struggling with locating the correct cell. To
counter this problem, we added a Gaussian Heatmap to introduce spatial tolerance in the target.
This changes the true center cell to a Gaussian-distributed truth value y. So the neighboring
cells are not automatically discounted as possible center cells. To tune this, one can change the
shape and range of the following distribution:
H(x,y) = exp
## 
## −
(x− x
c
## )
## 2
+ (y− y
c
## )
## 2
## 2σ
## 2
## 
## .(2.4)
## 3

Even though it is not the standard loss calculation for this type of parameter, it grew from our
first working build without the Gaussian distribution. In the last step, the center loss is weighted
to balance the many negative cells and the positive cells. As well as a normalization over all
the grid cells. As for the third part of our loss calculation, the box loss is assessed. In this loss
factor, the loss is easily calculated through the difference in the parameters that are necessary
for the building of the box. Essentially, one can calculate the absolute difference between the
center coordinates, the box height, and the width. And apply a linear regression.
SmoothL1(x) =
## 
## 
## 
## 0.5x
## 2
## ,  |x| < 1,
|x|− 0.5,  else.
## (2.5)
In our case, the regression smoothes for values smaller than one to prevent faulty gradients close
to zero. After that, the entries in this vector get summed up, and the mean is calculated for the
final definition of the box loss.
The total loss is then calculated by summing all loss components and weighing them according
to our training goals. With the box weight as a trainable hyperparameter.
## L
total
= w
c
## L
center
+ w
b
## L
box
+ w
cl
## L
class
## (2.6)
where w
c
= 3,  w
b
= 5,  w
cl
## = 1
## 4

2.3  Hyperparameter Optimization and Results
To increase the model’s accuracy, we introduced a hyperparameter optimization. For our purpose,
the best way to implement an HPO was using a grid search over the model’s hyperparameters.
Said parameters are
learning rateweight decayσk-factorbox weight
## [1e−4, 3e−4][1e−4, 1e−5][0.3, 0.5,0.7][6, 8, 10][3, 5, 8]
Table 2.1: Trained hyperparameters and value sets
By using a grid search, each hyperparameter is assigned a set of values that are combined in
all possible ways. With our chosen hyperparameter sets, the grid search becomes very lengthy.
When using the grid in table 2.1, it would take 2· 2· 3· 3· 3 = 108 trials. Because the training
is very time-consuming, we opted for a staged grid search, which decreases the total amount
of trials we need to train to just 2· 2 + 3· 3 + 3· 3 = 22. The drawback to this is that some
parameter combinations will not be tested, and we have to assume that the best combination of
each stage is the best overall. Note that the k-factor was trained in two training sets. While most
hyperparameters are self-explanatory, σ refer to the width of the Gaussian heatmap and the k-
factor, which weighs the negative and positive cells in the center loss calculation. The first stage
tested the learning rate and weight decay. The second stage tested sigma and the k-factor. The
third stage tested k-factor and box weight. In the first stage, the middle value of the untrained
hyperparameters was used. Later stages used the best combination of hyperparameters from
the previous stages.  The best combination of hyperparameters is highlighted in Table 2.1,
although it must be noted that the decisive value was the class accuracy. Unfortunately, our
Tensorboard data has been corrupted. Therefore, we can only show the training and validation
data unformatted.
Figure 2.2: Training data from all runs.
## 5

In figure 2.2 we can still see two distinct trends. First, the center loss shows three groupings of
curves. This drift corresponds to the increase of σ, and is rather a matter of not normalizing
the loss computation regarding the radius of the Gaussian heatmap. Second, the model very
confidently predicts center, class, and box with high accuracy. While this data looks promising,
the validation data in figure 2.3 shows signs of overfitting and much less confidence in center
accuracy.
Figure 2.3: Validation data from all runs.
In the validation data, there are two concerning characteristics. We can see signs of overfitting
in the fact that the center accuracy is rising rapidly and then plateauing at around 0.78. This
then increases the center loss. The second characteristic is the roughness of the class loss and
therefore the class accuracy. Despite the good class accuracy, some batches of validation data
were not predicted correctly.
## 6

3  Conclusion and Outlook
In the following chapter, we want to summarize the strengths and limitations of our model and,
lastly, outline how its performance could be improved.
Figure 3.1: 8x8 grid with center, class, and bounding box predictions
3.1  Strengths of our model
As seen in the training metrics (see figure 4.2), our model shows a stable training behavior and
converges early on. This indicates that our overall implementation works so that the loss terms
are calculated as intended and therefore the model learns steadily with each epoch without di-
vergence.
Also, the model achieves a very high-class accuracy with an average of roughly 96 %, which
shows that the classification task works well.
Given correct localization of the center, our model can reliably predict the bounding box. This
can be seen in the validation metrics (see figure 4.3), as the bounding box loss converges steadily
to almost zero. However, the boxes placement is highly dependent on the center’s accuracy, so
where the center was predicted. If this prediction is not accurate, the placement of the bounding
box will also be shifted.
## 7

Figure 3.2: Confusion matrix showing the class accuracy
3.2  Limitations of our model
The main limitation of our model is the center accuracy. Here we only achieve an accuracy of
roughly 77 % in the validation dataset. This is mainly caused by our model architecture. We
use a 10x10 grid, where the grid cell with the highest probability will be chosen as the center
cell. Based on this predicted center cell, the model predicts the bounding box and the class. The
class prediction is seemingly not too affected by a wrong center cell prediction, but the bounding
box is, which can be seen in (see figure 3.1).
We are also limited by a slight overfitting, which can be seen in the gap when comparing the
center accuracy in the training dataset (see figure 4.2) and the validation dataset (see figure
4.3), which is likely due to not well optimized hyperparameters (e.g., learning rate, weight decay,
smoothness of the Gaussian heatmap).
## 3.3  Outlook
As stated in 3.1 our model achieves a high classification accuracy with stable training and good
box regression, but it also needs improvement.  Mainly, the low center accuracy needs to be
increased, and the overfitting should be reduced further (see 3.2).
## 8

To improve on these issues, we suggest the following improvements.
## 1
These improvements are:
- Different pretrained backbone
For our model, we used the pretrained ResNet18-ImageNet model. As this model is a less
complex model compared to other (newer) models, using a different backbone with higher
complexity could improve the generalization of the model and therefore improve the center
prediction accuracy. For this, it could be viable to test a different model like ResNet34,
ResNet50, ConvNeXt-T, or EfficientNet as the backbone.
- Dropouts or fine-tuning in backbone
As the gap in center accuracy between the training dataset and the validation dataset
shows, overfitting is a limitation of our model. Firstly, an idea would be to use dropout in
the network, so intentionally induce noise during training by randomly turning off parts of
the neural network during training.
Secondly, a similar approach would be to fine-tune the learning of the backbone, so start
by training only the heads of the model and then progressively unfreeze certain backbone
layers.
- Optimize the center-based bounding box prediction
The center head is our main limitation, since inference depends on selecting the correct
grid cell. More robust localization could be achieved by switching from our current dis-
crete argmax cell selection approach to a continuous center estimate, e.g., sub-cell offset
regression or soft-argmax on our heatmap.
- Other bounding box prediction method
Instead of trying to optimize the center-based bounding box prediction, it might be more
viable to introduce a different method. For this, one could, e.g., use a corner-based approach
to predict the bounding box.
- More thorough Hyperparameter Optimization
As of now, we could not do a thorough hyperparameter optimization. With enough time
and/or more powerful hardware, more combinations of hyperparameters can be tested.
Potentially this could lead to finding a set of hyperparameters that improve the model’s
performance.
## 1
It should be noted that this list is not complete - there might be more possibilities to improve our model’s
performance.
## 9

## 4  Appendix
Figure 4.1: File-Stucture
4.1  Training and Inference Commands
The following console commands can be used to start training or run inference.
Example command for inference:
python infer_main.py --checkpointPath {path_to_checkpoint}
--imagePath {path_to_image_folder} --device cuda --imgsz 320 --stride_S 32
Example command for training:
python train_main.py --datasetRoot {path_to_training_data} --device cuda
## --lr 3e-4 --weight_decay 2e-2 --batch_size 64 --epochs 15 --amp
## 10

4.2  Tensorboard metrics of the last run
Figure 4.2: Tensorboard training metrics of the last run
Figure 4.3: Tensorboard validation metrics of the last run
## 11

## Bibliography
[1] University  of  Luxembourg.    Space  object  image  dataset.    https://cvi2.uni.lu/
spark-2022-dataset/, 2022. Accessed: 2026-01-04.
## 12
