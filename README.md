# Adversarial example generation for face recognition model

---

## Abstract

In 2014, Goodfellow et al. published a paper entitled Explaining and Harnessing Adversarial Examples, which showed an intriguing property of deep neural networks — it’s possible to purposely perturb an input image such that the neural network misclassified it. This type of perturbation is called an adversarial attack. This paper shows how adversarial attacks can be used to trick face recognition models.
<br />
For this purpose we are going to implement a function which will calculate gradients of the cost function of the face recognition model. In this paper we are going to use a pre-trained keras model VGGFace2 to recognise celebrities’ faces and Multi-task Cascaded Convolutional Neural Network (MTCNN) to detect them on images.
<br />

## Adversarial attack

Adversarial attacks are a method of creating imperceptible changes to an image that can cause seemingly robust image classification techniques to misclassify an image consistently. Back in 2006, the top state-of-the-art machine learning models included Support Vector Machines (SVMs) and Random Forests (RFs) — it’s been shown that both these types of models are susceptible to adversarial attacks.
<br />
With the rise in popularity of deep neural networks starting in 2012, it was hoped that these highly non-linear models would be less susceptible to attacks; however, Goodfellow et al. (among others) dashed these hopes.
<br />

## Face detection. 

Before performing face recognition it is necessary to detect faces. Face detection is a process of automatically locating faces in a photograph and localizing them by drawing a bounding box around them.
<br />
To perform face detection Multi-task Cascaded Convolutional Neural Network (MTCNN) is being used. This is a state-of-the-art deep learning model for face detection, described in the 2016 paper titled “Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks.”. This neural network can be found in the package mtcnn.
<br />
The result of face detection using MTCNN face detector class is a list of bounding boxes, where each bounding box defines a lower-left-corner of the bounding box, as well as the width and height.
<br />
Using these coordinates of the bounding boxes we crop the image.
<br />
<br />

<img align="left" alt="Original Image" width="300px" src=https://github.com/yhetman/adversarial_example/blob/main/sharon_stone1.jpg /> &nbsp;&nbsp;&nbsp;<br /> <br /> <br /> <img align="center" alt="Cropped Image" width="200px" src=https://github.com/yhetman/adversarial_example/blob/main/faceSharon.png /> 
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />

## Face Identification.

To perform face recognition VGGFace2 model will be used. It is needed to install a third-party library for using VGGFace2 models in Keras which is called keras-vggface.
<br />
A VGGFace model can be created using the VGGFace() constructor and specifying the type of model to create via the ‘model‘ argument. 
<br />
The keras-vggface library provides three pre-trained VGGModels, a VGGFace1 model via model=’vgg16′ (the default), and two VGGFace2 models ‘resnet50‘ and ‘senet50‘. The first time that a model is created, the library will download the model weights and save them. The size of the weights for the resnet50 model is about 158 megabytes.
<br />

## Dataset of the pretrained model. 

The model expects input color images of faces with the shape of 244×244 and the output will be a class prediction of 8,631 people. This makes sense given that the pre-trained models were trained on 8,631 identities in the MS-Celeb-1M dataset. Unfortunately, readable labels of categories of the dataset were not found, that is why it was needed to preprocess a binary file with labels retrieving all categories and their indexes.

## Non-adversarial face recognition.

This Keras model can be used directly to predict the probability of a given face belonging to one or more of more than eight thousand known celebrities. Once a prediction is made, the class integers can be mapped to the names of the celebrities, and the top five names with the highest probability can be retrieved. This behavior is provided by the decode_predictions() function in the keras-vggface library. 
<br />
The pixel values must be scaled in the same way that data was prepared when the VGGFace model was fit. Specifically, the pixel values must be centered on each channel using the mean from the training dataset. This can be achieved using the preprocess_input() function provided in the keras-vggface library and specifying the ‘version=2‘ so that the images are scaled using the mean values used to train the VGGFace2 models. 
<br />
As a result the name of the celebrity on the photo will be predicted. This name will be used as a key to a particular index of categories. Here are the results of face recognition using the previous image of Sharon Stone face:

| **Name** | **Confidence** |
|:----:|:----------:|
| Sharon Stone | 99.429% |
| Anita Lipnicka | 0.223% |
| Bobbi Sue Luther | 0.08% |
| Emma Atkins | 0.029% |
| Noelle Reno | 0.017% |


## Adversarial examples generation for face recognition.

The function to generate the perturbation will take in an image to apply perturbations to, and the label that correctly classifies the image.
<br />
The Sparse Categorical Cross-entropy computes the categorical cross-entropy loss between the labels and predictions and will be used as a loss function.  Generation of an adversarial example consists of several stages during iterating over steps:
- add perturbation vector to the base image and preprocess the resulting image;
- run this newly constructed image tensor through the model and calculate the loss with respect to the original category index;
- calculate the gradient of loss with respect to the perturbed vector;
- update the weights, clip the perturbation vector and update the values.

The resulting perturbation vector - the final noise vector - will allow us to construct the adversarial attack used to fool the model. Final adversarial image is constructed by adding the noise vector to the base image. To demonstrate that the function works, we create an adversarial image
<br />
<br />
<img align="left" alt="Original Image" width="200px" src=https://github.com/yhetman/adversarial_example/blob/main/faceSharon.png /> &nbsp;&nbsp;&nbsp;  <img align="center" alt="Adversarial Image" width="200px" src=https://github.com/yhetman/adversarial_example/blob/main/adverImage.png /> 
<br />
<br />
After constructing an adversarial example, it’s directed as an input for the face recognition model. 

| **Name** | **Confidence** |
|:----:|:----------:|
| Bobbi Sue Luther | 99.992156% |
| Maura Rivera | 0.001006% |
| Sam Pinto | 0.000491% |
| Dira Paes | 0.000470% |
| Georgina Verbaan | 0.000452% |
<br />
It can be seen that the predictions generated by VGGFace2 differ, because an adversarial example is incorrectly classified.

## Conclusions.

Adversarial examples have attracted significant attention in machine learning. Researchers argue that there are microscopic features in every dataset that humans are unable to perceive, but neural networks utilize to their fullest extent. Adversarial examples manipulate these features to throw off classifiers, and they can do so without much effect to human viewers.



