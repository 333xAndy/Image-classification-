# Image-classification

Training an image classifier using tensorflow.

<details>
  <summary>Table of contents</summary>
  
  1. [About the project](#-About-the-project)
  2. [Getting started](#-Getting-started)
  3. [Usage](#-Usage)
  4. [Acknowledgemnets](#-Acknowledgemnets)
  
</details> 


# About the project
The purpose of this recreate an image classification using the keras sequential model.
This is accomplished by loading data using keras preprocessing. 
Basic machine learning workflow goes as follows:
1. Examine/Understand data
2. Build input pipeline (Sequential)
3. Build model
4. Train
5. Test

# Getting started

To use tensorflow, follow the installation guide on the official website (Link in acknowldegments)
Then import the needed libraries (Im using python) 

```python
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
```

# Usage

This model is useful for single - label applications. That is, prediciting what an image is most likely to represent. The model is trained to predict from thousands of images. This model works for cats and dogs. A different model should be used for different objects.

Model accuracy at the time of this README is 60%.

# Acknowledgemnets
[Tensorflow install](https://www.tensorflow.org/install)  
[Tensorflow image classification](https://www.tensorflow.org/tutorials/images/classification)  
[Keras Developer Guides](https://keras.io/guides/)  
[Cats and Dogs imageset](https://www.kaggle.com/datasets)  
