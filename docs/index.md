# Introduction
The objective of this project was to train a neural network your **vowel recognition** based on **Inertial Movement Unit (IMU)** inputs, and deploy the trained network on the devboard [STEVAL-STWINKT1](https://www.st.com/en/evaluation-tools/steval-stwinkt1.html) from STM32.

Workflow overview:

![Project Workflow](/docs/_images/workflow.png)


## Problem definition 

The vowel recognition is a classification problem. I defined the following constrains, context and architecture for this particular problem.

* Number of classes: 5 (A, E, I, O, U)
* Data:
    * 200 samples/class
    * Split 60/30/10

* Input: a 20x20x6 matrix based on IMU data.
* Output: a 5x1 probability vector.
* Architecture: Convolutional Neural Network + Fully Connected + Softmax. 
* Loss function: Cross Entropy, which is typical in multiclass classification. 


##  Data collection and preparation
I collected data from the ISM330DHCX sensor at a sampling rate of 200Hz, and decided to collect data from the accelerometer and gyroscope, because it has been proven that using multiple sensors improves gesture characterization. [1](https://www.mdpi.com/1424-8220/21/17/5713), [2](https://www.mdpi.com/2076-3417/10/18/6507), [3](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6603535/).

That means that each data sample have the following form. $s^T = [ a_x, a_y, a_z, g_x, g_y, g_z]$






