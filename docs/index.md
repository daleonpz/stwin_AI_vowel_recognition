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

That means that each data sample have the following form. $s^T = [ a_x, a_y, a_z, g_x, g_y, g_z]$, where $a_i$ and $g_i$ are the accelerometer and gyroscope measurements along i-axis. 

```math
\mathbf{F} = 
\begin{bmatrix}
 s^T_1 \\
 \vdots \\ 
 s^T_{200}  
\end{bmatrix} = \begin{bmatrix} \mathbf{a_x} & \mathbf{a_y} & \mathbf{a_z} &\mathbf{g_x} & \mathbf{g_y} & \mathbf{g_z}  \end{bmatrix}
= \begin{bmatrix} \mathbf{f_1} & \mathbf{f_2} & \mathbf{f_3} &\mathbf{f_4} & \mathbf{f_5} & \mathbf{f_6}  \end{bmatrix}
``` 
such that 
```math
\mathbf{a_x} = \begin{bmatrix} a^{(1)}_x \\ \vdots \\  a^{(200)}_x  \end{bmatrix},
\mathbf{g_x} = \begin{bmatrix} g^{(1)}_x \\ \vdots \\  g^{(200)}_x  \end{bmatrix}
```


Before inputting data into the model, I conducted the normalization operation. Since the IMU signals differ in value and range, thus I normalize each the IMU signal $\mathbf{f_i}$ between (1, 0) with function following function.

```math
\mathbf{f_n(i)} = \frac{\mathbf{f_i} -\min\mathbf{f_i} }{\max\mathbf{f_i} - \min\mathbf{f_i}}, i = 1,2,\cdots ,6
```

Thus the normalized feature matrix $\mathbf{F_n}$ is defined as follows: 

```math
\mathbf{F_n} = \begin{bmatrix} \mathbf{f_n(1)} & \mathbf{f_n(2)} & \mathbf{f_n(3)} &\mathbf{f_n(4)} & \mathbf{f_n(5)} & \mathbf{f_n(6)}  \end{bmatrix}
``` 

It was shown in [4](https://arxiv.org/vc/arxiv/papers/1803/1803.09052v1.pdf) that encoding feature vectors or matrices as images could take advantage of the great performance of CNN on images.








