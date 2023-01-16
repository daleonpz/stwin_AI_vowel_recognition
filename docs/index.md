# Introduction
The objective of this project was to train a neural network your **vowel recognition** based on **Inertial Movement Unit (IMU)** inputs and deploy the trained network on the dev board [STEVAL-STWINKT1](https://www.st.com/en/evaluation-tools/steval-stwinkt1.html) from STM32.

Workflow overview:

![Project Workflow](/docs/_images/workflow.png)

## Used tools
- Pytorch 1.10.2+cu102
- ONNIX-tensorflow 1.10.0
- STM32 CUBE-AI 7.1.0
- Sensor ISM330DHCX
- 18.04.1-Ubuntu

# Problem definition 

The vowel recognition is a classification problem. I defined the following constrains, context and architecture for this particular problem.

* Number of classes: 5 (A, E, I, O, U)
* Data:
    * 200 samples/class
    * Split 60/30/10

* Input: a 20x20x6 matrix based on IMU data.
* Output: a 5x1 probability vector.
* Architecture: Convolutional Neural Network + Fully Connected + Softmax. 
* Loss function: Cross Entropy, which is typical in multiclass classification. 


#  Data collection and preparation
I collected **200 samples for each class** and each sample corresponds to **2sec** of data from the **ISM330DHCX sensor** at a **sampling rate of 200Hz**. I decided to collect data from the accelerometer and gyroscope, because it has been proven that using multiple sensors improves gesture characterization. [1](https://www.mdpi.com/1424-8220/21/17/5713), [2](https://www.mdpi.com/2076-3417/10/18/6507), [3](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6603535/).

![Class pattern](/docs/_images/vowels_pattern.png)

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


Before inputting data into the model, I conducted the normalization operation. Since the IMU signals differ in value and range, thus I normalize each signal $\mathbf{f_i}$ between (1, 0) with the following function.

```math
\mathbf{f_n(i)} = \frac{\mathbf{f_i} -\min\mathbf{f_i} }{\max\mathbf{f_i} - \min\mathbf{f_i}}, i = 1,2,\cdots ,6
```

Thus the normalized feature matrix $\mathbf{F_n}$ is defined as follows: 

```math
\mathbf{F_n} = \begin{bmatrix} \mathbf{f_n(1)} & \mathbf{f_n(2)} & \mathbf{f_n(3)} &\mathbf{f_n(4)} & \mathbf{f_n(5)} & \mathbf{f_n(6)}  \end{bmatrix}
``` 

[4](https://arxiv.org/vc/arxiv/papers/1803/1803.09052v1.pdf) showed that encoding feature vectors or matrices as images could take advantage of the performance of CNN on images. Thus the shape of input image $\mathbf{I_f}$ is 20x20x6. 

![Input image](/docs/_images/input.png)


For the data collection I wrote some python scripts and C code for the microcontroller. All the relevant code for this task can be found [here](/datacollector).  The main script is [collect.sh](/datacollector/pythonscripts/collect.sh), where I defined the labels for each class. 


# Neural Network Architecture

I tried two architectures based on Convolution neural and fully connected networks. The papers I used as a reference are the following:
- Jianjie, Lu & Raymond, Tong. (2018). Encoding Accelerometer Signals as Images for Activity Recognition Using Residual Neural Network. 
- Jiang, Yujian & Song, Lin & Zhang, Junming & Song, Yang & Yan, Ming. (2022). Multi-Category Gesture Recognition Modeling Based on sEMG and IMU Signals. Sensors. 22. 5855. 10.3390/s22155855. 

## First architecture
The first architecture was the following. 
![first architecture](/docs/_images/CNN1.png)

* Number of parameters: 
   * Conv2D + Batch + ReLU: $24*6*3*3 + 24 + 24 + 24 + 0 = 1368$
   * Conv2D + Batch + ReLU: $48*24*3*3 + 48 + 48 + 48 + 0 = 10512$
   * FC + ReLU:  $512*4800 + 512 + 0 = 2458112$
   * FC + ReLU:  $512*32 + 32 + 0 = 16416$
   * FC:    $5*32 + 5 = 165$
   * Total number of parameters: 2486573 

* Number of elements per parameter: 4 (float)
* Size of the model:  $2486573 * 4 = 9713.754 KB = 9.48 MB$ 

As you can see, the model size is too big 9.48 MB for the STWINK devboard, which has ARM Cortex-M4 MCU with 2048 kbytes of flash. But I still trained the model to see if I was on a right direction.

Here are the results:

| Model        | n_samples | num_epochs | learning_rate | criterion    | optimizer | batch_size | train_acc | train_loss | val_acc | val_loss | test_acc | test_loss | size(float32) |
| ------------ | --------- | ---------- | ------------- | ------------ | --------- | ---------- | --------- | ---------- | ------- | -------- | -------- | --------- | ------------- |
| cnn (512,32) | 200       | 10         | 0.00001       | CrossEntropy | Adam      | 16         | 91.67     | 0.514      | 83.33   | 0.544    | 82.5     | 0.5727    | 9MB           |
| cnn (512,32) | 200       | 50         | 0.00001       | CrossEntropy | Adam      | 16         | 100       | 0.328      | 95      | 0.376    | 97.5     | 0.3458    | 9MB           |
| cnn (512,32) | 200       | 20         | 0.00001       | CrossEntropy | Adam      | 16         | 96.67     | 0.402      | 86.67   | 0.469    | 92.5     | 0.4352    | 9MB           |

As you can see, the accuracy in all datasets (train, validation and test) is above 90% which is a good indication. Although I plotted the curves "Accuracy vs epoch" and "Loss vs epoch", as well as the "confusion matrix" that validate performance of this model, I do not shown them because this model was not deployed.

## Model 
The model I ended up deploying is the following
![Architecture](/docs/_images/CNN2.png)

* Number of parameters: 
   * Conv2D + Batch + ReLU: $12*6*3*3 + 12 + 12 + 12 + 0  = 684$
   * Conv2D + Batch + ReLU: $24*12*3*3 + 24 + 24 + 24 + 0 = 2664$
   * FC + ReLU:  $8*600 + 8 + 0 = 4808$
   * FC:    $5*8 + 5 = 45$
   * Total number of parameters: 32804 

* Number of elements per parameter: 4 (float)
* Size of the model:  $32804 * 4 = 32.332 KB$ 

This model is small enough to be deployed on the microcontroller, it uses only 1.58% of the available flash memory.

The results of training are the following:

![Results](/docs/_images/results.png)

The overall performance is above 90%. 

For the training I wrote python scripts and C files for the microcontroller. All the relevant code for this task can be found [here](/trainer).  Run `python3 train_model.py  --dataset_path ../data --num_epochs 200` to train the model. 

# Deployment on the microcontroller
The CUBE-AI tool from STMicroelectronics doesn't support pytorch models, so I had two options. Either train the whole model again in tensorflow and then export it to tensorflow lite, or convert the pytorch model to ONNX (Open Neural Network Exchange). I chose the latter.

To export from pytorch to ONNX is straighforward:

```python
from models.cnn_2    import CNN

# Load pytorch model
loadedmodel     = CNN(fc_num_output=5, fc_hidden_size=[8]).to(DEVICE) # my model
loadedmodel.load_state_dict(torch.load('results/model.pth')
loadedmodel.eval()

# Fuse some modules. it may save on memory access, make the model run faster, and improve its accuracy.
# https://pytorch.org/tutorials/recipes/fuse.html
torch.quantization.fuse_modules(loadedmodel,
                                [['conv1', 'bn1','relu1'], 
                                 ['conv2', 'bn2','relu2']],
                                inplace=True)

# Convert to ONNX. 
# Explanation on why we need a dummy input
# https://github.com/onnx/tutorials/issues/158
dummy_input = torch.randn(1, 6, 20, 20) 
torch.onnx.export(loadedmodel,
                  dummy_input, 
                  'model.onnx', 
                  input_names=['input'], 
                  output_names=['output'])

```

Once the the model is exported in ONNX format, it's time to import it into cube ai 7.1.0. STM has a CLI tool `stm32ai` that imports the ONNX model and generates the corresponding C files. 

```sh
$ stm32ai generate -m <model_path>/model.onnx --type onnx -o <output_dir> --name <project>
```
I wrote a script for that ([link](/trainer/create_C_files_for_stm32.sh)).  The script generates the following files:

```sh
$  ll
drwxrwxr-x 2 me me   4096 13.01.2023 22:56 stm32ai_ws/
-rw-rw-r-- 1 me me  24356 13.01.2023 22:56 model.c
-rw-rw-r-- 1 me me   1500 13.01.2023 22:56 model_config.h
-rw-rw-r-- 1 me me  90718 13.01.2023 22:56 model_data.c
-rw-rw-r-- 1 me me   2624 13.01.2023 22:56 model_data.h
-rw-rw-r-- 1 me me  17926 13.01.2023 22:56 model_generate_report.txt
-rw-rw-r-- 1 me me   8766 13.01.2023 22:56 model.h
```

[Here](/docs/embedded_client_api.html) you can find the documentation in HTML of the API for the CUBE-AI framework. 


## C Implementation Details
For the deployment the following modules were required:

- Ring buffer of size 600x6, because the sampling rate of the sensor is 200Hz and each sample is 2 seconds that means I needed at least an array of (200x2x6) elements.
- Normalization of the samples between (0,1) before feeding the model.
- Inference module with a threshold to detect the vowels. I set the threshold at 0.8. 

Since the model is small and the quantization model from Pytorch is not as good as from Tensorflow, I decide to use floats for my inference. In the future, I would build a quantized model using tensorflow and see how it performs. 

# Results
   



