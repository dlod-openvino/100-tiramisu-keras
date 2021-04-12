
# Semantic segmentation with The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation

Updated by alex zhang

What are updated:
1. Installation guide from the scratch
2. update some code to make it supported by TensorFlow 2.2.0
3. Provide OpenVINO 2021.3 support
4. provide OpenVINO Inference code for benchmark in Intel CPU/iGPU/VPU ...

Original paper:
https://arxiv.org/abs/1611.09326

Keras implementation of 100 layer Tiramisu for semantic segmentaton. Model FC-DenseNet103 from the paper above.

![alt text](https://raw.githubusercontent.com/xxmarl/100-tiramisu-keras/master/images/test_image3_small.png)
![alt text](https://raw.githubusercontent.com/xxmarl/100-tiramisu-keras/master/images/test_image3_outcome.png)

Tested with:
Python 3.8.2  
Tensorflow 2.2.0  
Keras 2.4.3 

## Installation guide
### Step 1: Create a environment in Anaconda with python=3.8:
~~~
conda create -n tf2_2 python=3.8
~~~
### Step 2: Install tensorflow 2.2:
~~~
pip install --ignore-installed --upgrade tensorflow==2.2.0
~~~
### Step 3: Install cudatoolkit=10.1 cudnn=7.6.5
~~~
conda install cudatoolkit=10.1 cudnn=7.6.5
~~~
### Step 4: Install Keras==2.4.3 pillow
~~~
pip install keras==2.4.3 pillow
~~~



## Running a demo
Before running a demo, download weights trained on CamVid data from the following link and place it under models\\  
https://drive.google.com/file/d/1T7GP7h0Q8DMLCQ3vgQdadBrFD_vZ9io3/view?usp=sharing

or Download from Baidu-online-storage
Link：https://pan.baidu.com/s/1Jy3NwJQOZN4C8TDh5739Mw 
PWD：9xy0 

Test network trained with CamVid data on custom image by running:  
~~~
python run_tiramisu_camvid.py
~~~

With optional arguments:  
~~~
optional arguments:
  -h, --help            show this help message and exit
  --path_to_test_file PATH_TO_TEST_FILE
                        Path to the image you would like to test with. Default
                        is: images/testImage0.png
  --path_to_result PATH_TO_RESULT
                        Path to the folder and filename where the result of
                        segmentation should be saved. Default is:
                        images/test_image1_outcome.png
  --path_to_model PATH_TO_MODEL
                        Path to the h5 file with the model weight that should
                        be used for inference. Default is:
                        models/my_tiramisu.h5
  --path_to_labels_list PATH_TO_LABELS_LIST
                        Path to file defining classes used in camvid dataset.
                        Only used if convert_from_camvid = True. Default is
                        camvid-master/label_colors.txt
~~~
## Training

Run the following to train with default configuration (training on CamVid dataset) - Please download the dataset from Baidu-online-storage
Link：https://pan.baidu.com/s/1Jy3NwJQOZN4C8TDh5739Mw 
PWD：9xy0 

Save the png files into 
camvid-master\train  #This folder is for images
<<<<<<< HEAD
camvid-master\trainannot   #This folder is for the image annotations
=======
camvid-master\trainannot  #This folder is for image annotations
>>>>>>> d4be29e0774eeee56fcba90df79cc0d13eb680e1
```
python train.py
```
with optional arguments:

```
  --output_path OUTPUT_PATH
                        Path for saving a training model as a *.h5 file.
                        Default is models/new_model.h5
  --path_to_raw PATH_TO_RAW
                        Path to raw images used for training. Default is
                        camvid-master/train/
  --image_size IMAGE_SIZE
                        Size of the input image. Default is [360, 480]
  --path_to_labels PATH_TO_LABELS
                        Path to labeled images used for training. Default is
                        camvid-master/trainannot/
  --path_to_labels_list PATH_TO_LABELS_LIST
                        Path to file defining classes used in camvid dataset.
                        Only used if convert_from_camvid = True. Default is
                        camvid-master/label_colors.txt
  --log_dir LOG_DIR     Path for storing tensorboard logging. Default is
                        logging/
  --convert_from_camvid CONVERT_FROM_CAMVID
                        Flag that defines if camvid data is used. If enabled
                        it maps camvid data labeling to integers. Default:
                        True
  --training_percentage TRAINING_PERCENTAGE
                        Defines percentage of total data that will be used for
                        training. Default: 70 training 30 validation
  --no_epochs NO_EPOCHS
                        Defines number of epochs used for training. Default:
                        250
  --learning_rate LEARNING_RATE
                        Defines learning rate used for training. Default: 1e-3
  --patience PATIENCE   Defines patience for early stopping. Default: 50
  --path_to_model_weights PATH_TO_MODEL_WEIGHTS
                        Path to saved model weights if training should be
                        resumed. Default: models/new_model.h5
  --train_from_zero TRAIN_FROM_ZERO
                        Boolean, defines if training from scratch or resuming
                        from saved h5 file. Default: True
```
Tensorboard is supported to view run:
~~~
tensorboard --logdir=path/to/log-directory
~~~
## Convert the model into ONNX format
Starting from the 2020.4 release, OpenVINO™ supports reading native ONNX models. Core::ReadNetwork() method provides a uniform way to read models from IR or ONNX format, it is a recommended approach to reading models. Example:
~~~
InferenceEngine::Core core;
auto network = core.ReadNetwork("model.onnx");
~~~
**STEP1**: Installation keras2onnx refer to: https://pypi.org/project/keras2onnx/
install from source:
~~~
pip install -U git+https://github.com/microsoft/onnxconverter-common
pip install -U git+https://github.com/onnx/keras-onnx
~~~

**STEP2**: Convert the Keras h5 model to ONNX model, run the convert_to_onnx.py script as below:
```python
from keras.models import Model
from keras.layers import *
from tiramisu.model import create_tiramisu
import keras2onnx
# Set the weight file name
keras_model_weights = "models/my_tiramisu.h5"
onnx_model_weights = keras_model_weights.split('.')[0]+'.onnx'
# Load model and weights
input_shape = (224, 224, 3)
number_classes = 32  # CamVid data consist of 32 classes
# Prepare the model information
img_input = Input(shape=input_shape, batch_size=1)
x = create_tiramisu(number_classes, img_input)
model = Model(img_input, x)
# Load the keras model weights
model.load_weights(keras_model_weights)
onnx_model = keras2onnx.convert_keras(model, model.name)
# Save the onnx model weights
keras2onnx.save_model(onnx_model, onnx_model_weights)
```
run the convert_to_onnx.py script
~~~
python convert_to_onnx.py
~~~
**STEP3**: Install OpenVINO 2021.2: openvino_2021.2.185

**STEP4**: Initialize the OpenVINO environment
~~~
 c:\Program Files (x86)\Intel\openvino_2021.2.185>bin\setupvars.bat
~~~

**STEP5**: Convert the onnx model to IR model
~~~
 c:\Program Files (x86)\Intel\openvino_2021.2.185\deployment_tools\model_optimizer>python mo_onnx.py --input_model d:\100-tiramisu-keras\models\my_tiramisu.onnx --output_dir d:\100-tiramisu-keras\models
~~~

**STEP6**: run the ov_infer_demo.py to do the inference based on OpenVINO
~~~
 D:\100-tiramisu-keras>python ov_infer_demo.py -d CPU
~~~
## Notes
CamVid dataset: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/
