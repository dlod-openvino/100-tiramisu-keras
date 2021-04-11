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