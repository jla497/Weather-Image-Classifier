
# Webcams, Predictions, and Weather

Script to process the YVR-weather and KatKam image data into training and validation datasets, extracts image features from a VGG16 CNN, then runs various machine learning algorithms on the features.

## Getting Started

Follow instructions below.

### Prerequisites

Python2.7 (Keras predict_gen has issues with Python3)
Tensorflow
Theanos
Keras

1. Download, extract, and move katkam-scaled folder into the main folder.

2. Download, extract, and move the 'yvr-weather' folder into the main folder.

3. In CLI, run python Weather+Webcam+.py. 

Directory Structure:

- Main/ (Directory containing Weather+Webcam+.py)
 - katkam-scaled/
 - yvr-weather/
 
### Running the Script

usage: Weather+Webcam+.py [-p] [-vgg16] [-n] [-s] [-r] [-t PREDICT_IMG] [-h]

Script to run the program.

##### arguments:

  -p, --process_data    [Run this first time to process the image and weather
                        dataset.]
                        
  -vgg16, --get_features_from_vgg16
                        [Run this the first time to extract features from the
                        image dataset off VGG16.]
                        
  -n, --test_neuralnet  [Run this to test the features on a neural net.]
  
  -s, --test_svm        [Run this to test the features on a svm.]
  
  -r, --test_randomforest
                        [Run this to test the features on a random forest.]
                        
  -t PREDICT_IMG, --predict_img PREDICT_IMG
                        [Run the rf classifier to predict the weather on the
                        img.]
                        
  -h, --help            [Show this message and exit.]
  
##### Must run --process_data --get_features_from_vgg16 first time to process datasets and extract features.

### Results

#### Neural Net Score on Features: 
Train on 960 samples, validate on 320 samples

Epoch 1/50

960/960 [==============================] - 0s - loss: 6.9854 - acc: 0.4073 - val_loss: 4.6032 - val_acc: 0.4625
Epoch 2/50
960/960 [==============================] - 0s - loss: 1.4568 - acc: 0.6583 - val_loss: 0.4359 - val_acc: 0.8500
Epoch 3/50
960/960 [==============================] - 0s - loss: 0.7560 - acc: 0.7260 - val_loss: 0.4019 - val_acc: 0.8625
Epoch 4/50
960/960 [==============================] - 0s - loss: 0.6261 - acc: 0.7469 - val_loss: 0.5478 - val_acc: 0.7375
Epoch 5/50
960/960 [==============================] - 0s - loss: 0.5714 - acc: 0.7573 - val_loss: 0.3984 - val_acc: 0.8688
Epoch 6/50
960/960 [==============================] - 0s - loss: 0.4981 - acc: 0.8073 - val_loss: 0.3282 - val_acc: 0.9031
Epoch 7/50
960/960 [==============================] - 0s - loss: 0.5325 - acc: 0.8021 - val_loss: 0.4513 - val_acc: 0.8281
Epoch 8/50
960/960 [==============================] - 0s - loss: 0.4686 - acc: 0.8104 - val_loss: 0.2360 - val_acc: 0.9281
Epoch 9/50
960/960 [==============================] - 0s - loss: 0.4329 - acc: 0.8406 - val_loss: 0.2429 - val_acc: 0.9187
Epoch 10/50
960/960 [==============================] - 0s - loss: 0.3842 - acc: 0.8573 - val_loss: 0.2680 - val_acc: 0.9031
Epoch 11/50
960/960 [==============================] - 0s - loss: 0.4443 - acc: 0.8333 - val_loss: 0.2411 - val_acc: 0.9250
Epoch 12/50
960/960 [==============================] - 0s - loss: 0.3445 - acc: 0.8646 - val_loss: 0.2071 - val_acc: 0.9187
Epoch 13/50
960/960 [==============================] - 0s - loss: 0.3142 - acc: 0.8865 - val_loss: 0.2303 - val_acc: 0.9219
Epoch 14/50
960/960 [==============================] - 0s - loss: 0.3554 - acc: 0.8615 - val_loss: 0.3317 - val_acc: 0.8656
Epoch 15/50
960/960 [==============================] - 0s - loss: 0.3341 - acc: 0.8646 - val_loss: 0.2045 - val_acc: 0.9281
Epoch 16/50
960/960 [==============================] - 0s - loss: 0.2937 - acc: 0.8885 - val_loss: 0.2032 - val_acc: 0.9344
Epoch 17/50
960/960 [==============================] - 0s - loss: 0.3247 - acc: 0.8760 - val_loss: 0.2576 - val_acc: 0.9156
Epoch 18/50
960/960 [==============================] - 0s - loss: 0.3101 - acc: 0.8740 - val_loss: 0.2434 - val_acc: 0.9062
Epoch 19/50
960/960 [==============================] - 0s - loss: 0.2967 - acc: 0.8854 - val_loss: 0.2169 - val_acc: 0.9219
Epoch 20/50
960/960 [==============================] - 0s - loss: 0.2695 - acc: 0.9000 - val_loss: 0.1810 - val_acc: 0.9375
Epoch 21/50
960/960 [==============================] - 0s - loss: 0.2840 - acc: 0.8927 - val_loss: 0.2238 - val_acc: 0.9156
Epoch 22/50
960/960 [==============================] - 0s - loss: 0.2764 - acc: 0.8948 - val_loss: 0.2972 - val_acc: 0.8938
Epoch 23/50
960/960 [==============================] - 0s - loss: 0.2746 - acc: 0.8979 - val_loss: 0.1975 - val_acc: 0.9313
Epoch 24/50
960/960 [==============================] - 0s - loss: 0.2734 - acc: 0.9031 - val_loss: 0.2031 - val_acc: 0.9344
Epoch 25/50
960/960 [==============================] - 0s - loss: 0.2635 - acc: 0.9042 - val_loss: 0.5730 - val_acc: 0.8000
Epoch 26/50
960/960 [==============================] - 0s - loss: 0.2509 - acc: 0.9062 - val_loss: 0.2216 - val_acc: 0.9281
Epoch 27/50
960/960 [==============================] - 0s - loss: 0.2690 - acc: 0.9125 - val_loss: 0.2149 - val_acc: 0.9281
Epoch 28/50
960/960 [==============================] - 0s - loss: 0.2628 - acc: 0.9062 - val_loss: 0.2199 - val_acc: 0.9125
Epoch 29/50
960/960 [==============================] - 0s - loss: 0.2301 - acc: 0.9135 - val_loss: 0.2095 - val_acc: 0.9313
Epoch 30/50
960/960 [==============================] - 0s - loss: 0.2471 - acc: 0.9115 - val_loss: 0.3005 - val_acc: 0.9031
Epoch 31/50
960/960 [==============================] - 0s - loss: 0.2692 - acc: 0.9042 - val_loss: 0.1940 - val_acc: 0.9437
Epoch 32/50
960/960 [==============================] - 0s - loss: 0.2198 - acc: 0.9115 - val_loss: 0.2823 - val_acc: 0.9094
Epoch 33/50
960/960 [==============================] - 0s - loss: 0.2339 - acc: 0.9083 - val_loss: 0.2157 - val_acc: 0.9187
Epoch 34/50
960/960 [==============================] - 0s - loss: 0.2305 - acc: 0.9104 - val_loss: 0.1978 - val_acc: 0.9250
Epoch 35/50
960/960 [==============================] - 0s - loss: 0.2384 - acc: 0.9167 - val_loss: 0.1724 - val_acc: 0.9437
Epoch 36/50
960/960 [==============================] - 0s - loss: 0.1790 - acc: 0.9219 - val_loss: 0.2490 - val_acc: 0.9156
Epoch 37/50
960/960 [==============================] - 0s - loss: 0.2232 - acc: 0.9198 - val_loss: 0.1869 - val_acc: 0.9344
Epoch 38/50
960/960 [==============================] - 0s - loss: 0.2001 - acc: 0.9167 - val_loss: 0.1896 - val_acc: 0.9375
Epoch 39/50
960/960 [==============================] - 0s - loss: 0.2216 - acc: 0.9083 - val_loss: 0.1673 - val_acc: 0.9406
Epoch 40/50
960/960 [==============================] - 0s - loss: 0.2357 - acc: 0.9198 - val_loss: 0.1807 - val_acc: 0.9437
Epoch 41/50
960/960 [==============================] - 0s - loss: 0.1746 - acc: 0.9281 - val_loss: 0.2662 - val_acc: 0.9187
Epoch 42/50
960/960 [==============================] - 0s - loss: 0.1734 - acc: 0.9323 - val_loss: 0.2278 - val_acc: 0.9375
Epoch 43/50
960/960 [==============================] - 0s - loss: 0.1845 - acc: 0.9281 - val_loss: 0.2162 - val_acc: 0.9281
Epoch 44/50
960/960 [==============================] - 0s - loss: 0.1807 - acc: 0.9229 - val_loss: 0.2068 - val_acc: 0.9281
Epoch 45/50
960/960 [==============================] - 0s - loss: 0.2058 - acc: 0.9292 - val_loss: 0.1948 - val_acc: 0.9500
Epoch 46/50
960/960 [==============================] - 0s - loss: 0.1851 - acc: 0.9344 - val_loss: 0.1784 - val_acc: 0.9406
Epoch 47/50
960/960 [==============================] - 0s - loss: 0.1949 - acc: 0.9354 - val_loss: 0.2212 - val_acc: 0.9313
Epoch 48/50
960/960 [==============================] - 0s - loss: 0.1892 - acc: 0.9260 - val_loss: 0.2618 - val_acc: 0.9344
Epoch 49/50
960/960 [==============================] - 0s - loss: 0.2178 - acc: 0.9240 - val_loss: 0.2109 - val_acc: 0.9500
Epoch 50/50
960/960 [==============================] - 0s - loss: 0.1973 - acc: 0.9281 - val_loss: 0.2141 - val_acc: 0.9375

#### Random Forest score on features:
('random forest accuracy on validation data: ', 0.94374999999999998)


### Image Predictions

![](cloudy_predict.png)

![](snow_predict.png)




