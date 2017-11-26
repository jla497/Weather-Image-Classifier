import pandas as pd
import numpy as np
import glob, os    
import difflib
import re
import datetime
import shutil
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model,Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras import applications
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from scipy.misc import imresize
from keras.utils.np_utils import to_categorical
from argparse import ArgumentParser
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

path = os.getcwd()
imgpath = path+'/katkam-scaled'
num_trg_samples = 960
num_val_samples = 320
num_test_samples = 1568
img_x = 224
img_y = 224
batch_size = 32
epochs = 50
labels = []

#Functions
def get_matches(row):
    string = row['Weather'].strip()
    
    string = string.lower()
   
    or_string = string
    
    if 'clear' in string:
       
        string = 'clear'
    
    elif 'cloudy' in string:
        string = 'cloudy'
    
    elif 'snow' in string:
        string = 'snow'
    
    elif 'drizzle' in string:
        string = 'rain'
    
    elif 'shower' in string:
        string = 'rain'
    
    elif 'fog' in string:
        string = 'cloudy'
        
    elif 'rain' in string:
        string = 'rain'
    
    elif 'ice' in string:
        string = 'snow'
    
    elif 'storms' in string:
        string = 'rain'
    
#     print(or_string,'|',string)
    return string

def process_data():
    #1. Read all CSV files. 
    path_w =os.getcwd()+'/yvr-weather' # use your path
    allFiles = glob.glob(path_w + "/*.csv")
    data = pd.DataFrame()
    dataframes = []
    for file_ in allFiles:
        df = pd.read_csv(file_,index_col=None, skiprows=15)
        dataframes.append(df)
    data = pd.concat(dataframes)
    # print(frame.columns)
    print("Finished loading yvr-weather dataset...")
    
    #2. Drop unnecessary columns
    data = data[['Date/Time','Weather']]
    data['Date/Time'] = data['Date/Time'].values

    #3. Impute missing weather information by backfilling with limit of 1 empty row.
    #Drop all rows with NaN in the Weather column after
    data = data.fillna(method='bfill',limit=1)
    data = data.dropna(axis=0)
    # print(data.head)

    #4. Find all unique weather labels
    # print(data['Weather'].unique())

    #I separate multi-class labelled samples out for now to make the model simpler
    multi_labels = data.loc[data['Weather'].str.contains(',')]
    data = data.loc[data['Weather'].str.contains(',')==False]
    # print(data["Weather"].unique())

    #I chose the following labels for simplicity: Clear, Cloudy, Stormy, Rain, Fog, Snow, Ice Pellets
    data['Weather'] = data.apply(get_matches,axis=1)
    labels = data["Weather"].unique()

    # FYI, there is only one sample with weather label 'hail'. Moreover, there is no corresponding input image.

    # Create label subfolders
    for l in labels:
        if not os.path.exists(imgpath+'/'+l):
            os.makedirs(imgpath+'/'+l)

    #convert data['Date/Time'] from string to datetime object
    data["Date/Time"] = data["Date/Time"].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d %H:00"))

    # 5. Extract date time from each img filenames and move img to corresponding label subfolders
    for path_file_name in glob.iglob(imgpath+'/*.jpg'):
        title,ext = os.path.splitext(os.path.basename(path_file_name))
        match = re.search(r'(\d{4})(\d{2})(\d{2})(\d{2})',title)

        date_time = match.group(1)+"-"+match.group(2)+"-"+match.group(3)+' '+match.group(4)

        datetime_obj = datetime.datetime.strptime(date_time,"%Y-%m-%d %H")

        if data[data["Date/Time"]==datetime_obj].Weather.empty == False:

            subfolder = data[data['Date/Time']==date_time].Weather.to_string()
            subfolder = subfolder.split("  ")[2]
            subfolder = subfolder.strip()
            new_file_name = match.group(1)+"_"+match.group(2)+"_"+match.group(3)+'_'+match.group(4)+".jpg"
            dest = imgpath+'/'+subfolder+'/'+new_file_name

            #print(path_file_name," | ",dest)
            shutil.move(path_file_name,dest)

    #6. split label subfolders into training and validation folders

    #run the below commands once to create directories
    if not os.path.exists(imgpath+'/training'):
        os.makedirs(imgpath+'/training')
    if not os.path.exists(imgpath+'/validation'):
        os.makedirs(imgpath+'/validation')

    for l in labels:
        if not os.path.exists(imgpath+'/training/'+l):
            os.makedirs(imgpath+'/training/'+l)
        if not os.path.exists(imgpath+'/validation/'+l):
            os.makedirs(imgpath+'/validation/'+l)

    validation_size = 0.3
    for l in labels:
        path = imgpath+'/'+l
        train_path = imgpath+'/training/'+l
        val_path = imgpath+'/validation/'+l

        num_files = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]) 
        if num_files < 1:
            continue
        rand_array = np.random.randint(low=0, high=num_files,size = int(num_files*validation_size))


        for i,file_name in enumerate(glob.iglob(imgpath+'/'+l+'/*.jpg')):

            if i in rand_array:
                dest = val_path+'/'+str(i)+'.jpg'
                #print(dest)
                shutil.move(file_name,dest)
            else:
                dest = train_path+'/'+str(i)+'.jpg'
                #print(dest)
                shutil.move(file_name,dest)
    
    print("Finished matching katkam image dataset to corresponding weather labels...")
    
    #decrease number of images in training and validation dataset to even out the distribution over 4 labels
    for l in labels:
        train_path = imgpath+'/training/'+l
        
        num_imgs = {'clear':300,'cloudy':300,'rain':300,'snow':60}
        
        folder_size = num_imgs[l]
        
        for i,file_name in enumerate(glob.iglob(train_path+'/*.jpg')):   
            #delete files with index > folder_size
            if i >= folder_size:

                os.unlink(file_name)
        
    for l in labels:
        val_path = imgpath+'/validation/'+l
        
        num_imgs = {'clear':100,'cloudy':100,'rain':100,'snow':20}
        
        folder_size = num_imgs[l]
        
        for i,file_name in enumerate(glob.iglob(val_path+'/*.jpg')):   
            #delete files with index > folder_size
            if i >= folder_size:

                os.unlink(file_name)
                
    print("Finished processing the image dataset...")
    
def VGG16_feature_extraction():
    """
    Bottleneck feature extraction from VGG16 Code reference:
    https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069

    """
    datagen = ImageDataGenerator(rescale= 1./255)

    model = applications.VGG16(include_top=False, weights='imagenet')

    gen = datagen.flow_from_directory(imgpath+'/training',
                                        batch_size=batch_size,
                                        target_size=(img_x,img_y),
                                        class_mode=None,
                                        shuffle=False
                                       )

    bottleneck_features_train = model.predict_generator(gen, num_trg_samples // batch_size,verbose=1)

    np.save(open('bottleneck_features_train.npy', 'w'),
            bottleneck_features_train)

    gen = datagen.flow_from_directory(imgpath+'/validation',
                                        batch_size=batch_size,
                                        target_size=(img_x,img_y),
                                        class_mode=None,
                                        shuffle=False
                                       )
    
    bottleneck_features_validation = model.predict_generator(gen, num_val_samples // batch_size,verbose=1)

    np.save(open('bottleneck_features_validation.npy', 'w'),
            bottleneck_features_validation)

def top_model_compile(train_data):
    """
     
    function to compile simple neural net
     
    """
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    #lrate = 0.01
    #decay = lrate/epochs
    #sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    #model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def train_test_nn():
    """

    function to test nn on extracted features

    """
    with open('bottleneck_features_train.npy','rb') as f:

        train_data = np.load(f)

    with open('bottleneck_features_validation.npy','rb') as f:

        validation_data = np.load(f)
        
    train_labels = np.array(
        [0] * (300) + [1] * (300) + [2] * (300) + [3] * (60))

    validation_labels = np.array(
        [0] * (100) + [1] * (100) + [2] *(100) + [3] * (20))

    #train_labels = train_labels.reshape((,train_labels.shape[0]))

    #validation_labels = validation_labels.reshape((,validation_labels.shape[0]))

    train_labels = to_categorical(train_labels)
    validation_labels = to_categorical(validation_labels) 

    model = top_model_compile(train_data)

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
	      validation_data=(validation_data,validation_labels))

    #print("Neural net score on validation data: ",model.evaluate(validation_data,validation_labels))
    model.save_weights("top_model_weights.h5")

    return model
    
def fine_tune():
    """
    function to add a top model to vgg16 then fine-tune on the image dataset

    """

    vgg16_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_x,img_y,3))

    last = vgg16_model.output

    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(4, activation='softmax'))

    # note that it is necessary to start with a fully-trained
    # classifier, including the top classifier,
    # in order to successfully do fine-tuning
    top_model.load_weights('top_model_weights.h5')

    agg_model = Model(inputs=vgg16_model.input,outputs=top_model(vgg16_model.output))   


    # set the layers of vgg16 untrainable
    # to non-trainable (weights will not be updated)
    for layer in agg_model.layers[:25]:
        layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.

    epochs = 50
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

    agg_model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_gen = train_datagen.flow_from_directory(imgpath+'/training',
                                                  batch_size=batch_size,
                                                  target_size=(img_x,img_y),
                                                  class_mode='categorical')

    validation_gen = test_datagen.flow_from_directory(imgpath+'/validation',
                                                      target_size=(img_x,img_y),
                                                      batch_size=batch_size,
                                                      class_mode='categorical')


    # fine-tune the model
    agg_model.fit_generator(
        train_gen,
        samples_per_epoch=num_trg_samples,
        epochs=epochs,
        validation_data=validation_gen,
        nb_val_samples=num_val_samples)


def test_rf():
    """

     function to test rf on extracted features

    """

    validation_data = None
    train_data = None

    with open('bottleneck_features_train.npy','rb') as f:

        train_data = np.load(f)

    with open('bottleneck_features_validation.npy','rb') as f:

        validation_data = np.load(f)

    train_labels = np.array(
            [0] * (300) + [1] * (300) + [2] * (300) + [3] * (60))


    validation_labels = np.array(
        [0] * (100) + [1] * (100) + [2] *(100) + [3] * (20))

    train_data = train_data.reshape((960,-1))

    validation_data = validation_data.reshape((320,-1))

    clf = RandomForestClassifier(n_estimators=90,random_state=39)

    clf.fit(train_data,train_labels)
    print("random forest accuracy on validation data: ",clf.score(validation_data,validation_labels))
    return clf

def test_svm():
    """
     run SVM on features extracted from VGG16

    """
    validation_data = None
    train_data = None

    with open('bottleneck_features_train.npy','rb') as f:

        train_data = np.load(f)

    with open('bottleneck_features_validation.npy','rb') as f:

        validation_data = np.load(f)

    train_labels = np.array(
            [0] * (300) + [1] * (300) + [2] * (300) + [3] * (60))


    validation_labels = np.array(
        [0] * (100) + [1] * (100) + [2] *(100) + [3] * (20))

    train_data = train_data.reshape((960,-1))

    validation_data = validation_data.reshape((320,-1))

    clf = SVC()

    clf.fit(train_data,train_labels)

    print("svm accuracy on validation data: ",clf.score(validation_data,validation_labels))

def predict_img(filename):

    bmodel = applications.VGG16(include_top=False, weights='imagenet', input_shape=(img_x,img_y,3))

    img = image.load_img(filename,target_size=(img_x,img_y))

    x = image.img_to_array(img)

    x = np.expand_dims(x,axis=0)

    x = preprocess_input(x)

    x = x/255
    #print(x.shape)

    features = bmodel.predict(x)

    model = top_model_compile(features)

    model.load_weights('top_model_weights.h5')

    predictions = model.predict(features)
    
    print("The weather in the image is: ",decode_pred(predictions))
    
    #features = features.reshape((1,-1))

    #clf = test_rf()
    #pred = clf.predict(features)
        
    #print("random_forest - The weather in the image is: ",decode_pred(pred[0]))

def decode_pred(arr):
    
    mdict = {0:'clear',1:'cloudy',2:'rain',3:'snow'}
    if type(arr) == int:
        return mdict[arr]

    x = np.argmax(arr)
    return mdict[x]

def main():

    #check if img folder exists
    if os.path.isdir(imgpath) is False:
        print("Missing 'katkam-scaled' folder")
        return

    #check if yvr-weather folder is missing
    if os.path.isdir(os.getcwd()+'/yvr-weather') is False:
        print("Missing yvr-weather folder")
        return


    argument_parser = ArgumentParser(
        description="Script to run the program.",
        add_help=False)

    argument_parser.add_argument(
    '-p', '--process_data',
    action='store_true',
    required=False,
    help="Run this first time to process the image and weather dataset.")  

    argument_parser.add_argument(
    '-vgg16', '--get_features_from_vgg16',
    required=False,
    action='store_true',
    help="Run this the first time to extract features from the image dataset off VGG16.")

    argument_parser.add_argument(
    '-n', '--test_neuralnet',
    required=False,
    action='store_true',
    help="Run this to test the features on a neural net.")

    argument_parser.add_argument(
    '-s', '--test_svm',
    required=False,
    action='store_true',
    help="Run this to test the features on a svm.")

    argument_parser.add_argument(
    '-r', '--test_randomforest',
    action='store_true',
    required=False,
    help="Run this to test the features on a random forest.")

    argument_parser.add_argument(
    '-t', '--predict_img',
    type=str,
    help="Run the rf classifier to predict the weather on the img.")
    
    argument_parser.add_argument(
        '-h', '--help',
        action='help',
        help="Show this message and exit.")

    arguments = argument_parser.parse_args()

    if arguments.process_data is True:
          process_data()

    if arguments.get_features_from_vgg16 is True:
          VGG16_feature_extraction()

    if arguments.test_neuralnet is True:
          train_test_nn()

    if arguments.test_randomforest is True:
          test_rf()

    if arguments.test_svm is True:
          test_svm()

    if arguments.predict_img != None:
          predict_img(arguments.predict_img)

if __name__=='__main__':
    main()
