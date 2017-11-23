
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import glob, os    
import difflib
import re
import datetime
import shutil
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.constraints import maxnorm
imgpath = os.getcwd()+'/img'
# #Functions
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
    
#     elif 'ice' in string:
#         string = 'hail'
    
    elif 'storms' in string:
        string = 'rain'
    
    elif 'ice' in string:
        string ='snow'
        
#     print(or_string,'|',string)
    return string

#1. Read all CSV files. 
path =os.getcwd()+'/yvr-weather' # use your path


allFiles = glob.glob(path+'/*.csv')
data = pd.DataFrame()
dataframes = []
for file_ in allFiles:

    df = pd.read_csv(file_,index_col=None, skiprows=15)

    dataframes.append(df)

data = pd.concat(dataframes)

# print(data.head)


#2. Drop unnecessary columns
data = data[['Date/Time','Weather']]
data['Date/Time'] = data['Date/Time'].values

#3. Impute missing weather information by backfilling with limit of 1 empty row.
#Drop all rows with NaN in the Weather column after
data = data.fillna(method='bfill',limit=1)
data = data.fillna(method='ffill',limit=1)
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
# labels = data["Weather"].unique()

labels=['clear','cloudy','rain','snow']
# There is only one sample with weather label 'ice pellets'. Moreover, there is no corresponding input image.
# Drop this label

# Create label subfolders
for l in labels:
    if not os.path.exists('img/'+l):
        os.makedirs('img/'+l)

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
    
        print(path_file_name," | ",dest)
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
    
    num_files = len([1 for x in list(os.scandir(path)) if x.is_file()]) 
    if num_files < 1:
        continue
    rand_array = np.random.randint(low=0, high=num_files,size = int(num_files*validation_size))
    
    
    for i,file_name in enumerate(glob.iglob(imgpath+'/'+l+'/*.jpg')):
      
        if i in rand_array:
            dest = val_path+'/'+str(i)+'.jpg'
            print(dest)
            shutil.move(file_name,dest)
        else:
            dest = train_path+'/'+str(i)+'.jpg'
            print(dest)
            shutil.move(file_name,dest)
        
        
        
# 5. datagenerator for training and validation
datagen = ImageDataGenerator( rotation_range=40,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                rescale=1./255,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                data_format="channels_last",
                                fill_mode='nearest')

train_gen = datagen.flow_from_directory(imgpath+'/training',
                                        batch_size=32,
                                        target_size=(32,24),
                                        class_mode='categorical'
                                         )
validation_gen = datagen.flow_from_directory(imgpath+'/validation',
                                             target_size=(32,24),
                                             batch_size=16,
                                            class_mode='categorical')


#CNN model 
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32, 24,3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))
# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit_generator(
        train_gen,
        steps_per_epoch=1000,
        epochs=epochs,
        validation_data=validation_gen,
        validation_steps=100)

#CNN model weights
model.save_weights('weights.h5') 