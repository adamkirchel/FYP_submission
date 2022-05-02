import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from os import walk
from density_estimator import DensityEstimate
from PIL import Image as im
import time
import random
import cv2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from PIL import Image as im
from PIL import ImageOps
import math
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
import glob
from tensorflow.keras.preprocessing.image import load_img, array_to_img, img_to_array
import random
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

class Architecture:

    ## Unit testing ##

    # Save tracking density files
    def SaveData(path):

        # Find file names
        dist_files = []
        for (dirpath, dirnames, filenames) in walk(path + '/raw'):
            dist_files.extend(filenames)
            break

        # Create densities and save as image
        distributions = []
        for i,name in enumerate(dist_files):
            model = DensityEstimate(path + '/raw/' + name,10)
            density = model.density
            max_d = np.max(density)
            min_d = np.min(density)
            new_density = (density - min_d)/(max_d - min_d)

            data = im.fromarray(np.uint8(new_density * 255) ,'L')
            data.save(path + '/images/' + name[:-4] + '.png')

    # Calculate distances between nodes in preliminary layer
    def CalculateDistance(self,x_index,y_index,profile,distribution,n=9):

        xmin = x_index - 1
        xmax = x_index + 1
        ymin = y_index - 1
        ymax = y_index + 1

        x_i = 1
        y_i = 1

        # Define boundaries
        if xmin < 0:
            xmin = 0
            x_i = 0

        if xmax > profile.shape[1]-1:
            xmax = profile.shape[1]-1

        if ymin < 0:
            ymin = 0
            y_i = 0

        if ymax > profile.shape[0]-1:
            ymax = profile.shape[0]-1

        # Array of n-nearest nodes
        new_arr = profile[ymin:ymax+1,xmin:xmax+1]

        data = []

        # Calculate distances
        for j,elem in np.ndenumerate(new_arr):
            distance = math.sqrt(((j[0] - y_i)**2) + ((j[1] - x_i)**2))
            data.append([distance,elem,distribution[y_index,x_index]])
        return data

    # Get csv data
    def GetRawData(self,path):

        # Get filenames
        files = []
        for (dirpath, dirnames, filenames) in walk(path):
            files.extend(filenames)
            break

        # Load data
        data = []
        for i,name in enumerate(files):
            with open(path + '/' + name) as file_name:
                data.extend([np.loadtxt(file_name, delimiter=",").tolist()])

        return data

    # Calculate change ingeometrical descriptors
    def GetChange(self,values):

        delta = []
        for i,value in enumerate(values):
            if (i+1) % 4 == 0 | i < 1:
                delta.extend([value])
            else:
                delta.extend([np.subtract(value,values[i-1])])

        return delta

    # Build preliminary layer
    def BuildPreliminary(self,feature_path,label_path,n_nodes,res,change):

        # Get feature data
        values = self.GetRawData(feature_path)

        # Compute change in values if required
        if change:
            # Compute difference in values
            values= self.GetChange(values)

        values = np.array(values)

        # Load distributions
        dist_files = []
        for (dirpath, dirnames, filenames) in walk(label_path):
            dist_files.extend(filenames)
            break

        print('Loading tracking density files complete')

        # Calculate tracking densities
        distributions = []
        for i,name in enumerate(dist_files):
            density = DensityEstimate(label_path + '/' + name,res).density
            distributions.extend([density.tolist()])

        print('Rescale of tracking density complete')

        dist_arr = np.array(distributions)

        data_arr = []

        # Calculate distances - number nodes default at 9
        for i,elem in np.ndenumerate(values):
            data_arr.extend(self.CalculateDistance(i[2],i[1],values[i[0]],dist_arr[i[0]]))

        df = pd.DataFrame(data_arr,columns = ['Distance','Curvature','Density'])

        # Separate labels and features
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Separate training and test set
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

        # Feature scale data
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

        print('Train and test set generated')

        # Build and fit model
        regressor = RandomForestRegressor(n_estimators = 50, random_state = 0)
        regressor.fit(x_train, y_train)
        y_pred = regressor.predict(x_test)
        accuracy = mean_squared_error(y_test, y_pred)

        print('Model created')

        return regressor, accuracy

    # Get prediction of preliminary layer
    def PredictPreliminary(x,feature_path,label_path,n_nodes,res):

        # Build model
        model = BuildPreliminary(feature_path,label_path,n_nodes,res)

        # Predict test set
        y_pred = regressor.predict(x)

        return y_pred

    # Callback function for stopping training early in model
    def get_callbacks(self):
        return [
            tfdocs.modeling.EpochDots(),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
        ]

    # First layer in architecture for unit testing
    def BuildFirstLayer(self,path,size,n_layers,n_neurons,n_epochs):

        # Load train data automatically into tensor
        train_data = tf.keras.utils.image_dataset_from_directory(
            path,
            labels='inferred',
            label_mode='categorical',
            color_mode='grayscale',
            batch_size=8,
            image_size=(size, size),
            seed=123,
            shuffle=True,
            validation_split=0.2,
            subset='training',
            interpolation='bilinear')

        # Load test data automatically into tensor
        test_data = tf.keras.utils.image_dataset_from_directory(
            path,
            labels='inferred',
            label_mode='categorical',
            color_mode='grayscale',
            batch_size=4,
            image_size=(size, size),
            seed=123,
            shuffle=True,
            validation_split=0.2,
            subset='validation',
            interpolation='bilinear')

        class_names = train_data.class_names
        print(class_names)

        # Data augmentation module
        data_augmentation = tf.keras.Sequential(
          [
            tf.keras.layers.RandomFlip("horizontal",input_shape=(size,size,1)),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
          ]
        )

        # Initiate CNN
        cnn = tf.keras.models.Sequential()

        # Augment data
        cnn.add(data_augmentation)

        # Rescale
        cnn.add(tf.keras.layers.Rescaling(1./255, input_shape=(size, size, 1)))

        # Apply convolutional layer
        cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[size, size, 1]))

        # Apply pooling
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        # Apply second convolutional layer
        cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        # Apply second convolutional layer
        #cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
        #cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        cnn.add(tf.keras.layers.Dropout(0.2))

        # Flatten the data
        cnn.add(tf.keras.layers.Flatten())

        for i in range(1,n_layers):
            # Apply full connection
            cnn.add(tf.keras.layers.Dense(units=n_neurons, activation='relu'))

        # Apply output layer
        cnn.add(tf.keras.layers.Dense(units=5, activation='softmax'))

        # Compile CNN
        cnn.compile(optimizer = 'adam',
                    loss = 'categorical_crossentropy',
                    metrics = ['accuracy'])

        # Fitting model and evaluating on test set
        history = cnn.fit(x = train_data, validation_data = test_data, epochs = n_epochs)

        return cnn,test_data,history

    # Predict first layer
    def PredictFirstLayer(x,path,size):

        model = BuildFirstLayer(path,size)

        model.predict(x)

    # Get labels for second layer
    def GetLabels(self,path,cl,params):

        new_path = path + '/' + cl

        files = []
        for (dirpath, dirnames, filenames) in walk(new_path):
            files.extend(filenames)
            break

        labels = []
        for i,name in enumerate(files):
            [num,start,end] = name.split('_')
            start = int(start)
            end = int(end[0])
            num = int(num[-1])
            n_passes = end - start
            norm_passes = (n_passes - 1)/3
            s_params = params[cl][num-1]

            row = []
            if cl == 'c2o':
                width = s_params[0] + (s_params[0] * (s_params[4] - 1))*((start)/3)
                n_width = (width - 50)/350
                height = s_params[1] + (s_params[1] * (s_params[4] - 1))*((start)/3)
                n_height = (height - 50)/350
                xpos = s_params[2]
                n_xpos = (xpos - 140)/110
                ypos = s_params[3]
                n_ypos = (ypos - 140)/110
                scale = (s_params[4]-1)*((n_passes-1)/3)+1
                n_scale = (scale - 1)/4
                labels.append([n_width,n_height,n_xpos,n_ypos,n_scale,norm_passes])
            elif cl == 'over':
                n_width = (s_params[0] - 100)/300
                n_height = (s_params[1] - 50)/350
                n_xpos = (s_params[2] - 100)/150
                n_ypos = (s_params[3] - 100)/150
                labels.append([n_width,n_height,n_xpos,n_ypos,norm_passes])
            elif cl == 'ver':
                width = s_params[0] + (s_params[0] * (s_params[2] - 1))*((start)/3)
                n_width = (width - 50)/350
                x_pos = s_params[1]
                n_xpos = (s_params[1] - 140)/110
                scale = (s_params[2]-1)*((n_passes-1)/3)+1
                n_scale = (scale - 1)/5
                labels.append([n_width,n_xpos,n_scale,norm_passes])
            elif cl == 'hor':
                height = s_params[0] + (s_params[0] * (s_params[2] - 1))*((start)/3)
                n_height = (height - 50)/350
                y_pos = s_params[1]
                n_ypos = (s_params[1] - 140)/110
                scale = (s_params[2]-1)*((n_passes-1)/3)+1
                n_scale = (scale - 1)/5
                labels.append([n_height,n_ypos,n_scale,norm_passes])
            elif cl == 'tri':
                n_width = (s_params[0] - 100)/300
                n_height = (s_params[1] - 100)/300
                n_xpos = (s_params[2] - 100)/150
                n_ypos = (s_params[3] - 50)/200
                start_pass = start/3
                labels.append([n_width,n_height,n_xpos,n_ypos,start_pass,norm_passes])

        return labels

    # Build second layer for unit testing
    def BuildSecondLayer(self,train_data,test_data,size,n_outputs,n_epochs):

        # Data augmentation module
        data_augmentation = tf.keras.Sequential(
          [
            tf.keras.layers.RandomFlip("horizontal",input_shape=(size,size,1)),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
          ]
        )

        # Initiate CNN
        cnn = tf.keras.models.Sequential()

        # Augment data
        cnn.add(data_augmentation)

        # Rescale
        cnn.add(tf.keras.layers.Rescaling(1./255, input_shape=(size, size, 1)))

        # Apply convolutional layer
        cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[size, size, 1]))

        # Apply pooling
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        # Apply second convolutional layer
        cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        cnn.add(tf.keras.layers.Dropout(0.2))

        # Flatten the data
        cnn.add(tf.keras.layers.Flatten())

        # Apply full connection
        cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

        # Apply output layer
        cnn.add(tf.keras.layers.Dense(units=1,activation='linear'))

        # Compile CNN
        cnn.compile(optimizer = 'adam',
                    loss = 'mae',
                    metrics = ['mae'])

        # Fitting model and evaluating on test set
        history = cnn.fit(x = train_data, validation_data = test_data, epochs = n_epochs)

        return cnn,test_data,history

    ## Integration testing ##

    # Reverse engineer output from ml architecture to get strategy parameters
    def ExchangeLabel(self,cl,vals):

        labels = []
        if cl == 'c2o':
            width = (vals[0] * 350) + 50
            height = (vals[1] * 350) + 50
            xpos = (vals[2] * 110) + 140
            ypos = (vals[3] * 110) + 140
            scale = (vals[4]*4) + 1
            passes = (vals[5]*3) + 1
            labels.append([width,height,xpos,ypos,scale,passes])
        elif cl == 'over':
            width = (vals[0] * 300) + 100
            height = (vals[1] * 350) + 50
            xpos = (vals[2] * 150) + 100
            ypos = (vals[3] * 150) + 100
            passes = (vals[4]*3) + 1
            labels.append([width,height,xpos,ypos,passes])
        elif cl == 'ver':
            width = (vals[0] * 350) + 50
            xpos = (vals[1] * 110) + 140
            scale = (vals[2] * 5) + 1
            passes = (vals[3]*3) + 1
            labels.append([width,xpos,scale,passes])
        elif cl == 'hor':
            height = (vals[0] * 350) + 50
            ypos = (vals[1] * 110) + 140
            scale = (vals[2] * 5) + 1
            passes = (vals[3]*3) + 1
            labels.append([height,ypos,scale,passes])
        elif cl == 'tri':
            width = (vals[0] * 300) + 100
            height = (vals[1] * 300) + 100
            xpos = (vals[2] * 150) + 100
            ypos = (vals[3] * 200) + 50
            start_pass = vals[4] * 3
            passes = (vals[5]*3) + 1
            labels.append([width,height,xpos,ypos,start_pass,passes])

        return labels

    # Get labels for first and second layers
    def GetAllLabels(self,path,classes,params):

        labels = []
        for cl in classes:
            new_path = path + '/' + cl

            files = []
            for (dirpath, dirnames, filenames) in walk(new_path):
                files.extend(filenames)
                break

            for i,name in enumerate(files):

                [num,start,end] = name.split('_')
                start = int(start)
                end = int(end[0])
                num = int(num[-1])
                n_passes = end - start
                norm_passes = (n_passes - 1)/3
                s_params = params[cl][num-1]

                row = []
                if cl == 'c2o':
                    width = s_params[0] + (s_params[0] * (s_params[4] - 1))*((start)/3)
                    n_width = (width - 50)/350
                    height = s_params[1] + (s_params[1] * (s_params[4] - 1))*((start)/3)
                    n_height = (height - 50)/350
                    xpos = s_params[2]
                    n_xpos = (xpos - 140)/110
                    ypos = s_params[3]
                    n_ypos = (ypos - 140)/110
                    scale = (s_params[4]-1)*((n_passes-1)/3)+1
                    n_scale = (scale - 1)/4
                    labels.append((cl,[n_width,n_height,n_xpos,n_ypos,n_scale,norm_passes]))
                elif cl == 'over':
                    n_width = (s_params[0] - 100)/300
                    n_height = (s_params[1] - 50)/350
                    n_xpos = (s_params[2] - 100)/150
                    n_ypos = (s_params[3] - 100)/150
                    labels.append((cl,[n_width,n_height,n_xpos,n_ypos,norm_passes]))
                elif cl == 'ver':
                    width = s_params[0] + (s_params[0] * (s_params[2] - 1))*((start)/3)
                    n_width = (width - 50)/350
                    x_pos = s_params[1]
                    n_xpos = (s_params[1] - 140)/110
                    scale = (s_params[2]-1)*((n_passes-1)/3)+1
                    n_scale = (scale - 1)/5
                    labels.append((cl,[n_width,n_xpos,n_scale,norm_passes]))
                elif cl == 'hor':
                    height = s_params[0] + (s_params[0] * (s_params[2] - 1))*((start)/3)
                    n_height = (height - 50)/350
                    y_pos = s_params[1]
                    n_ypos = (s_params[1] - 140)/110
                    scale = (s_params[2]-1)*((n_passes-1)/3)+1
                    n_scale = (scale - 1)/5
                    labels.append((cl,[n_height,n_ypos,n_scale,norm_passes]))
                elif cl == 'tri':
                    n_width = (s_params[0] - 100)/300
                    n_height = (s_params[1] - 100)/300
                    n_xpos = (s_params[2] - 100)/150
                    n_ypos = (s_params[3] - 50)/200
                    start_pass = start/3
                    labels.append((cl,[n_width,n_height,n_xpos,n_ypos,start_pass,norm_passes]))

        return labels

    # Build first layer in the combined architecture
    def BuildFirstLayerComb(self,size,n_layers,n_neurons):

        # Data augmentation module
        data_augmentation = tf.keras.Sequential(
          [
            tf.keras.layers.RandomFlip("horizontal",input_shape=(size,size,1)),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
          ]
        )

        # Initiate CNN
        cnn = tf.keras.models.Sequential()

        # Augment data
        cnn.add(data_augmentation)

        # Rescale
        cnn.add(tf.keras.layers.Rescaling(1./255, input_shape=(size, size, 1)))

        # Apply convolutional layer
        cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[size, size, 1]))

        # Apply pooling
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        # Apply second convolutional layer
        cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        # Apply second convolutional layer
        #cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
        #cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        cnn.add(tf.keras.layers.Dropout(0.2))

        # Flatten the data
        cnn.add(tf.keras.layers.Flatten())

        for i in range(1,n_layers):
            # Apply full connection
            cnn.add(tf.keras.layers.Dense(units=n_neurons, activation='relu'))

        # Apply output layer
        cnn.add(tf.keras.layers.Dense(units=5, activation='softmax'))

        # Compile CNN
        cnn.compile(optimizer = 'adam',
                    loss = 'categorical_crossentropy',
                    metrics = ['accuracy'])

        return cnn

    # Build branch in second layer
    def build_branch(self,inputs,size,n_epochs,paramName,n_neurons,n_layers):

        x = inputs

        #x = tf.keras.layers.RandomFlip("horizontal",input_shape=(size,size,1))(x)
        #x = tf.keras.layers.RandomRotation(0.1)(x)
        #x = tf.keras.layers.RandomZoom(0.1)(x)

        # Rescale
        x = tf.keras.layers.Rescaling(1./255, input_shape=(size,size,1))(x)

        # Apply convolutional layer
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[size,size,1])(x)

        # Apply pooling
        x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(x)

        # Apply second convolutional layer
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')(x)
        x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(x)

        # Apply second convolutional layer
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')(x)
        x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(x)

        #x = tf.keras.layers.Dropout(0.2)(x)

        # Flatten the data
        x = tf.keras.layers.Flatten()(x)

        for i in range(1,n_layers):
            # Apply full connection
            x = tf.keras.layers.Dense(units=n_neurons, activation='relu')(x)

        # Apply output layer
        x = tf.keras.layers.Dense(units=1,activation='linear',name=paramName)(x)

        return x

    # Build full second layer
    def build(self,size, names, n_epochs,n_neurons,n_layers):
		# initialize the input shape and channel dimension (this code
		# assumes you are using TensorFlow which utilizes channels
		# last ordering)
        inputs = (size, size, 1)
        new_inputs = tf.keras.layers.Input(shape=inputs)
		#chanDim = -1
		# construct both the "category" and "color" sub-networks
		#inputs = Input(shape=inputShape)
        branch = []
        for paramName in names:
            branch.append(self.build_branch(new_inputs,size,n_epochs,paramName,n_neurons,n_layers))

		# create the model using our input (the batch of images) and
		# two separate outputs -- one for the clothing category
		# branch and another for the color branch, respectively
        model = Model(
        	inputs=new_inputs,
        	outputs=branch,
        	name="strat_predict")
        # return the constructed network architecture
        return model

    # Build full model
    def build_combined(self,classes, size, names, net_size_l1, net_size_l2, params, path):

        # Get labels
        labels = self.GetAllLabels(path,classes,params)

        # Get images
        images = []
        for cl in classes:
            images.extend(glob.glob(path + '/' + cl + '/*.jpg'))

        # Get images in correct format
        image_files = []
        for image in images:
            img = load_img(image,color_mode="grayscale")
            img = tf.keras.preprocessing.image.img_to_array(img)
            image_files.append(img)

        files_ds = zip(image_files, labels)
        files_ds = list(files_ds)

        # Divide into train, validation, test sets
        train_size = int(0.7 * len(files_ds))
        val_size = int(0.15 * len(files_ds))
        test_size = int(0.15 * len(files_ds))

        random.shuffle(files_ds)
        train_dataset = files_ds[0:train_size]
        val_dataset = files_ds[train_size:train_size+val_size]
        test_dataset = files_ds[train_size+val_size:]

        # Initialise training sets
        train_images_l1 = []
        train_labels_name = []
        train_labels_params = {
            'c2o':[],
            'over':[],
            'ver':[],
            'hor':[],
            'tri':[]
            }
        train_images_l2 = {
            'c2o':[],
            'over':[],
            'ver':[],
            'hor':[],
            'tri':[]
            }

        # Get images in correct arrays for each layer
        for image, label in train_dataset:
            train_images_l1.append(image)
            train_images_l2[label[0]].append(image)
            train_labels_params[label[0]].append(label[1])

            # One-hot encode categories
            one_hot = [0,0,0,0,0]
            i = classes.index(label[0])
            one_hot[i] = 1
            train_labels_name.append(one_hot)

        # np.array required
        train_labels_name = np.array(train_labels_name)
        train_images_l1 = np.array(train_images_l1)

        # Same for validation
        val_images_l1 = []
        val_labels_name = []
        val_labels_params = {
            'c2o':[],
            'over':[],
            'ver':[],
            'hor':[],
            'tri':[]
            }
        val_images_l2 = {
            'c2o':[],
            'over':[],
            'ver':[],
            'hor':[],
            'tri':[]
            }
        for image, label in val_dataset:
            val_images_l1.append(image)
            val_images_l2[label[0]].append(image)
            val_labels_params[label[0]].append(label[1])

            one_hot = [0,0,0,0,0]

            i = classes.index(label[0])
            one_hot[i] = 1
            val_labels_name.append(one_hot)

        val_labels_name = np.array(val_labels_name)
        val_images_l1 = np.array(val_images_l1)

        # Same for test
        test_images = []
        test_labels_param = []
        test_labels_name = []
        for image, label in test_dataset:
            test_images.append(image)
            test_labels_param.append(label[1])

            one_hot = [0,0,0,0,0]

            i = classes.index(label[0])
            one_hot[i] = 1
            test_labels_name.append(one_hot)

        # Build first layer
        model_1 = self.BuildFirstLayerComb(size,net_size_l1[0],net_size_l1[1])

        # Fit first layer
        l1_history = model_1.fit(x = train_images_l1, y = train_labels_name, validation_data = (val_images_l1,val_labels_name),epochs = net_size_l1[2])

        print('First layer built')

        # Train each CNN branch
        model_2 = {}
        l2_history = {}
        for cl in classes:
            # Pre-process data
            strat_train_images = np.array(train_images_l2[cl])
            strat_val_images = np.array(val_images_l2[cl])

            labels_arr = np.array(train_labels_params[cl])
            strat_train_labels = {}
            for i,name in enumerate(names[cl]):
                strat_train_labels[name] = labels_arr[:,i]

            labels_arr = np.array(val_labels_params[cl])
            strat_val_labels = {}
            for i,name in enumerate(names[cl]):
                strat_val_labels[name] = labels_arr[:,i]

            losses = {}
            for name in names[cl]:
                losses[name] = "mae"

            # Build and compile second layer
            model = self.build(size, names[cl], net_size_l2[cl][2], net_size_l2[cl][1], net_size_l2[cl][0])
            model.compile(optimizer='adam', loss=losses, metrics=["mae"])

            # Fit second layer
            H = model.fit(x=strat_train_images,
                y = strat_train_labels,
                validation_data=(strat_val_images,strat_val_labels),
                epochs=net_size_l2[cl][2])

            l2_history[cl] = H.history

            model_2[cl] = model

            print('Second layer, ' + cl + ' class built')

        return model_1,model_2,test_images,test_labels_name,test_labels_param,l1_history.history,l2_history

    # Make predictions using full model
    def predict_combined(self,classes, size, names, net_size_l1, net_size_l2, params, path):

        # Get fitted models for each layer
        model_1,model_2,test_images,test_labels_name,test_labels_param,l1_history,l2_history = self.build_combined(classes, size, names, net_size_l1, net_size_l2, params, path)

        # Save training histories for visualisation
        hist_df = pd.DataFrame(l1_history)
        hist_csv_file = 'C:/Users/adsk1/Documents/FYP/Python/data/plots/layer1_comb_hist.csv'
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

        # Save training histories for visualisation
        for cl in classes:
            hist_df = pd.DataFrame(l2_history[cl])
            hist_csv_file = 'C:/Users/adsk1/Documents/FYP/Python/data/plots/layer2_comb_hist_' + cl + '.csv'
            with open(hist_csv_file, mode='w') as f:
                hist_df.to_csv(f)

        test_images_l1 = np.array(test_images)

        # Predict first layer
        l1_results_prob = model_1.predict(test_images_l1)

        # Get value with highest probability
        l1_results = []
        for result in l1_results_prob:
            max_ind = np.argmax(result)
            l1_results.append(classes[max_ind])

        # Predict second layer given class predicted
        l2_results = []
        true_results = []
        for i,cl in enumerate(l1_results):

            # Get true value
            x = [test_labels_name[i]]
            max_ind = np.argmax(x)
            class_x = [classes[max_ind]]
            class_x.extend(self.ExchangeLabel(classes[max_ind],test_labels_param[i])[0])
            true_results.append(class_x)

            # Get predicted value
            a = [cl]
            s = np.reshape(test_images_l1[i],(-1,125,125,1))
            label_vals = self.ExchangeLabel(cl,model_2[cl].predict(s))
            label_vals = [i[0][0] for i in label_vals[0]]

            l2_results.append(a + label_vals)


        return l2_results,true_results
