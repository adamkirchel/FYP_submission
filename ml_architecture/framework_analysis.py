from ml_arch import Architecture
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
import pandas as pd
import glob
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, array_to_img, img_to_array
from itertools import zip_longest
import csv

# First layer

# Accuracy and time vs. resolution study
def ResComp1():
    paths = [
            'C:/Users/adsk1/Documents/FYP/Python/data/geometry/raw/images',
            'C:/Users/adsk1/Documents/FYP/Python/data/geometry/curvature/gaussian/images',
            'C:/Users/adsk1/Documents/FYP/Python/data/geometry/curvature/mean/images',
            'C:/Users/adsk1/Documents/FYP/Python/data/geometry/curvature/p1/images',
            'C:/Users/adsk1/Documents/FYP/Python/data/geometry/curvature/p2/images'
        ]

    sizes = [10,20,40,60,80,100,125]
    scores = []
    t_vec = []
    for path in paths:
        score = []
        t = []
        name = path.split('/')[-2]
        print(name)
        for size in sizes:
            start = time.time()
            model,test_data,history = Architecture().BuildFirstLayer(path,size,1,128,100)
            end = time.time()
            total = end - start

            sc = model.evaluate(test_data)

            score.append(sc[1])
            t.append(total)
            print('Size: ' + str(size) + ' Score: ' + str(sc[1]))
            hist_df = pd.DataFrame(history.history)
            hist_csv_file = "C:/Users/adsk1/Documents/FYP/Python/data/plots/layer1_res_acc_comp_" + str(size) + "_" + name + ".csv"
            with open(hist_csv_file, mode='w') as f:
                hist_df.to_csv(f)

        scores.append(score)
        t_vec.append(t)

    scores = np.array(scores)
    np.savetxt("C:/Users/adsk1/Documents/FYP/Python/data/plots/layer1_res_acc_comp.csv", scores, delimiter=",")

    t_vec = np.array(t_vec)
    np.savetxt("C:/Users/adsk1/Documents/FYP/Python/data/plots/layer1_res_time_comp.csv", t_vec, delimiter=",")

# Data augmentation study - assessment of overfitting - include change vs. no-change
def DataAugment1():
    paths1 = [
        'C:/Users/adsk1/Documents/FYP/Python/data/geometry/',
        'C:/Users/adsk1/Documents/FYP/Python/data/geometry_no_1/',
        'C:/Users/adsk1/Documents/FYP/Python/data/geometry_no_2/',
        'C:/Users/adsk1/Documents/FYP/Python/data/geometry_no_3/'
    ]
    paths2 = [
            'raw/images',
            'curvature/gaussian/images',
            'curvature/mean/images',
            'curvature/p1/images',
            'curvature/p2/images'
        ]

    scores = []
    for path1 in paths1:
        score = []
        for path2 in paths2:
            name1 = path1.split('/')[-2]
            name2 = path2.split('/')[-2]
            path = path1 + path2
            model,test_data,history = Architecture().BuildFirstLayer(path,125,1,128,100)
            sc = model.evaluate(test_data)


            score.append(sc[1])
            print('Score: ' + str(sc[1]))
            hist_df = pd.DataFrame(history.history)
            hist_csv_file = "C:/Users/adsk1/Documents/FYP/Python/data/plots/layer1_aug1_" + name1 + "_" + name2 + ".csv"
            with open(hist_csv_file, mode='w') as f:
                hist_df.to_csv(f)

        scores.append(score)

    scores = np.array(scores)
    np.savetxt("C:/Users/adsk1/Documents/FYP/Python/data/plots/layer1_aug1.csv", scores, delimiter=",")

# Got optimum data

# Tune parameters

# overfitting
def PlotOverfit1():
    num_hidden = [[1,16],[2,16],[3,64],[4,512]]
    names = ['One layer','Two layers','Three layers','Four layers']
    path = 'C:/Users/adsk1/Documents/FYP/Python/data/geometry/curvature/p1/images'
    plotter = tfdocs.plots.HistoryPlotter(metric = 'categorical_crossentropy', smoothing_std=10)
    size_histories = {}

    for i,(layers,neurons) in enumerate(num_hidden):
        print(i,layers,neurons)
        model,test_data,history = Architecture().BuildFirstLayer(path,125,layers,neurons,10000)
        size_histories[names[i]] = history
        hist_df = pd.DataFrame(history.history)
        hist_csv_file = 'C:/Users/adsk1/Documents/FYP/Python/data/plots/overfit_history' + str(i) + '.csv'
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

    plotter.plot(size_histories)
    a = plt.xscale('log')
    #plt.xlim([5, max(plt.xlim())])
    #plt.ylim([0.5, 0.7])
    plt.xlabel("Epochs [Log Scale]")
    plt.show()

    plotter.plot(size_histories)
    #plt.xlim([5, max(plt.xlim())])
    #plt.ylim([0.5, 0.7])
    plt.xlabel("Epochs")
    plt.show()

# Second layer

# Resolution compare - not required

# Data augmentation study - assessment of overfitting - include change vs. no-change
def DataAugment2(names,cl,n_epochs):
    paths1 = [
        'C:/Users/adsk1/Documents/FYP/Python/data/geometry/',
        'C:/Users/adsk1/Documents/FYP/Python/data/geometry_no_1/',
        'C:/Users/adsk1/Documents/FYP/Python/data/geometry_no_2/',
        'C:/Users/adsk1/Documents/FYP/Python/data/geometry_no_3/'
    ]
    paths2 = [
            'raw/images',
            'curvature/gaussian/images',
            'curvature/mean/images',
            'curvature/p1/images',
            'curvature/p2/images'
        ]

    #scores = []
    for i,path1 in enumerate(paths1):
        score = []
        for path2 in paths2:
            path = path1 + path2
            labels = Architecture().GetLabels(
                path,
                cl,
                classes)

            labels = np.array(labels)

            images = glob.glob(path + '/' + cl + '/*.jpg')

            image_files = []
            for image in images:
                img = load_img(image,color_mode="grayscale")
                img = tf.keras.preprocessing.image.img_to_array(img)
                image_files.append(img)

            files_ds = tf.data.Dataset.from_tensor_slices((image_files, labels))
            files_ds = files_ds.batch(1)

            train_size = int(0.8 * len(files_ds))
            val_size = int(0.2 * len(files_ds))

            files_ds = files_ds.shuffle(len(files_ds)+1)
            train_dataset = files_ds.take(train_size)
            val_dataset = files_ds.skip(train_size)

            model = Architecture().build(125, names, n_epochs)

            losses = {
            	names[0]: "mae",
            	names[1]: "mae",
                names[2]: "mae",
                names[3]: "mae",
                names[4]: "mae",
                names[5]: "mae"
            }

            model.compile(optimizer='adam', loss=losses, metrics=["mae"])

            train_images = []
            train_labels = []
            for image, label in train_dataset.take(len(train_dataset)):
                train_images.append(image[0])
                label_arr = []
                for j in label:
                    label_arr.append(j)
                train_labels.append(label_arr[0])

            train_labels = np.array(train_labels)
            train_images = np.array(train_images)
            #print(train_images)
            #print(train_labels)
            #print(train_images.shape)
            #print(train_labels.shape)

            val_images = []
            val_labels = []
            for image, label in val_dataset.take(len(val_dataset)):
                val_images.append(image[0])
                label_arr = []
                for j in label:
                    label_arr.append(j)
                val_labels.append(label_arr[0])

            val_labels = np.array(val_labels)
            val_images = np.array(val_images)

            H = model.fit(x=train_images,
            	y={
                    names[0]: train_labels[:,0],
                    names[1]: train_labels[:,1],
                    names[2]: train_labels[:,2],
                    names[3]: train_labels[:,3],
                    names[4]: train_labels[:,4],
                    names[5]: train_labels[:,5]
                },
            	validation_data=(val_images,
            		{
                        names[0]: val_labels[:,0],
                        names[1]: val_labels[:,1],
                        names[2]: val_labels[:,2],
                        names[3]: val_labels[:,3],
                        names[4]: val_labels[:,4],
                        names[5]: val_labels[:,5]
                    }),
            	epochs=n_epochs)
            sc = model.evaluate(x=val_images,
            	y={
                    names[0]: val_labels[:,0],
                    names[1]: val_labels[:,1],
                    names[2]: val_labels[:,2],
                    names[3]: val_labels[:,3],
                    names[4]: val_labels[:,4],
                    names[5]: val_labels[:,5]
                })

            score.append(sc)
        score = np.array(score)
        np.savetxt("C:/Users/adsk1/Documents/FYP/Python/data/plots/layer2_aug_" + cl + "_" + str(i) + ".csv", score, delimiter=",")
        #scores.append(score)

    #scores = np.array(scores)


# Got optimum data

# Tune parameters

# overfitting for all strategies for 4 different sizes
def PlotOverfit2(names,cl,classes):
    num_hidden = [[1,16],[2,16],[3,64],[4,512]]
    layer_names = ['One layer','Two layers','Three layers','Four layers']
    path = 'C:/Users/adsk1/Documents/FYP/Python/data/geometry/curvature/p1/images'
    plotter = tfdocs.plots.HistoryPlotter(metric = 'mae', smoothing_std=10)
    size_histories = {}

    for i,(layers,neurons) in enumerate(num_hidden):
        print(i,layers,neurons)

        labels = Architecture().GetLabels(
            path,
            cl,
            classes)

        labels = np.array(labels)

        #print(labels)

        images = glob.glob(path + '/' + cl + '/*.jpg')

        image_files = []
        for image in images:
            img = load_img(image,color_mode="grayscale")
            img = tf.keras.preprocessing.image.img_to_array(img)
            image_files.append(img)

        files_ds = tf.data.Dataset.from_tensor_slices((image_files, labels))
        files_ds = files_ds.batch(1)

        train_size = int(0.8 * len(files_ds))
        val_size = int(0.2 * len(files_ds))

        files_ds = files_ds.shuffle(len(files_ds)+1)
        train_dataset = files_ds.take(train_size)
        val_dataset = files_ds.skip(train_size)

        model = Architecture().build(125, names, 10000,neurons,layers)

        # do different sizes
        losses = {
            names[0]: "mae",
            names[1]: "mae",
            names[2]: "mae",
            names[3]: "mae",
            names[4]: "mae",
            names[5]: "mae"
        }

        model.compile(optimizer='adam', loss=losses, metrics=["mae"])

        train_images = []
        train_labels = []
        for image, label in train_dataset.take(len(train_dataset)):
            train_images.append(image[0])
            label_arr = []
            for j in label:
                label_arr.append(j)
            train_labels.append(label_arr[0])

        train_labels = np.array(train_labels)
        train_images = np.array(train_images)
        #print(train_images)
        #print(train_labels)
        #print(train_images.shape)
        #print(train_labels.shape)

        val_images = []
        val_labels = []
        for image, label in val_dataset.take(len(val_dataset)):
            val_images.append(image[0])
            label_arr = []
            for j in label:
                label_arr.append(j)
            val_labels.append(label_arr[0])

        val_labels = np.array(val_labels)
        val_images = np.array(val_images)

        # Different sizes
        H = model.fit(x=train_images,
            y={
                names[0]: train_labels[:,0],
                names[1]: train_labels[:,1],
                names[2]: train_labels[:,2],
                names[3]: train_labels[:,3],
                names[4]: train_labels[:,4],
                names[5]: train_labels[:,5]
            },
            validation_data=(val_images,
                {
                    names[0]: val_labels[:,0],
                    names[1]: val_labels[:,1],
                    names[2]: val_labels[:,2],
                    names[3]: val_labels[:,3],
                    names[4]: val_labels[:,4],
                    names[5]: val_labels[:,5]
                }),
            epochs=10000,callbacks=Architecture().get_callbacks())

        size_histories[layer_names[i]] = H
        hist_df = pd.DataFrame(H.history)
        hist_csv_file = 'C:/Users/adsk1/Documents/FYP/Python/data/plots/overfit_history_layer2_' + str(i) + '_' + cl + '.csv'
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

    plotter.plot(size_histories)
    a = plt.xscale('log')
    #plt.xlim([5, max(plt.xlim())])
    #plt.ylim([0.5, 0.7])
    plt.xlabel("Epochs [Log Scale]")
    plt.show()

    plotter.plot(size_histories)
    #plt.xlim([5, max(plt.xlim())])
    #plt.ylim([0.5, 0.7])
    plt.xlabel("Epochs")
    plt.show()

def practice(names,cl,classes):

    path = 'C:/Users/adsk1/Documents/FYP/Python/data/geometry/curvature/p1/images'
    #plotter = tfdocs.plots.HistoryPlotter(metric = 'mae', smoothing_std=10)
    #size_histories = {}

    #for i,(layers,neurons) in enumerate(num_hidden):
        #print(i,layers,neurons)

    labels = Architecture().GetLabels(
        path,
        cl,
        classes)

    labels = np.array(labels)

    #print(labels)

    images = glob.glob(path + '/' + cl + '/*.jpg')

    image_files = []
    for image in images:
        img = load_img(image,color_mode="grayscale")
        img = tf.keras.preprocessing.image.img_to_array(img)
        image_files.append(img)

    files_ds = tf.data.Dataset.from_tensor_slices((image_files, labels))
    files_ds = files_ds.batch(1)

    train_size = int(0.7 * len(files_ds))
    val_size = int(0.15 * len(files_ds))
    test_size = int(0.15 * len(files_ds))

    files_ds = files_ds.shuffle(len(files_ds)+1)
    train_dataset = files_ds.take(train_size)
    val_dataset = files_ds.skip(train_size)
    test_dataset = val_dataset.take(test_size)
    val_dataset = val_dataset.skip(test_size)

    model = Architecture().build(125, names, 10,32,1)

    # do different sizes
    losses = {
        names[0]: "mae",
        names[1]: "mae",
        names[2]: "mae",
        names[3]: "mae",
        names[4]: "mae",
        names[5]: "mae"
    }

    model.compile(optimizer='adam', loss=losses, metrics=["mae"])

    train_images = []
    train_labels = []
    for image, label in train_dataset.take(len(train_dataset)):
        train_images.append(image[0])
        label_arr = []
        for j in label:
            label_arr.append(j)
        train_labels.append(label_arr[0])

    train_labels = np.array(train_labels)
    train_images = np.array(train_images)
    #print(train_images)
    #print(train_labels)
    #print(train_images.shape)
    #print(train_labels.shape)

    val_images = []
    val_labels = []
    for image, label in val_dataset.take(len(val_dataset)):
        val_images.append(image[0])
        label_arr = []
        for j in label:
            label_arr.append(j)
        val_labels.append(label_arr[0])

    val_labels = np.array(val_labels)
    val_images = np.array(val_images)

    test_images = []
    test_labels = []
    for image, label in test_dataset.take(len(test_dataset)):
        test_images.append(image[0])
        label_arr = []
        for j in label:
            label_arr.append(j)
        test_labels.append(label_arr[0])

    test_labels = np.array(test_labels)
    test_images = np.array(test_images)
    print(test_images.shape)

    # Different sizes
    model.fit(x=train_images,
        y={
            names[0]: train_labels[:,0],
            names[1]: train_labels[:,1],
            names[2]: train_labels[:,2],
            names[3]: train_labels[:,3],
            names[4]: train_labels[:,4],
            names[5]: train_labels[:,5]
        },
        validation_data=(val_images,
            {
                names[0]: val_labels[:,0],
                names[1]: val_labels[:,1],
                names[2]: val_labels[:,2],
                names[3]: val_labels[:,3],
                names[4]: val_labels[:,4],
                names[5]: val_labels[:,5]
            }),
        epochs=10)

    x = model.predict(test_images)

    return x

params = {
    'c2o':[
        [120,120,250,250,3.333],
        [50,50,140,140,3.6],
        [50,100,150,250,4],
        [100,50,250,150,4],
        [60,50,210,210,5]
    ],
    'hor':[
        [120,250,3.333],
        [50,140,3.6],
        [60,230,6]
    ],
    'over':[
        [400,400,250,250],
        [180,180,140,140],
        [100,400,100,250],
        [400,100,250,100]
    ],
    'tri':[
        [400,400,250,50],
        [180,180,140,250],
        [100,400,100,50],
        [400,100,250,250]
    ],
    'ver':[
        [120,250,3.333],
        [50,140,3.6],
        [60,230,6]
    ]
}

n_epochs = 100
#names = ['width','height','centre_x','centre_y','start_pass','pass_n']
classes = ['c2o','over','ver','hor','tri']
names = {
    'c2o':[
        'min_width',
        'min_height',
        'centre_x',
        'centre_y',
        'scale',
        'pass_n'
    ],
    'hor':[
        'min_height',
        'centre_y',
        'scale',
        'pass_n'
    ],
    'over':[
        'width',
        'height',
        'centre_x',
        'centre_y',
        'pass_n'
    ],
    'tri':[
        'width',
        'height',
        'centre_x',
        'centre_y',
        'start_pass',
        'pass_n'
    ],
    'ver':[
        'min_width',
        'centre_x',
        'scale',
        'pass_n'
    ]
}

path = 'C:/Users/adsk1/Documents/FYP/Python/data/geometry/curvature/p1/images'

net_size_l1 = [1,16,130]
net_size_l2 = {
    'c2o':[3,64,100],
    'hor':[3,64,160],
    'over':[2,16,100],
    'tri':[4,512,100],
    'ver':[2,16,100]
}

size = 125
#DataAugment2(names,cl,n_epochs)
#PlotOverfit2(names,cl,classes)

#x = practice(names['c2o'],'c2o',params)
#print(x)
#time.sleep(100)

test_result,true_result = Architecture().predict_combined(classes, size, names, net_size_l1, net_size_l2, params, path)

with open('C:/Users/adsk1/Documents/FYP/Python/data/plots/comb_test_results.csv', 'w') as f:
    write = csv.writer(f)
    write.writerows(zip_longest(*test_result, fillvalue=''))

with open('C:/Users/adsk1/Documents/FYP/Python/data/plots/comb_true_results.csv', 'w') as f:
    write = csv.writer(f)
    write.writerows(zip_longest(*true_result, fillvalue=''))

#plt.plot(H.history['min_width_mae'],label='min_width_mae')
#plt.plot(H.history['val_min_width_mae'],label='val_min_width_mae')
#plt.plot(H.history['min_height_mae'],label='min_height_mae')
#plt.plot(H.history['val_min_height_mae'],label='val_min_height_mae')
#plt.plot(H.history['centre_x_mae'],label='centre_x_mae')
#plt.plot(H.history['val_centre_x_mae'],label='val_centre_x_mae')
#plt.plot(H.history['centre_y_mae'],label='centre_y_mae')
#plt.plot(H.history['val_centre_y_mae'],label='val_centre_y_mae')
#plt.plot(H.history['scale_mae'],label='scale_mae')
#plt.plot(H.history['val_scale_mae'],label='val_scale_mae')
#plt.plot(H.history['pass_n_mae'],label='pass_n_mae')
#plt.plot(H.history['val_pass_n_mae'],label='val_pass_n_mae')
#plt.xlabel('Epoch')
#plt.ylabel('Mean absolute error')
#plt.legend(loc='lower right')
#plt.show()


#model,test_data,history = Architecture().BuildSecondLayer(train_dataset,val_dataset,125,1,100)

#predictions = model.predict(test_data)

#for (img,label) in test_data:
#    print((label[0] * 350) + 50)
#
#print((predictions * 350) + 50)

#ResComp1()
#DataAugment1()
#PlotOverfit()
