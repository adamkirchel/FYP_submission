import csv
from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

class DensityEstimate():

    def __init__(self,name,num):

        arr = []
        with open(name, 'r') as fd:
            reader = csv.reader(fd)
            for row in reader:
                arr.append(row)

        density = np.array(arr).astype(np.float)

        # Extract x and y
        x = density[:, 0]
        y = density[:, 1]

        # Define the borders
        xmin = 0
        xmax = 1
        ymin = 0
        ymax = 1

        # Create meshgrid
        self.xx, self.yy = np.mgrid[xmin:xmax:complex(0,num), ymin:ymax:complex(0,num)]

        #print(self.xx)

        # Create kernel
        positions = np.vstack([self.xx.ravel(), self.yy.ravel()])
        values = np.vstack([x, y])
        kernel = st.gaussian_kde(values)
        self.density = np.reshape(kernel(positions).T, self.xx.shape)

    def get_distribution(self):

        return self.density

    def plot_distribution(self):

        fig = plt.figure(figsize=(8, 8))
        ax = plt.axes(projection='3d')
        surf = ax.plot_surface(self.xx, self.yy, self.density, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('PDF')
        #ax.set_title('Surface plot of Gaussian 2D KDE')
        fig.colorbar(surf, shrink=0.5, aspect=5) # add color bar indicating the PDF
        ax.view_init(60, 35)
        plt.show()

    def save_as_img(self,path,name):
        data = im.fromarray(self.density)

        data.save(path + '/' + name + '.png')

DensityEstimate('C:/Users/adsk1/Documents/FYP/Python/data_old/density/raw/c2o_1_tracking_dist_4.csv',100).plot_distribution()
