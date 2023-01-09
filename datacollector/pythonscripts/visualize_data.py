import sys
import os
import numpy as np
import matplotlib.pyplot as plt


def normalize_columns_between_0_and_1(matrix):
    mmin = np.min(matrix, axis=0)
    mmax = np.max(matrix, axis=0)
    matrix = matrix - mmin
    matrix = matrix / (mmax - mmin)
    return matrix



def main(argv):
    # read data from argument
    path = sys.argv[1]
    files = os.listdir(path)
    
    img_size = 20
    number_of_files = 2
    # select 15 files randomly
    files = np.random.choice(files, number_of_files, replace=False)

    # create a subplot row for each file
    fig, axes = plt.subplots(number_of_files, 4)

    fig.suptitle(f'{path}', fontsize=16)
    for i, file in enumerate(files):
        # read data
        data = np.loadtxt(path + file, delimiter=',')
        acc = data[:, 0:3]
        gyro = data[:, 3:6]

        # normalize data and reshape
        nacc = normalize_columns_between_0_and_1(acc)
        ngyro = normalize_columns_between_0_and_1(gyro)

        nacc = nacc.reshape(img_size, img_size, 3)
        ngyro = ngyro.reshape(img_size, img_size, 3)

        # plot data
        axes[i][0].plot(acc[:, 0])
        axes[i][1].plot(acc[:, 1])
        axes[i][2].plot(acc[:, 2])
        axes[i][3].imshow(nacc)

    plt.show()

if __name__ == '__main__':
    main(sys.argv[1:])
