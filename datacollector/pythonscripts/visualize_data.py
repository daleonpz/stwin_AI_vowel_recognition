import argparse
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


def normalize_acc_gyro(data):
    acc = data[:, 0:3]
    gyro = data[:, 3:6]

    flat_acc = acc.flatten()
    flat_gyro = gyro.flatten()

    mmin_acc = np.min(flat_acc)
    mmax_acc = np.max(flat_acc)
    mmin_gyro = np.min(flat_gyro)
    mmax_gyro = np.max(flat_gyro)

    acc = acc - mmin_acc
    acc = acc / (mmax_acc - mmin_acc)

    gyro = gyro - mmin_gyro
    gyro = gyro / (mmax_gyro - mmin_gyro)

    data[:, 0:3] = acc
    data[:, 3:6] = gyro

    return data


def main(args):
    # read data from argument
    path = args.path
    number_of_files = args.number_of_files
    one_file = args.one_file

    img_size = 20

    if one_file:
        number_of_files = 2
        files = [path]

    else:
        # select 15 files randomly
        files = os.listdir(path)
        files = np.random.choice(files, number_of_files, replace=False)

    # create a subplot row for each file
    fig, axes = plt.subplots(number_of_files, 5)

    fig.suptitle(f'{path}', fontsize=16)
    axes[0, 0].set_title('Accelerometer original')
    axes[0, 1].set_title('Accelerometer normalized')
    axes[0, 2].set_title('Accelerometer image')
    axes[0, 3].set_title('Gyro normalized')
    axes[0, 4].set_title('Gyro Image')
    
    for i, file in enumerate(files):
        # read data
        print(f'Reading file {file}')
        if one_file:
            data = np.loadtxt(path, delimiter=',')
        else:
            data = np.loadtxt(os.path.join(path, file), delimiter=',')

        acc = data[:, 0:3]
        gyro = data[:, 3:6]

        axes[i][0].plot(acc)
        ndata = normalize_acc_gyro(data)
        
        # 3-rd dimension are [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        ndata_image = np.zeros((6, img_size, img_size))
        for j in range(6):
            ndata_image[j, :, :] = ndata[:, j].reshape(img_size, img_size)

        print(ndata_image.shape) 
        print(ndata_image[:,0,0])
        print(ndata[0,:])
        ndata_image = np.transpose(ndata_image, (1, 2, 0))
        print(ndata_image.shape)

        nacc = ndata_image[:,:,0:3]
        ngyro = ndata_image[:,:,3:6]

        # plot data
        axes[i][1].plot(ndata[:, 0:3])
        axes[i][2].imshow(nacc)
        axes[i][3].plot(ndata[:, 3:6])
        axes[i][4].imshow(ngyro)
    
    plt.show()

if __name__ == '__main__':
    # parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--path', help='Path to the data')
    argparser.add_argument('--number_of_files', help='Number of files to visualize', default=5, type=int)
    argparser.add_argument('--one_file', help='Visualize only one file', default=False, type=bool)
    args = argparser.parse_args()

    main(args)

