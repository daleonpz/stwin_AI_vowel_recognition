import numpy as np
import matplotlib.pyplot as plt

def get_min_max(data):
    acc = data[:, 0:3]
    gyro = data[:, 3:6]

    flat_acc = acc.flatten()
    flat_gyro = gyro.flatten()

    mmin_acc = np.min(flat_acc)
    mmax_acc = np.max(flat_acc)
    mmin_gyro = np.min(flat_gyro)
    mmax_gyro = np.max(flat_gyro)

    return mmin_acc, mmax_acc, mmin_gyro, mmax_gyro

data_int = np.loadtxt('raw_int.csv')
data_float = np.loadtxt('raw_float.csv')

# convert data_int to float
data_int = data_int.astype(np.float)

# flatten the data
data_int = data_int.flatten()
data_float = data_float.flatten()

# euclidean distance
dist = np.linalg.norm(data_int - data_float)

print(f'Testing Conversion from int to float in CUBE-AI')
print(f'Euclidean distance between ai_float and tofloat(int): {dist}')

data_mixed = np.loadtxt('min_max.csv')

# find the min and max of the data per column
mmin_acc, mmax_acc, mmin_gyro, mmax_gyro = get_min_max(data_mixed)

min_max = np.array([mmin_acc, mmin_gyro, mmax_acc, mmax_gyro])

measurement = np.array([ -1002.000000,  -353010.000000,  1998.000000, 419090.000000])

print(f'Testing function to get max and min of a measurement')
dist = np.linalg.norm(measurement.flatten() - min_max.flatten())
print(f'Euclidean distance between C and python: {dist}')


obs_norm_data = np.loadtxt('norm.csv')

ref_norm_data = np.zeros((400,6))

def normalize_acc_gyro(data):
    acc = data[:, 0:3]
    gyro = data[:, 3:6]

    mmin_acc, mmax_acc, mmin_gyro, mmax_gyro = get_min_max(data)

    acc = acc - mmin_acc
    acc = acc / (mmax_acc - mmin_acc)

    gyro = gyro - mmin_gyro
    gyro = gyro / (mmax_gyro - mmin_gyro)

    data[:, 0:3] = acc
    data[:, 3:6] = gyro

    return data


ref_norm_data = normalize_acc_gyro(data_mixed)

print(f'Testing function to normalize data')
dist = np.linalg.norm(ref_norm_data.flatten() - obs_norm_data.flatten())
print(f'Euclidean distance between C and Python implementation: {dist}')
