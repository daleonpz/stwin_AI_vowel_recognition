import numpy as np
import matplotlib.pyplot as plt

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
min_max = np.array([data_mixed.min(axis=0), data_mixed.max(axis=0)])

measurement = np.array([[245.000000, -203.000000, -397.000000 ,24780.000000 ,-94150.000000 ,104090.000000 ],
                    [546.000000, -108.000000, 950.000000 ,202720.000000 ,72170.000000 ,163870.000000 ]])

print(f'Testing function to get max and min of a measurement')
dist = np.linalg.norm(measurement.flatten() - min_max.flatten())
print(f'Euclidean distance between C and python: {dist}')


obs_norm_data = np.loadtxt('norm.csv')

ref_norm_data = np.zeros((400,6))

def normalize_columns_between_0_and_1(matrix):
    mmin = np.min(matrix, axis=0)
    mmax = np.max(matrix, axis=0)
    matrix = matrix - mmin
    matrix = matrix / (mmax - mmin)
    return matrix

ref_norm_data = normalize_columns_between_0_and_1(data_mixed)

print(f'Testing function to normalize data')
dist = np.linalg.norm(ref_norm_data.flatten() - obs_norm_data.flatten())
print(f'Euclidean distance between C and Python implementation: {dist}')
