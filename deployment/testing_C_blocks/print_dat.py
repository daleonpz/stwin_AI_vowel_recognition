import numpy as np
import matplotlib.pyplot as plt

def normalize_columns_between_0_and_1(matrix):
    print(matrix.shape)
    mmin = np.min(matrix, axis=0)
    mmax = np.max(matrix, axis=0)
    matrix = matrix - mmin
    matrix = matrix / (mmax - mmin)
    return matrix


data = np.loadtxt('raw_lecture.csv')

data = normalize_columns_between_0_and_1(data)

acc = data[:, 0:3]
gyro = data[:, 3:6]

img = np.zeros((6,20,20))

for j in range(6):
        img[j, :, :] = data[:, j].reshape(20, 20)

print(img.shape) 
print(img[:,0,0])
ndata_image = np.transpose(img, (1, 2, 0))
print(ndata_image.shape)

nacc = ndata_image[:,:,0:3]
ngyro = ndata_image[:,:,3:6]

# create a subplot row for each file
fig, axes = plt.subplots(1, 4)


axes[0].set_title('Axis X')
axes[1].set_title('Axis Y')
axes[2].set_title('Axis Z')
axes[3].set_title('Created Image')


# plot data
axes[0].plot(acc[:, 0])
axes[1].plot(acc[:, 1])
axes[2].plot(acc[:, 2])
axes[3].imshow(nacc)
    
plt.show()



