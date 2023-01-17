import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('norm.csv')

acc = data[:, 0:3]
gyro = data[:, 3:6]

img = np.zeros((6,20,20))

for j in range(6):
        img[j, :, :] = data[:, j].reshape(20, 20)

print(img.shape) 
ndata_image = np.transpose(img, (1, 2, 0))
print(ndata_image.shape)

nacc = ndata_image[:,:,0:3]
ngyro = ndata_image[:,:,3:6]

# create a subplot row for each file
fig, axes = plt.subplots(1, 4)


axes[0].set_title('Acc')
axes[1].set_title('Acc Image')
axes[2].set_title('Gyro')
axes[3].set_title('Gyro Image')

# plot data
axes[0].plot(acc)
axes[1].imshow(nacc)
axes[2].plot(gyro)
axes[3].imshow(ngyro)
    
plt.show()



