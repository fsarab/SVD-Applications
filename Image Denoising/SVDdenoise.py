import numpy as np
import random
import cv2
import matplotlib.pyplot as plt


def noise(image,prob):

    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

img_bw = cv2.imread('img.jpg',0) 
print('Original Matrix Shape:',img_bw.shape)
print('Original Matrix Size:',img_bw.size)
M = img_bw.shape[0] 
N = img_bw.shape[1] 
img_matrix=np.matrix(img_bw)
print(img_matrix)

noise_img = noise(img_bw,0.05)
cv2.imwrite('noise.jpg', noise_img)
img_noise = cv2.imread('noise.jpg',0) 
print('Noise Matrix Shape:',img_noise.shape)
print('Noise Matrix Size:',img_noise.size)
R = img_noise.shape[0] 
D = img_noise.shape[1] 
noise_matrix=np.matrix(img_noise)
print(noise_matrix)



U, sigma, V = np.linalg.svd(noise_matrix)
n=range(10,400)
for k in (50,100,200,300,400):
    z= np.matrix(U[:, :k]) * np.diag(sigma[:k]) * np.matrix(V[:k, :])
    plt.imshow(z, cmap = 'gray')
    title = " Image after =  %s" %k
    plt.title(title)
    plt.show()

p=[]
for k in n:
    z = np.matrix(U[:, :k]) * np.diag(sigma[:k]) * np.matrix(V[:k, :])
    p.append(np.linalg.norm(img_matrix - z))

plt.scatter(n,p, color ="hotpink")
plt.show()

# U, sigma, V = np.linalg.svd(noise_matrix)

# z80= np.matrix(U[:, :80]) * np.diag(sigma[:80]) * np.matrix(V[:80, :])
# print('Matrix k=80 is:',z80)
# p80=np.linalg.norm(img_matrix - z80)
# print('Norm Ferobinious k=80 is:',p80)

