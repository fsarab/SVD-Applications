import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

##Convert Image Into Matrix
PATH='D:/STUDY/Computational Data Minig/CDM Projects/Project 4'

img_bw=plt.imread(PATH+f'/bLACK&White/img_bw.jpg')


##Shapeand Size of Matrix 
print('Original Matrix Shape:',img_bw.shape)
print('Original Matrix Size:',img_bw.size)
M = img_bw.shape[0] 
N = img_bw.shape[1] 




#imgbw = np.zeros((M,N))

# imgbw = np.array(list(img_bw.getdata(band=0)), float)
# imgbw.shape = (img_bw.size[1], img_bw.size[0])
imgbw= np.matrix(img_bw)



U, sigma, V = np.linalg.svd(imgbw)
for k in range(1,100,10):

  reconstimg = np.matrix(U[:, :k]) * np.diag(sigma[:k]) * np.matrix(V[:k, :])
  plt.imshow(reconstimg, cmap='gray')
  plt.title('SVD, k = {}'.format(k))
  plt.show()
  #plt.imsave('BWCompressed.png', reconstimg, cmap='gray')



  originalSize = M * N * 3
  compressedSize = k * (1 + M + N) * 3

  print('original size:',originalSize)
  print('compressed size:',compressedSize)
  proportion= compressedSize * 1.0 / originalSize
  print('Proportioan is : ',proportion)

  print('k={} , Compressed image size is '.format(k) + str(round(proportion * 100, 2)) + '% of the original image ')
