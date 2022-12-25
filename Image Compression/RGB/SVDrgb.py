import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

##Convert Image Into Matrix
PATH='D:/STUDY/Computational Data Minig/CDM Projects/Project 4'

img_rgb=plt.imread(PATH+f'/RGB/img_rgb.jpg')


##Shapeand Size of Matrix 
print('Original Matrix Shape:',img_rgb.shape)
print('Original Matrix Size:',img_rgb.size)
M = img_rgb.shape[0] 
N = img_rgb.shape[1] 


for k in range(1,100,10):
    
  imgRed = np.zeros((M, N))
  imgGreen = np.zeros((M, N))
  imgBlue = np.zeros((M, N))

  for m in range(0, M):
    for n in range(0, N):
        imgRed[m][n] = img_rgb[m][n][0]
        imgGreen[m][n] = img_rgb[m][n][1]
        imgBlue[m][n] = img_rgb[m][n][2]
        
 ##Red Channel
  redV, redSigmaVals, redWAdjoint = np.linalg.svd(imgRed, full_matrices=True)

  redTruncV = redV[:, :k]
  #print(redTruncV.shape)
  #print(redTruncV.shape)
  redTruncSigmaVals = redSigmaVals[:k]
  redTruncSigma = np.diag(redTruncSigmaVals)
  redTruncWAdjoint = redWAdjoint[:k]

  redImg = redTruncV @ redTruncSigma @ redTruncWAdjoint


  ##Green Channel

  greenV, greenSigmaVals, greenWAdjoint = np.linalg.svd(
    imgGreen, full_matrices=True)

  greenTruncV = greenV[:, :k]
  greenTruncSigmaVals = greenSigmaVals[:k]
  greenTruncSigma = np.diag(greenTruncSigmaVals)
  greenTruncWAdjoint = greenWAdjoint[:k]

  greenImg = greenTruncV @ greenTruncSigma @ greenTruncWAdjoint


  ##Blue Channel

  blueV, blueSigmaVals, blueWAdjoint = np.linalg.svd(imgBlue, full_matrices=True)

  blueTruncV = blueV[:, :k]
  blueTruncSigmaVals = blueSigmaVals[:k]
  blueTruncSigma = np.diag(blueTruncSigmaVals)
  blueTruncWAdjoint = blueWAdjoint[:k]

  blueImg = blueTruncV @ blueTruncSigma @ blueTruncWAdjoint


  ##Construct Matrix
  constructImg = np.zeros((M, N, 3))

  for m in range(0, M):
    for n in range(0, N):
        constructImg[m][n][0] = int(redImg[m][n])
        constructImg[m][n][1] = int(greenImg[m][n])
        constructImg[m][n][2] = int(blueImg[m][n])


  scaledConstructImg = np.array(constructImg/np.amax(constructImg)*255, np.uint8)

# print(constructTruncImg)

  plt.imshow(scaledConstructImg)
  plt.title('SVD, k = {}'.format(k))
  plt.show()
 #plt.imsave('RGBCompressed.png', scaledConstructImg)


  originalSize = m * n * 3
  compressedSize = k * (1 + m + n) * 3

  print('original size:', originalSize)


  print('compressed size:', compressedSize)


  proportion = compressedSize * 1.0 / originalSize
  print('Proportioan is : ', proportion)

  print('k={} , Compressed image size is '.format(k) +
      str(round(proportion * 100, 2)) + '% of the original image ')
