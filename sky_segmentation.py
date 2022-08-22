import cv2
import numpy as np
import os
import random
import shutil
import time

# To check if the input image has a bright or dark sky
def brightOrDark(inpImage):
    blur = cv2.medianBlur(inpImage, 255)
    is_bright = np.mean(blur) > 90
    if (is_bright):
        return True
    else:
        return False

# to convert the input image into a gradient image based on detected edges using sobel operator on input image
def returnGradientImg(inpImage):
    grayscaleImg = cv2.cvtColor(inpImage, cv2.COLOR_BGR2GRAY) # convert BGR image into grayscale
    edge_x_horizontal = cv2.Sobel(grayscaleImg, cv2.CV_64F, 1, 0) # compute and find the horizontal edges in input image using sobel operator
    edge_y_vertical = cv2.Sobel(grayscaleImg, cv2.CV_64F, 0, 1) # compute and find the vertical edges in input image using sobel operator
    gradientImg = np.hypot(edge_x_horizontal, edge_y_vertical) # combine horizontal and vertical edges detected using sobel to form gradient image
    return gradientImg

# to compute the optimal sky border position based on the previous computed sky border position by returnBorderPosition
def borderOptimisation(inpImage, gradientImg):
    minThresh=5
    maxThresh=600
    searchStep=5
    
    samplePoints = ((maxThresh - minThresh) // searchStep) + 1 # number of sample points in the search space
    
    #initialise optimal border list
    borderOptimal = None
    # initialise maximum energy function value
    energyMax = 0

    for c in range(1, samplePoints + 1):
        #threshold used to compute optimal sky border position function based on function/equation (6)
        thresh = minThresh + ((maxThresh - minThresh) // samplePoints - 1) * (c - 1)
        
        # obtains values of signifing the sky-land border points
        bordertmp = returnBorderPosition(gradientImg, thresh)
        
        # checking energy difference at that border point, to determine if its the optimal point in the border line
        # if it is more than the current energy maximum, the border point is recognised and assigned as the optimal border point and the energy maximum is updated
        energyVal = energyOptimisation(bordertmp, inpImage)

        if energyVal > energyMax:
            energyMax = energyVal
            borderOptimal = bordertmp
            
    return borderOptimal

# to find or output the sky border position
def returnBorderPosition(gradientImg, thresh):
    #creating the sky array with size of input images length and filling it with an intensity value of the size of the width
    skyBorder = np.full(gradientImg.shape[1], gradientImg.shape[0])

    for x in range(gradientImg.shape[1]):
        #np.argmax returns the index with the maximum intensity value on the vertical axis, highlighting the brightest point, most likely part of the boundary line
        borderPosition = np.argmax(gradientImg[:, x] > thresh)
        
        # if more than 0 it would be considered part of the sky or horizon line
        if borderPosition > 0:
            skyBorder[x] = borderPosition
            
    return skyBorder

# to find or output the energy optimisation
def energyOptimisation(bordertmp, inpImage):
    #binary mask based on estimated border values
    skyMask = createMask(bordertmp, inpImage)
    
    #determine ground and sky section in the mask created, returned as a 1-D array
    groundRegion = np.ma.array(inpImage, mask=cv2.cvtColor(cv2.bitwise_not(skyMask), cv2.COLOR_GRAY2BGR)).compressed()
    groundRegion.shape = (groundRegion.size//3, 3)
    skyRegion = np.ma.array(inpImage, mask=cv2.cvtColor(skyMask, cv2.COLOR_GRAY2BGR)).compressed()
    skyRegion.shape = (skyRegion.size//3, 3)
    
    # calculating covariance matrix of ground and sky regions of image, returning covariance value and average RGB values of the two regions
    covarGround, averageGround = cv2.calcCovarMatrix(groundRegion, None, cv2.COVAR_NORMAL | cv2.COVAR_ROWS | cv2.COVAR_SCALE)
    covarSky, averageSky = cv2.calcCovarMatrix(skyRegion, None, cv2.COVAR_NORMAL | cv2.COVAR_ROWS | cv2.COVAR_SCALE)
    
    #energy optimisation function Jn, function/equation (1)
    gamma = 2
    energyVal= 1 / (
        (gamma * np.linalg.det(covarSky) + np.linalg.det(covarGround)) +
        (gamma * np.linalg.det(np.linalg.eig(covarSky)[1]) +
            np.linalg.det(np.linalg.eig(covarGround)[1])))
    
    return energyVal
    
# to create binary mask, based on given input image and list of borderPoints
def createMask(borderPoint, inpImage):
    #creates mask the size (length x width) of the input image
    mask = np.zeros((inpImage.shape[0], inpImage.shape[1], 1), dtype=np.uint8)
    
    # checks the x and y coordinate of the border point, and assigns intensity value 255 to pixels below the border point.
    # based on the border points, initially, ground pixels are assigned with intensity value of 255 
    for x, y in enumerate(borderPoint):
        mask[y:, x] = 255
    
    # mask is inverted so that sky pixels are assigned with intensity value of 255, region of interest
    mask = cv2.bitwise_not(mask)
    return mask

#evaluation metric, calculate the intersection over union value between two input binary masks
def calculateIOU(mask1, mask2):
    
    #area or number of non-zero value pixels in the mask
    mask1Area = np.count_nonzero(mask1)
    mask2Area = np.count_nonzero(mask2)
    
    #calculating area or number of non-zero value pixels in an intersection between the 2 masks
    intersection = np.count_nonzero(np.logical_and(mask1, mask2))
    
    #calculating area or number of non-zero value pixels in a union of the 2 masks
    union = mask1Area + mask2Area - intersection
    iou = intersection/union
    return iou

#post processing to improve segmentation result of masks containing gaps within non-sky region
def postprocessing(inpMask):
    kernel = np.ones((20,20),np.uint8) * 255
    inv_inpMask = cv2.bitwise_not(inpMask)
    inv_inpMask = cv2.morphologyEx(inv_inpMask, cv2.MORPH_CLOSE, kernel)
    inpMask_1 = cv2.bitwise_not(inv_inpMask)
    return inpMask_1

#Create Results folder for storing processed and outputted binary masks with sky as region of interest
if os.path.exists("Results"):
    print("Results file exists and is going to be removed.")
    shutil.rmtree("Results")
else:
    print("Results file does not already exist.")
try:
    os.makedirs("Results", exist_ok = True)
    print("SUCCESS: The 'Results' directory was created successfully.")
    
except OSError as error:
    print("ERROR: The 'Results' directory was NOT CREATED successfully.")
print()
    
averageAccuracies = []

# to change the sample size of images per folder that are processed (choose lesser for lesser overall computation time, try going more than 8 if you experience errors
# accepts user input for sample size. Error handline performed to ensure correct input
sampleSize = int(input("Enter the number of images to sample from each folder/dataset (preferrably more than 8 for improved results)(-1 for all images): "))
while(True):
    if(sampleSize not in range (1, 725)) and (sampleSize != -1):
        sampleSize = int(input("Error! Enter the number of images to sample from each folder/dataset (preferrably more than 8 for improved results)(-1 for all images): "))
        continue
    else:
        break
    
totalImages = 0 #track total number of images segmented/processed
startTime = time.time() #start time for tracking the program's execution time

folderlist=os.listdir('Images')
for foldername in folderlist:
    imagelist = os.listdir('Images/'+foldername+"/")
    print("foldername:" + foldername)
    
    # option to process only a random sample of images with size, sampleSize, in the dataset folder
    if (sampleSize != -1):
        imagelist = random.sample(imagelist, sampleSize)
        
    skippedDarkImages = []
    imageAccPairs = []
    
    #Create Results directory
    try:
        os.makedirs("Results/" + foldername, exist_ok = True)
        print("SUCCESS: The '"+foldername+"' directory was created successfully.")
        
    except OSError as error:
        print("ERROR: The '"+foldername+"' directory was NOT CREATED successfully.")
    print()
    
    for imagename in imagelist:
        try:
            print("Folder: " + foldername + " Image: " + imagename)
            inpImage = cv2.imread("Images/"+foldername+"/"+imagename)
            groundTruthMask = cv2.imread("Masks/" + foldername + ".png") #need to make sure that the mask image file format is png
            groundTruthMask = cv2.cvtColor(groundTruthMask, cv2.COLOR_BGR2GRAY)
            
            if(brightOrDark(inpImage)):
                # if bright sky image, continue with the rest of the normal sky detection and segmentation steps
                gradientImg = returnGradientImg(inpImage)
                borderOptimal = borderOptimisation(inpImage, gradientImg)
                mask = createMask(borderOptimal, inpImage)
                postprocessed_mask = postprocessing(mask)
                cv2.imwrite("Results/"+foldername+"/"+ imagename[:-4] + "_mask.png", postprocessed_mask)
                
                acc = calculateIOU(groundTruthMask, postprocessed_mask)*100
                roundedAcc = round(acc, 4)
                imageAccPairs.append([imagename, roundedAcc])
                print("Accuracy for image "+ imagename +" from folder " + foldername + ": " + str(roundedAcc) + "%")
                print()
                
            else:
                # else if not bright sky image and dark sky image, add image to list of skipped images, which will be handled later.
                skippedDarkImages.append(imagename)
                print(imagename + " is a dark image for folder:" + foldername)
                print()
                continue
            
        except:
            print("Error processing image:"+imagename)
            print()
            continue
        
    # Handling the skipped dark images
    if(len(skippedDarkImages) != 0):
        print("Skipped Images list is not empty for folder:" + foldername)
        print("Skipped Dark Images:")
        print(skippedDarkImages)
        
        # sorting computed image name - segmentation accuracy pair in descending order of accuracy to obtain best performing bright sky mask
        print()
        imageAccPairs.sort(key=lambda x: x[1], reverse=True)
        bestImageName = imageAccPairs[0][0]
        bestImageAcc = imageAccPairs[0][1]
        print("For folder " + foldername + " the best image name is " + bestImageName + " with an accuracy of " + str(round(bestImageAcc, 4)) + "%")
        print()
        
        # obtaining best performing bright sky mask
        bestImageMask = cv2.imread("Results/" + foldername + "/" + bestImageName[:-4] + "_mask.png")
        for imgName in skippedDarkImages:
            cv2.imwrite("Results/"+foldername+"/"+ imgName[:-4] + "_mask.png", bestImageMask)
            imageAccPairs.append([imgName, bestImageAcc])
        print("Final Processed Images and Accuracy for images of folder " + foldername + ":")
        print(imageAccPairs)
        print()
    
    # adds total number of images processed
    totalImages += len(imageAccPairs)
    
    # calculating average segmentation accuracy for folder of images
    totalImageAcc = 0
    for imageAccPair in imageAccPairs:
        totalImageAcc += imageAccPair[1]
    averageImageAcc = totalImageAcc/len(imageAccPairs)
    print("Average of segmentation accuracy for folder " + foldername + " images: " + str(round(averageImageAcc, 4)) + "%")
    print()
    averageAccuracies.append([foldername, averageImageAcc])
    
endTime = time.time() #end time for tracking program execution time
duration = round(endTime-startTime, 4) #calculating program execution time duration

totalfolderAcc = 0

# Displaying program performance metrics and results
print("*******************")
print("Performance Metrics")
print("*******************\n")

print("Average segmentation accuracy for each folder of images:")

# calculating total accuracies and printing average segmentation accuracy for each folder
for folderAccPair in averageAccuracies:
    totalfolderAcc += folderAccPair[1]
    print("Folder: " + folderAccPair[0] + " Average segmentation accuracy: " + str(round(folderAccPair[1], 4)) + "%")

# calculating average total accuracy of the program
averageFolderAcc = totalfolderAcc/len(averageAccuracies)
print()

print("Average of segmentation accuracy for all folders' images: " + str(round(averageFolderAcc, 4)) + "%")
print()

# displaying total program execution time and average execution time for a single image
print("Total program execution time for " + str(totalImages) +" images: " + str(duration) + "s")
print("Average program execution time for each image: " + str(round(duration/(totalImages),4)) + "s")