# Sky Region Detection and Segmentation Using a Computer Vision Algorithm With Python and OpenCV
Detection and extraction of the sky pixels (region) in an image using a computer vision algorithm implemented with Python and OpenCV.

Before executing the program, first ensure that there are two files in the same directory as this sky_segmentation.py file, the Images folder and the Masks folder. In the Images folder, add the dataset folders containing the images, such as the 1093, 4795, 8438, 10870 or any other dataset from the Skyfinder dataset or website. In the Masks folder, add the ground truth image for the previous dataset folders, and make sure the name of the ground truth image file is the same as its corresponding dataset folder. For example, ground truth image (in the Masks folder) for dataset folder 1093, should have the name 1093.png.

After that, execute the sky_segmentation.py file, and enter the sample or number of images you would like to test with the program in the Python terminal at the corresponding prompt. Enter -1 to test all the images in the Images folder for each dataset folder. Ideally to prevent errors, enter a minimum of 8.
