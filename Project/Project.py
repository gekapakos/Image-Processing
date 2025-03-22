#KAPAKOS GEORGIOS: 03165
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from os.path import join, dirname, realpath

import sys
import cv2
from PIL import Image, ImageFilter, ImageDraw
from scipy import ndimage
from IPython.display import display
from tkinter import Label, Tk, Button, ttk

#Prints the operations in the filter menu
def print_filter_menu():
    print("Available filters and operations:\n1)Smooth\n2)Bright\n3)Sharpen\n4)Grayscale\n5)Bilateral\n6)Color mix\n7)Exit\n8)Load meme\n9)Save\n10)Green\n11)Erase filters\n12)Binary\n13)Add text\n")

#Prints the operations in the meme menu
def print_meme_menu():
    print("Available meme spot operations:\n1)Face operation\n2)Mouth operation\n3)Eyes operation\n4)Head operation\n5)Top left background operation\n6)Torso operation\n7)Top right background operation\n")

#Checks if an image is in an RGB or a GBR  format
def rgb_or_bgr(image):
    # Check the color channel order using image metadata
    if cv2.cvtColor(image, cv2.COLOR_BGR2RGB) is not None:
        return 1# BGR Format 
    elif cv2.cvtColor(image, cv2.COLOR_RGB2BGR) is not None:
        return 1# RGB Format
    else:
        print("Failed to determine color channel order")

#Transform an image from 3-channel to 4-channel
def transparent(image):
    height, width, _ = image.shape

    new_image = np.zeros((height, width, 4), dtype=np.uint8)
    new_image[:,:,0] = image[:,:,0]
    new_image[:,:,1] = image[:,:,1]
    new_image[:,:,2] = image[:,:,2]
    new_image[:,:,3] = 255
    
    return new_image

#Function to calculate the size of the image format
def calc_size(img_path):
    size = i = j = counter = 0
    for i in range(len(img_path)):
        if(counter == 1):
            size+=1
        if(img_path[i] == '.'):
            counter+=1
        if(counter > 1):
            return 0
    return size

#Function to return the format of the image I want to load
def format_finder(img_path, size_format):
    if(size_format == 0):
        return None
    counter = i = j = 0
    img_format = np.zeros(shape = (size_format)).astype(str)
    for i in range (len(img_path)):
        if(counter == 1 and i < len(img_path)):
            img_format[j] = img_path[i]
            j+=1
        if(img_path[i] == '.'):
            counter += 1
        elif(counter > 1):
            break
        else:
            continue
    return img_format

#Function that checks if the images' format is allowed.
def check_format(img_format):
    if((img_format) == 'jpg' or (img_format) == 'jpeg' or (img_format) == 'tiff' or (img_format) == 'png'):
        return 1
    else:
        return 0

#Function that checks if the images' format is jpeg:
def check_format_jpeg(img_format):
    if((img_format) == 'jpeg'):
        return 1
    else:
        return 0

#Plots the image
def plot_img(img):
    
    if(rgb_or_bgr(img) == 1):
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        im_rgb = img
        
    plt.figure(figsize=(5,5))
    plt.imshow(im_rgb, cmap = 'gray')
    plt.axis('off')
    print(img.shape)
    plt.pause(.5)

#Function to convert a list into a string 
def list_to_string(in_list):
    my_string = ''
    for i in in_list:
        my_string += '' + i
    return my_string

#function to find how many times the '\n' exists in it
def count_newline(string):
    sum = 0
    for i in range(len(string)):
        if(string[i] == '\n'):
            sum+=1
    return sum

#Function that detects the face and the processes the face by putin a meme on it
def face_detect(image, meme):
    #I use the haar code to detect the face of my image:
    face_Cascade = cv2.CascadeClassifier("Haar/haarcascade_frontalface_default.xml")
    
    #Transform the image into a transparent one and the meme also:
    new_image = transparent(image)
    meme = transparent(meme)
    
    #Convert the image to grayscale one:
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #Choose the parameters to detect the face:
    faces = face_Cascade.detectMultiScale(imgGray, 1.1, 9)
    
    #This is an arbitrary function to scale the images so that they will look more realistic:
    scale = int((new_image.shape[0]+new_image.shape[1])/50)
    
    #Do an iteration and create a rectangle around the face:
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x-scale*2, y-scale*4), (x + w+scale*2, y + h+scale*1), (255, 0, 0), 2)
    
    #Resize the meme based on the coordinates of the rextangle and the scale factor:
    meme_resized = cv2.resize(meme, (w+scale*4, h+scale*5))
    
    #Coordinates of the meme:
    rows,cols,channels = meme_resized.shape 
    
    #The region of interest(Roi), when I will put on my meme:
    roi = new_image[y-scale*4:y+h+scale, x-scale*2:x+w+scale*2]

    #grayscale meme
    gray_meme = cv2.cvtColor(meme_resized, cv2.COLOR_BGR2GRAY)
    
    #Convert a grayscale image to black and white using binary thresholding 
    (thresh, binary_meme) = cv2.threshold(gray_meme, 1, 255, cv2.THRESH_BINARY)
    binary_meme_inv = cv2.bitwise_not(binary_meme)
        
    #Check if there is a difference in the shapes and there is return None
    if((binary_meme_inv.shape[0]!=roi.shape[0]) or binary_meme_inv.shape[1]!=roi.shape[1]):
        return None

    # Now black-out the area of logo in ROI
    image_bg = cv2.bitwise_and(roi,roi,mask = binary_meme_inv)

    # Take only region of logo from logo image.
    meme_fg = cv2.bitwise_and(meme_resized,meme_resized,mask = binary_meme)

    # Put logo in ROI and modify the main image
    dst = cv2.add(image_bg,meme_fg)
    
    #After the masking obtain the final image:
    new_image[y-scale*4:y+h+scale, x-scale*2:x+w+scale*2] = dst

    return new_image

#Function that detects the mouth and the processes the mouth by putin a meme on it
def mouth_detect(image, meme):
    
    #I use the haar code to detect the mouth of my image:
    mouth_Cascade = cv2.CascadeClassifier("Haar/haarcascade_mouth.xml")
    
    #Transform the image into a transparent one and the meme also:
    new_image=transparent(image)
    meme = transparent(meme)
    
    #Convert the image to grayscale one:
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #Choose the parameters to detect the face:
    mouths = mouth_Cascade.detectMultiScale(imgGray, 1.5, 30)
    
    #This is an arbitrary function to scale the images so that they will look more realistic:
    scale = int((new_image.shape[0]+new_image.shape[1])/50)
    
    #Initialize the mouth values:
    mx = 0 
    my = 0 
    mw = 0 
    mh = 0
    #Do an iteration and create a rectangle around the mouth:
    for (mx, my, mw, mh) in mouths:
        cv2.rectangle(image, (mx, my), (mx + mw, my + mh), (255, 0, 0), 2)
    
    #Check if the coordinates are zero if they are return None:
    if(mw == 0):
        return None
    
    #Resize the meme based on the coordinates of the rextangle and the scale factor:
    meme_resized = cv2.resize(meme, (mw, mh))
    
    #Coordinates of the meme:
    rows,cols,channels = meme_resized.shape 
    
    #The region of interest(Roi), when I will put on my meme:
    roi = new_image[my+(scale//2):my+mh+(scale//2), mx+(scale*3):mx+mw+(scale*3)]

    #grayscale meme
    gray_meme = cv2.cvtColor(meme_resized, cv2.COLOR_BGR2GRAY)
    #Convert a grayscale image to black and white using binary thresholding 
    (thresh, binary_meme) = cv2.threshold(gray_meme, 1, 255, cv2.THRESH_BINARY)
    binary_meme_inv = cv2.bitwise_not(binary_meme)
    
    #Check if there is a difference in the shapes and there is return None
    if((binary_meme_inv.shape[0]!=roi.shape[0]) or binary_meme_inv.shape[1]!=roi.shape[1]):
        return None

    # Now black-out the area of logo in ROI
    image_bg = cv2.bitwise_and(roi,roi,mask = binary_meme_inv)

    # Take only region of logo from logo image.
    meme_fg = cv2.bitwise_and(meme_resized,meme_resized,mask = binary_meme)

    # Put logo in ROI and modify the main image
    dst = cv2.add(image_bg,meme_fg)
    
    #After the masking obtain the final image:
    new_image[my+(scale//2):my+mh+(scale//2), mx+(scale*3):mx+mw+(scale*3)] = dst

    return new_image

#Function that detects the eyes and then processes the eyes by putin a meme on them
def eyes_detect(image, meme):
    
    # Load the pre-trained Haar cascade classifier for eye detection
    eye_cascade = cv2.CascadeClassifier('Haar/haarcascade_eye.xml')
    
    #Transform the image into a transparent one and the meme also:
    new_image = transparent(image)
    meme = transparent(meme)

    # Convert the image to grayscale for better processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect eyes in the image
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(30, 30))

    # Check if eyes were detected
    if len(eyes) >= 2:
        # Calculate the bounding box for both eyes
        x, y, w, h = eyes[0]  # First eye
        x2, y2, w2, h2 = eyes[1]  # Second eye

        # Calculate the coordinates for the rectangle that frames both eyes
        x_min = min(x, x2)
        y_min = min(y, y2)
        x_max = max(x + w, x2 + w2)
        y_max = max(y + h, y2 + h2)
        
        #This is an arbitrary function to scale the images so that they will look more realistic:
        scale = int((new_image.shape[0]+new_image.shape[1])/50)
        
        # Draw the rectangle on the image
        cv2.rectangle(image, (x_min-scale*2, y_min-scale*2), (x_max+scale*2, y_max+scale*2), (0, 255, 0), 2)
        
        #Resize the meme based on the coordinates of the rextangle and the scale factor:
        meme_resized = cv2.resize(meme, (x_max-x_min+4*scale, y_max-y_min+4*scale))
        
        #Coordinates of the meme:
        rows,cols,channels = meme_resized.shape
        
        #The region of interest(Roi), when I will put on my meme:
        roi = new_image[y_min-2*scale:y_max+scale*2, x_min-scale*2:x_max+scale*2]
        
        #grayscale meme
        gray_meme = cv2.cvtColor(meme_resized, cv2.COLOR_BGR2GRAY)
        #Convert a grayscale image to black and white using binary thresholding 
        (thresh, binary_meme) = cv2.threshold(gray_meme, 1, 255, cv2.THRESH_BINARY)
        binary_meme_inv = cv2.bitwise_not(binary_meme)
        
        #Check if there is a difference in the shapes and there is return None
        if((binary_meme_inv.shape[0]!=roi.shape[0]) or binary_meme_inv.shape[1]!=roi.shape[1]):
            return None

        # Now black-out the area of logo in ROI
        image_bg = cv2.bitwise_and(roi,roi,mask = binary_meme_inv)

        # Take only region of logo from logo image.
        meme_fg = cv2.bitwise_and(meme_resized,meme_resized,mask = binary_meme)

        # Put logo in ROI and modify the main image
        dst = cv2.add(image_bg,meme_fg)
        new_image[y_min-scale*2:y_max+scale*2, x_min-scale*2:x_max+scale*2] = dst
        #return image
        return new_image
    else:
        print("Could not detect both eyes.")

#Function that detects the face and then processes the torso by puting a meme on it
def torso_detect(image, meme):
    #I use the haar code to detect the face of my image:
    face_Cascade = cv2.CascadeClassifier("Haar/haarcascade_frontalface_default.xml")
    
    #Transform the image into a transparent one and the meme also:
    new_image=transparent(image)
    meme = transparent(meme)
    
    #Coordinates of the image:
    rows_im,cols_im,channels_im = new_image.shape
    
    #Convert the image to grayscale one:
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #Choose the parameters to detect the face:
    faces = face_Cascade.detectMultiScale(imgGray, 1.1, 9)
    
    #This is an arbitrary function to scale the images so that they will look more realistic:
    scale = int((new_image.shape[0]+new_image.shape[1])/50)
    
    #Do an iteration and create a rectangle around the face:
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    #Use these variables to describe in a better way the coordinates:
    y_min = y
    y_max = y+h
    x_min= x
    x_max = x+w
    
    factor = scale*7
    
    #Resize the meme based on the coordinates of the rextangle and the scale factor:
    meme_resized = cv2.resize(meme, (w, factor))
    
    #Coordinates of the meme:
    rows,cols,channels = meme_resized.shape 
    
    #The region of interest(Roi), when I will put on my meme:
    roi = new_image[y_max:y_max+factor, x:x+w]

    #grayscale meme
    gray_meme = cv2.cvtColor(meme_resized, cv2.COLOR_BGR2GRAY)
    #Convert a grayscale image to black and white using binary thresholding 
    (thresh, binary_meme) = cv2.threshold(gray_meme, 1, 255, cv2.THRESH_BINARY)
    binary_meme_inv = cv2.bitwise_not(binary_meme)
    
    #Check if there is a difference in the shapes and there is return None
    if((binary_meme_inv.shape[0]!=roi.shape[0]) or binary_meme_inv.shape[1]!=roi.shape[1]):
        return None
    
    # Now black-out the area of logo in ROI
    image_bg = cv2.bitwise_and(roi,roi,mask = binary_meme_inv)

    # Take only region of logo from logo image.
    meme_fg = cv2.bitwise_and(meme_resized,meme_resized,mask = binary_meme)

    # Put logo in ROI and modify the main image
    dst = cv2.add(image_bg,meme_fg)
    
    #After the masking obtain the final image:
    new_image[y_max:y_max+factor, x:x+w] = dst

    return new_image

#Function that detects the face and then processes the head by putin a meme on it
def head_detect(image, meme):
    #I use the haar code to detect the face of my image:
    face_Cascade = cv2.CascadeClassifier("Haar/haarcascade_frontalface_default.xml")
    
    #Transform the image into a transparent one and the meme also:
    new_image=transparent(image)
    meme = transparent(meme)
    
    #Convert the image to grayscale one:
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #Choose the parameters to detect the face:
    faces = face_Cascade.detectMultiScale(imgGray, 1.1, 12)
    
    #This is an arbitrary function to scale the images so that they will look more realistic:
    scale = int((new_image.shape[0]+new_image.shape[1])/50)
    
    #Do an iteration and create a rectangle around the face:
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    #Use these variables to describe in a better way the coordinates:
    y_min = y
    y_max = y+h
    x_min= x
    x_max = x+w
    
    #Resize the meme based on the coordinates of the rextangle and the scale factor:
    meme_resized = cv2.resize(meme, (x_max-x_min+1*scale, (int(y_min+scale*2))))
    
    #Coordinates of the meme:
    rows,cols,channels = meme_resized.shape
    
    #The region of interest(Roi), when I will put on my meme:
    roi = new_image[(scale*1):y_min+scale*3, x_min-scale//2:x_max+scale//2]

    #grayscale meme
    gray_meme = cv2.cvtColor(meme_resized, cv2.COLOR_BGR2GRAY)
    #Convert a grayscale image to black and white using binary thresholding 
    (thresh, binary_meme) = cv2.threshold(gray_meme, 1, 255, cv2.THRESH_BINARY)
    binary_meme_inv = cv2.bitwise_not(binary_meme)
    
    #Check if there is a difference in the shapes and there is return None
    if((binary_meme_inv.shape[0]!=roi.shape[0]) or binary_meme_inv.shape[1]!=roi.shape[1]):
        return None
    
    # Now black-out the area of logo in ROI
    image_bg = cv2.bitwise_and(roi,roi,mask = binary_meme_inv)

    # Take only region of logo from logo image.
    meme_fg = cv2.bitwise_and(meme_resized,meme_resized,mask = binary_meme)

    # Put logo in ROI and modify the main image
    dst = cv2.add(image_bg,meme_fg)
    
    #After the masking obtain the final image:
    new_image[(scale*1):y_min+scale*3, x_min-scale//2:x_max+scale//2] = dst

    return new_image

#Function that puts a meme on the top left corner
def background_top_left(img, meme):
    #Transform the image into a transparent one and the meme also:
    new_img = transparent(img)
    meme = transparent(meme)
    
    #The coordinates of the image:
    size_rows,size_cols,size_channels = new_img.shape 
    
    #This is an arbitrary function to scale the images so that they will look more realistic:
    scale = int((new_img.shape[0]+new_img.shape[1])/50)
    
    #Resize the meme
    meme_resized = cv2.resize(meme, (scale*10, scale*10))
    meme_resized = transparent(meme_resized)

    # I want to put logo on top-left corner, So I create a ROI
    rows,cols,channels = meme_resized.shape 
    roi = new_img[0:scale*10, 0:scale*10]

    # Now create a mask of logo and create its inverse mask also
    memegray = cv2.cvtColor(meme_resized,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(memegray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

    # Take only region of logo from logo image.
    meme_fg = cv2.bitwise_and(meme_resized,meme_resized,mask = mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img_bg,meme_fg)
    new_img[0:scale*10, 0:scale*10] = dst

    return new_img

#Function that puts a meme on the top left corner
def background_top_right(img, meme):
    #Transform the image into a transparent one and the meme also:
    new_img = transparent(img)
    meme = transparent(meme)
    
    #The coordinates of the image:
    size_rows,size_cols,size_channels = new_img.shape 
    
    #This is an arbitrary function to scale the images so that they will look more realistic:
    scale = int((new_img.shape[0]+new_img.shape[1])/50)
    
    #Resize the meme
    meme_resized = cv2.resize(meme, (scale*10, scale*10))
    meme_resized = transparent(meme_resized)

    # I want to put logo on top-left corner, So I create a ROI
    rows,cols,channels = meme_resized.shape 
    roi = new_img[0:scale*10, size_cols-scale*10:size_cols]

    # Now create a mask of logo and create its inverse mask also
    memegray = cv2.cvtColor(meme_resized,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(memegray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

    # Take only region of logo from logo image.
    meme_fg = cv2.bitwise_and(meme_resized,meme_resized,mask = mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img_bg,meme_fg)
    new_img[0:scale*10, size_cols-scale*10:size_cols] = dst

    return new_img

#To acquire each image path you just have to type the image as it is in the folder, not the whole path.
processed_img = None
while(1):
    #Choose what you would like to do in the main menu:
    choice = input("\n~~~~~~~~~~~~~~~~~~~~~~\nLoad a new image: 'l' \nPrint the available images: 'p' \nRemove an image form the file: 'r' \nPrint made meme: 'm' \nDelete processed meme image: 'rm' \nExit 'e'\n~~~~~~~~~~~~~~~~~~~~~~\n")
    #Print the stored memes:
    if(choice == 'm'):
        #As input you must type the name of the image and it's format (e.x Jordan_Peterson_meme.jpeg)
        img_meme_path = input("Which image would you like to load?:\n")
        
        #Determine if the image has the right atributes to be proccessed
        #Find the size of the format
        size_meme_format = calc_size(img_meme_path)
        img_meme_format = format_finder(img_meme_path, size_meme_format)
        #If the size of the format is greater than zero proceed:
        if(size_meme_format != 0):
            img_meme_format = ''.join(img_meme_format)
            #Check if the image format is valid acoording to the available formats:
            check_img_meme = check_format(img_meme_format)
            if(check_img_meme == 1):
                img_meme_path = "processed_memes/" + img_meme_path
                #Find out if the image exists:
                if(os.path.exists(img_meme_path)):
                    img_meme = cv2.imread(img_meme_path)
                    plot_img(img_meme)
    #Remove an image from the images folder where it has been stored:
    if(choice == 'r'):
        #Retrieve the image's path
        img_rmv = input("Which image would you like to remove?:\n")
        img_path = "images/" + img_rmv
        #If the image exists in the folder "images" delete it:
        if(os.path.exists(img_path)):
            os.remove(img_path)
        else:
            print("There is no such image\n")
    #Remove an processed image from the processed_memes folder where it has been stored:
    if(choice == "rm"):
        #Retrieve the image's path
        img_meme_rmv = input("Which image would you like to remove?:\n")
        img_meme_path = "processed_memes/" + img_meme_rmv
        #If the image exists in the folder "images" delete it:
        if(os.path.exists(img_meme_path)):
            os.remove(img_meme_path)
        else:
            print("There is no such image\n")

    #Load and process an image
    elif(choice == 'l'):
        #As input you must type the name of the image and it's format (e.x Jordan_Peterson.jpg)
        img_path = input("Which image would you like to load?:\n")
        
        #Determine if the image has the right atributes to be proccessed
        #Find the size of the format
        size_format = calc_size(img_path)
        img_format = format_finder(img_path, size_format)
        #If the size of the format is greater than zero proceed:
        if(size_format != 0):
            img_format = ''.join(img_format)
            #Check if the image format is valid acoording to the available formats:
            check = check_format(img_format)
            if(check == 1):
                img_path = "images/" + img_path
                #Find out if the image exists:
                if(os.path.exists(img_path)):
                    img = cv2.imread(img_path)
                    plot_img(img)
                    #Copy the initial image to the processed image:
                    processed_img = img.copy()
                    while(1):
                        print_filter_menu()
                        img_filter_choice = input("How would you like to process the image?\n:")
                        #Smoothness Filter
                        if(img_filter_choice == "Smooth"):
                            processed_img = cv2.GaussianBlur(processed_img, (13,13), 0)
                            plot_img(processed_img)
                        #Brightness Filter
                        elif(img_filter_choice == "Bright"):
                            alpha = 1.5 # Contrast control
                            beta = 10 # Brightness control
                            # call convertScaleAbs function
                            processed_img = cv2.convertScaleAbs(processed_img, alpha=alpha, beta=beta)
                            plot_img(processed_img)
                        #Sharpness Filter
                        elif(img_filter_choice == "Sharpen"):
                            # Create the sharpening kernel
                            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                            # Apply the sharpening kernel to the image using filter2D
                            processed_img = cv2.filter2D(processed_img, -1, kernel)
                            plot_img(processed_img)
                        #Grayscale Filter
                        elif(img_filter_choice == "Grayscale"):
                            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
                            plt.figure(figsize=(5,5))
                            plot_img(processed_img)
                        #Bilateral Filter
                        elif(img_filter_choice == "Bilateral"):
                            # Apply bilateral filter with d = 15, 
                            # sigmaColor = sigmaSpace = 75.
                            #Convert the image to an uint8 type first:
                            processed_img = processed_img.astype('uint8')
                            #Check if the image is rgb or bgr format:
                            if(rgb_or_bgr(processed_img) == 1):
                                processed_img2 = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                            processed_img = cv2.bilateralFilter(processed_img2, 15, 75, 75)
                            #Convert it back to rgb format:
                            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                            plot_img(processed_img)
                        #Color mix Filter
                        elif(img_filter_choice == "Color mix"):
                            #If the image is in grayscale then convert it to rgb first and then proceed:
                            if(len(processed_img.shape)<=2):
                                processed_img = cv2.cvtColor(processed_img,cv2.COLOR_GRAY2RGB)
                            processed_img = np.array(processed_img, copy=True).astype(np.uint8)
                            processed_img[:,:,0] = np.clip(0.7*processed_img[:,:,0].astype(np.uint8) + 0.3*processed_img[:,:,2].astype(np.uint8), 0, 255)
                            processed_img[:,:,1] = np.clip(0.7*processed_img[:,:,1].astype(np.uint8) + 0.3*processed_img[:,:,0].astype(np.uint8), 0, 255)
                            processed_img[:,:,2] = np.clip(0.5*processed_img[:,:,2].astype(np.uint8) + 0.5*processed_img[:,:,1].astype(np.uint8), 0, 255)
                            plot_img(processed_img)
                        #Exit
                        elif(img_filter_choice == "Exit" or img_filter_choice == "exit" or img_filter_choice == "EXIT"):
                            print("No filter applied\n");break;
                        #Load meme
                        elif(img_filter_choice == 'Load meme'):
                            #As input you must type the name of the meme and its' format (e.x glasses.png)
                            meme_path = input("Which meme would you like to load?:\n")
                            #Find out if the meme's format size:
                            size_meme_format = calc_size(meme_path)
                            #Find out the format of the meme:
                            meme_format = format_finder(meme_path, size_meme_format)
                            if(size_meme_format != 0):
                                meme_format = ''.join(meme_format)
                                #Check if the format is a valid one:
                                check_meme = check_format(meme_format)
                                if(check_meme == 1):
                                    meme_path = "meme_assets/" + meme_path
                                    #Find out if the meme exists:
                                    if(os.path.exists(meme_path)):
                                        meme = cv2.imread(meme_path)
                                        plot_img(meme)
                                        print_meme_menu()
                                        meme_choice = input("Which detection method would you like to choose?:\n")
                                        #Face operation:
                                        if(meme_choice == 'Face operation'):
                                            processed_img2 = processed_img.copy()
                                            processed_img2 = face_detect(processed_img, meme)
                                            if(processed_img2 is not None):
                                                processed_img = processed_img2.copy()
                                                plot_img(processed_img)
                                            else:
                                                print("Could not detect face...\n")
                                        #Mouth operation:
                                        elif(meme_choice == 'Mouth operation'):
                                            processed_img2 = processed_img.copy()
                                            processed_img2 = mouth_detect(processed_img, meme)
                                            if(processed_img2 is not None):
                                                processed_img = processed_img2.copy()
                                                plot_img(processed_img)
                                            else:
                                                print("Could not detect mouth...\n")
                                        #Eyes operation:
                                        elif(meme_choice == 'Eyes operation'):
                                            processed_img2 = processed_img.copy()
                                            processed_img2 = eyes_detect(processed_img, meme)
                                            if(processed_img2 is not None):
                                                processed_img = processed_img2.copy()
                                                plot_img(processed_img)
                                            else:
                                                print("Could not detect eyes...\n")
                                        #Head operation:
                                        elif(meme_choice == 'Head operation'):
                                            processed_img2 = processed_img.copy()
                                            processed_img2 = head_detect(processed_img, meme)
                                            if(processed_img2 is not None):
                                                processed_img = processed_img2.copy()
                                                plot_img(processed_img)
                                            else:
                                                print("Could not detect head...\n")
                                        #Torso operation:
                                        elif(meme_choice == 'Torso operation'):
                                            processed_img2 = processed_img.copy()
                                            processed_img2 = torso_detect(processed_img, meme)
                                            if(processed_img2 is not None):
                                                processed_img = processed_img2.copy()
                                                plot_img(processed_img)
                                            else:
                                                print("Could not detect torso...\n")
                                        #Top left background operation
                                        elif(meme_choice == 'Top left background operation'):
                                            processed_img = background_top_left(processed_img, meme)
                                            plot_img(processed_img)                                            
                                        #Top right background operation
                                        elif(meme_choice == 'Top right background operation'):
                                            processed_img = background_top_right(processed_img, meme)
                                            plot_img(processed_img)
                                        else:
                                            print("There is no such choice for detection and filter available\n")
                        #Save the image:
                        elif(img_filter_choice == 'Save'):
                            save_choice = input("Are you sure that you would like to save the processed image?\n")

                            if(save_choice == "Yes" or save_choice == "YES" or save_choice == "yes"):
                                new_image_name = input("What is the name of the new image?\n")
                                new_image_path = "processed_memes/" + new_image_name
                                
                                #Check again if the format is in correct form:
                                processed_size_format = calc_size(new_image_path)
                                processed_img_format = format_finder(new_image_path, processed_size_format)
                                if(processed_size_format != 0):
                                    processed_img_format = ''.join(processed_img_format)
                                    processed_check = check_format_jpeg(processed_img_format)
                                    if(processed_check == 1):
                                        processed_img_path = "processed_memes/" + new_image_path
                                        #Store the processed image:
                                        cv2.imwrite(new_image_path, (processed_img))
                                    else:
                                        print("Image will not be saved\n")
                                else:
                                    print("Image will not be saved\n")
                        #Green filter:
                        elif(img_filter_choice == "Green"):
                            #If the processed image is in grayscale format convert before using the filter: 
                            if(len(processed_img.shape) <= 2):
                                print("Image doesn't have 3-channels and can't be processed\n");continue
                            
                            processed_img = np.array(processed_img, copy=True).astype(np.uint8)
                            processed_img[:,:,0] = np.clip(1*processed_img[:,:,0].astype(np.uint8) + 0*processed_img[:,:,2].astype(np.uint8), 0, 255)
                            processed_img[:,:,1] = np.clip(1*processed_img[:,:,1].astype(np.uint8) + 0*processed_img[:,:,1].astype(np.uint8), 0, 255)
                            processed_img[:,:,2] = np.clip(0.0*processed_img[:,:,2].astype(np.uint8) + 1*processed_img[:,:,0].astype(np.uint8), 0, 255)
                            plot_img(processed_img)
                        #Erase all the filters of the image:
                        elif(img_filter_choice == "Erase filters"):
                            processed_img = img.copy()
                            plt.pause(.5)
                            plot_img(processed_img)
                        #Binary filter:
                        elif(img_filter_choice == "Binary"):
                            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
                            (thresh, processed_img) = cv2.threshold(processed_img, 100, 255, cv2.THRESH_BINARY)
                            plot_img(processed_img)
                        #Add text starting from the left bottom corner of the image
                        elif(img_filter_choice == "Add text"):
                            
                            #scale
                            scale = (processed_img.shape[0]+processed_img.shape[1])/1700
                            # font
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            # fontScale
                            fontScale = 1.2*scale
                            # Color in BGR
                            while(1):
                                color = input("Choose the color of the text:\n")
                                print("Options are:\n1)black\n2)white\n3)green\n4)red\n5)blue\n6)yellow\n")
                                if(color == 'black'):
                                    color = (0, 0, 0)
                                    break
                                elif(color == 'white'):
                                    color = (255, 255, 255)
                                    break
                                elif(color == 'green'):
                                    color = (0, 255, 0)
                                    break
                                elif(color == 'red'):
                                    color = (0, 0, 255)
                                    break
                                elif(color == 'blue'):
                                    color = (255, 0, 0)
                                    break
                                elif(color == 'yellow'):
                                    color = (0, 255, 255)
                                    break
                                else:
                                    print("Invalid color choice please type a color that exists:\n")
                            # Line thickness of 2 px
                            thickness = 2
                            
                            #Choose one the available texts, they are in the form: (e.x Jordan_Peterson_quote.txt)
                            img_txt = input("What quote would you like to use?\n")
                            img_text = "texts/" + img_txt
                            
                            #Check if the path exists:
                            if(os.path.exists(img_text)):
                                text = open(img_text, "r")
                                text_contents = text.read()
                                #Convert the text to a string:
                                text_str = list_to_string(text_contents)
                                
                                #Find out the number of the text's lines:
                                size = count_newline(text_contents)
                                size +=1
                                
                                #Split the text beased on the new line operators:
                                text_split = text_str.split("\n")

                                # org
                                #Initialize the org list
                                org_list = ['']*size
                                org_list = list(org_list)

                                #Organise the text 
                                for i in range(size):
                                    prod = (i+1)*60
                                    org_list[size-i-1] = (0, (processed_img.shape[0]-prod))

                                #Write the text in the image
                                for i in range(size):
                                    image = cv2.putText(processed_img, text_split[i], org_list[i], font, fontScale, color, thickness, cv2.LINE_AA)

                                # Displaying the image
                                plot_img(processed_img)
                            else:
                                print("Invalid path...\n")

                        else:
                            print("Unvalid choice. Please type that you would like again\n")             
                else:
                    print("Image does not exist\n")
            else:
                print("Error wrong image format\n")
        else:
            print("Error wrong image format\n")
    #Print the contents of the images file, what are the available pictures to create memes:
    elif(choice == 'p'):
        # Get the list of all files and directories
        path = "images//"
        dir_list = os.listdir(path)
        print("Files and directories in '", path, "' :\n")
        # prints all files
        print(dir_list)
    #Exit from the program:
    elif(choice == 'e'):
        print("Exit...\n")
        break
    else:
        print("This choice is not available\n")