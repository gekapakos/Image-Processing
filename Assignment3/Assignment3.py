#Kapakos Georgios 03165
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import imageio.v2 as imageio
import cv2

from scipy import ndimage


img = imageio.imread('images/pencils.png')
plt.imshow(img)


img.shape


plt.figure(figsize=(15,10))
plt.subplot(131); plt.imshow(img[:,:,0], cmap="gray"); plt.title("Red")
plt.subplot(132); plt.imshow(img[:,:,1], cmap="gray"); plt.title("Green")
plt.subplot(133); plt.imshow(img[:,:,2], cmap="gray"); plt.title("Blue")



img_redb = np.array(img, copy=True).astype(np.uint32)
img_redb[:,:,0] = np.clip(img_redb[:,:,0]+100, 0, 255)

img_blub = np.array(img, copy=True).astype(np.uint32)
img_blub[:,:,2] = np.clip(img_blub[:,:,2]+100, 0, 255)

plt.figure(figsize=(15,10))
plt.subplot(131); plt.imshow(img); plt.title("Original")
plt.subplot(132); plt.imshow(img_redb); plt.title("Red boosted")
plt.subplot(133); plt.imshow(img_blub); plt.title("Blue boosted")


img_colormix = np.array(img, copy=True).astype(np.uint32)
img_colormix[:,:,0] = np.clip(0.7*img[:,:,0].astype(np.uint32) + 0.3*img[:,:,2].astype(np.uint32), 0, 255)
img_colormix[:,:,1] = np.clip(0.7*img[:,:,1].astype(np.uint32) + 0.3*img[:,:,0].astype(np.uint32), 0, 255)
img_colormix[:,:,2] = np.clip(0.5*img[:,:,2].astype(np.uint32) + 0.5*img[:,:,1].astype(np.uint32), 0, 255)

plt.figure(figsize=(12,10))
plt.subplot(121); plt.imshow(img); plt.title("Original")
plt.subplot(122); plt.imshow(img_colormix); plt.title("Color mix")


img2 = imageio.imread('images/dog.jpg')
plt.imshow(img2)


img_hsv = mpl.colors.rgb_to_hsv(img2)

plt.figure(figsize=(15,10))
plt.subplot(131); plt.imshow(img_hsv[:,:,0], cmap="hsv"); plt.title("Hue")
plt.subplot(132); plt.imshow(img_hsv[:,:,1], cmap="gray"); plt.title("Saturation")
plt.subplot(133); plt.imshow(img_hsv[:,:,2], cmap="gray"); plt.title("Value")


#Euclidean Distance
def euclidean_distance(point_p, point_q)->float:
    return np.power(np.sum(np.power(point_p-point_q, 2)), 0.5)


#Calculates the histogram of a color or a greyscale image
def color_histogram(img, bins):
    
    size = np.array(img.shape)
    
    if(len(size) == 3):
        #RGB colors of the image:
        img_red = img[:,:,0]
        img_green = img[:,:,1]
        img_blue = img[:,:,2]

        #Get the histograms of the image for each chanel:
        hist_red,bin_red = np.histogram(img_red.ravel(), bins, range = [0, 255])
        hist_green,bin_green = np.histogram(img_green.ravel(),bins, range = [0,255])
        hist_blue,bin_blue = np.histogram(img_blue.ravel(),bins,range = [0,255])

        #Expand the dimensions:
        hist_red2 = np.expand_dims(hist_red, axis = 1)
        hist_green2 = np.expand_dims(hist_green, axis = 1)
        hist_blue2 = np.expand_dims(hist_blue, axis = 1)

        #Concatenate them with the expanded dimensions:
        concatenated_hist = np.concatenate((hist_red2, hist_green2, hist_blue2), 1)
        
        #Calculate the sum of the elements in the histogram:
        sum=0
        for i in range(concatenated_hist.shape[0]):
            for j in range(concatenated_hist.shape[1]):
                sum += concatenated_hist[i][j]
                
        print(sum)
        
        #Normalization of the values:
        concatenated_hist = concatenated_hist/sum
    
    else:
        #Get the histogram of the image:
        hist_,bin_ = np.histogram(img.ravel(), bins, range = [0, 255])
        
        #Expand the dimensions:
        hist_2 = np.expand_dims(hist_, axis = 1)
        
        #Calculate the sum of the elements in the histogram:
        sum=0
        for i in range(hist_2.shape[0]):
            sum += hist_2[i]
        
        #Normalization of the values:
        concatenated_hist = hist_2/sum
    
    return concatenated_hist


#Normalizes the color or grayscale image depending on the min and max value of it.
def min_max_normalization(img):
    
    size = np.array(img.shape)
    
    if(len(size) == 3):
        # red
        img_red_min = img[..., 0].min()
        img_red_max = img[..., 0].max()
        # green
        img_green_min = img[..., 1].min()
        img_green_max =img[..., 1].max()
        # blue
        img_blue_min = img[..., 2].min()
        img_blue_max = img[..., 2].max()

        #Get the max and the min pixel value:
        max_pixel = max(img_red_max, img_green_max, img_blue_max)
        min_pixel = min(img_red_min, img_green_min, img_blue_min)

        #Find the values of the normalized image in the [0,1] space:
        normalized_img = (img-min_pixel)/(max_pixel-min_pixel)
    else:
        
        #Get the max and the min pixel value:
        img_min = img.min()
        img_max = img.max()

        #Find the values of the normalized image in the [0,1] space:
        normalized_img = (img-img_min)/(img_max-img_min)
    
    return normalized_img


# In[85]:


#Calculates the luminance of an image
def luminance(img)->np.array:
    size = np.array(img.shape)
    if(len(size) == 3):
        #Define the chanels:
        channel_0 = img[:,:,0]
        channel_1 = img[:,:,1]
        channel_2 = img[:,:,2]

        #Calculate the luminance image:
        img_luminance = channel_0*0.299 + channel_1*0.587 + channel_2*0.114
    
        #Find the normalized image in the [0,1] area:
        img_luminance_norm = min_max_normalization(img_luminance)

        #Multiply by 255 to get the new luminance image to the [0,255] range:
        img_lum = img_luminance_norm*255
        
        return img_lum
    else:
        return img
    

#Mountain
img_mountain = imageio.imread('images/mountain.jpg')
mountain_hist = color_histogram(img_mountain, 16)

plt.figure(figsize=(8,6))
plt.subplot(1,2,1)
plt.imshow(img_mountain,cmap='gray')
plt.title('Mountain')

plt.subplot(1,2,2)
plt.plot(mountain_hist)
plt.title("RGB")
plt.show()

plt.figure(figsize=(8,6))
plt.subplot(2,3,1)
plt.bar(range(16), mountain_hist[:,0])
plt.title("Red")

plt.subplot(2,3,2)
plt.bar(range(16), mountain_hist[:,1])
plt.title("Green")

plt.subplot(2,3,3)
plt.bar(range(16), mountain_hist[:,2])
plt.title("Blue")
plt.show()


#Pencils
img_pencils = imageio.imread('images/pencils.png')
pencils_hist = color_histogram(img_pencils, 16)

plt.figure(figsize=(8,6))
plt.subplot(1,2,1)
plt.imshow(img_pencils,cmap='gray')
plt.title('Mountain')

plt.subplot(1,2,2)
plt.plot(pencils_hist)
plt.title("RGB")
plt.show()

plt.figure(figsize=(8,6))
plt.subplot(2,3,1)
plt.bar(range(16), pencils_hist[:,0])
plt.title("Red")

plt.subplot(2,3,2)
plt.bar(range(16), pencils_hist[:,1])
plt.title("Green")

plt.subplot(2,3,3)
plt.bar(range(16), pencils_hist[:,2])
plt.title("Blue")
plt.show()


#Pencils 2
img_pencils2 = imageio.imread('images/pencils2.png')
pencils2_hist = color_histogram(img_pencils2, 16)

plt.figure(figsize=(8,6))
plt.subplot(1,2,1)
plt.imshow(img_pencils2,cmap='gray')
plt.title('Mountain')

plt.subplot(1,2,2)
plt.plot(pencils2_hist)
plt.title("RGB")
plt.show()

plt.figure(figsize=(8,6))
plt.subplot(2,3,1)
plt.bar(range(16), pencils2_hist[:,0])
plt.title("Red")

plt.subplot(2,3,2)
plt.bar(range(16), pencils2_hist[:,1])
plt.title("Green")

plt.subplot(2,3,3)
plt.bar(range(16), pencils2_hist[:,2])
plt.title("Blue")
plt.show()


#Euclidean distance:
distance_1_2 = euclidean_distance(mountain_hist, pencils_hist)
distance_1_3 = euclidean_distance(mountain_hist, pencils2_hist)
distance_2_3 = euclidean_distance(pencils_hist, pencils2_hist)

print('Euclidean distance between mountain and pencils:', distance_1_2, '\n')
print('Euclidean distance between mountain and pencils2:', distance_1_3, '\n')
print('Euclidean distance between pencils and pencils2:', distance_2_3)


#Mountain

img_lum_mountain = luminance(img_mountain)
mountain_lum_hist = color_histogram(img_lum_mountain, 16)
plt.figure(figsize=(8,6))
plt.subplot(1,2,1)
plt.imshow(img_lum_mountain,cmap='gray')
plt.title('Mountain')

plt.subplot(1,2,2)
plt.plot(mountain_lum_hist)
plt.title("Grey")
plt.show()

plt.figure(figsize=(8,6))
plt.subplot(2,3,1)
plt.bar(range(16), mountain_lum_hist.ravel())
plt.title("Grey")
plt.show()


#Pencils
img_lum_pencils = luminance(img_pencils)
pencils_lum_hist = color_histogram(img_lum_pencils, 16)

plt.figure(figsize=(8,6))
plt.subplot(1,2,1)
plt.imshow(img_lum_pencils,cmap='gray')
plt.title('Pencils')

plt.subplot(1,2,2)
plt.plot(pencils_lum_hist)
plt.title("Grey")
plt.show()

plt.figure(figsize=(8,6))
plt.subplot(2,3,1)
plt.bar(range(16), pencils_lum_hist.ravel())
plt.title("Grey")
plt.show()


#Pencils2
img_lum_pencils2 = luminance(img_pencils2)
pencils2_lum_hist = color_histogram(img_lum_pencils2, 16)

plt.figure(figsize=(8,6))
plt.subplot(1,2,1)
plt.imshow(img_lum_pencils2,cmap='gray')
plt.title('Pencils')

plt.subplot(1,2,2)
plt.plot(pencils2_lum_hist)
plt.title("Grey")
plt.show()

plt.figure(figsize=(8,6))
plt.subplot(2,3,1)
plt.bar(range(16), pencils_lum_hist.ravel())
plt.title("Grey")
plt.show()



#Euclidean distance:
distance_lum_1_2 = euclidean_distance(mountain_lum_hist, pencils_lum_hist)
distance_lum_1_3 = euclidean_distance(mountain_lum_hist, pencils2_lum_hist)
distance_lum_2_3 = euclidean_distance(pencils_lum_hist, pencils2_lum_hist)

print('Euclidean distance between mountain and pencils:', distance_lum_1_2, '\n')
print('Euclidean distance between mountain and pencils2:', distance_lum_1_3, '\n')
print('Euclidean distance between pencils and pencils2:', distance_lum_2_3)


#Observations:
#We observe that the distance between the images
#in the luminance images is greater than in the color
#images.