#!/usr/bin/env python
# coding: utf-8

# In[52]:


#Kapakos Georgios 03165
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


# In[53]:


img = imageio.imread('images/scarlett.jpg')
plt.figure(figsize=(10,10))

# bit-wise operators
img_b1 = img & 0b10000000 #8 bit as our img is grayscale [0,255]
img_b2 = img & 0b01000000 #8 bit as our img is grayscale [0,255]
img_b3 = img & 0b00100000 #8 bit as our img is grayscale [0,255]
img_b4 = img & 0b00010000 #8 bit as our img is grayscale [0,255]
img_b5 = img & 0b00001000 #8 bit as our img is grayscale [0,255]
img_b6 = img & 0b00000100 #8 bit as our img is grayscale [0,255]
img_b7 = img & 0b00000010 #8 bit as our img is grayscale [0,255]
img_b8 = img & 0b00000001 #8 bit as our img is grayscale [0,255]

plt.subplot(331)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.subplot(332)
plt.imshow(img_b1, cmap='gray')
plt.axis('off')
plt.subplot(333)
plt.imshow(img_b2, cmap='gray')
plt.axis('off')
plt.subplot(334)
plt.imshow(img_b3, cmap='gray')
plt.axis('off')
plt.subplot(335)
plt.imshow(img_b4, cmap='gray')
plt.axis('off')
plt.subplot(336)
plt.imshow(img_b5, cmap='gray')
plt.axis('off')
plt.subplot(337)
plt.imshow(img_b6, cmap='gray')
plt.axis('off')
plt.subplot(338)
plt.imshow(img_b7, cmap='gray')
plt.axis('off')
plt.subplot(339)
plt.imshow(img_b8, cmap='gray')
plt.axis('off')


# In[54]:


A = np.random.randint(0,8,[5,5])
print(A)


# In[55]:


def histogram(A, no_levels):
    N, M = A.shape # NxM image/matrix --> img may not be square
    hist = np.zeros(no_levels).astype(int) #bucket
    for i in range(no_levels):
        pixel_value_i = np.sum(A==i)
        hist[i] = pixel_value_i
        
    return hist


# In[56]:


hist_A = histogram(A,8)
plt.bar(range(8), hist_A)
plt.xlabel('Intensity')
plt.ylabel('Frequency')


# In[57]:


img1 = imageio.imread('images/nap.jpg')
img2 = imageio.imread('images/scarlett.jpg')

# compute histograms
hist_img1 = histogram(img1, 256)
hist_img2 = histogram(img2, 256)

# visualization
plt.figure(figsize=(10,10))
plt.subplot(221)
plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.subplot(223)
plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
plt.axis('off')

plt.subplot(222)
plt.bar(range(256), hist_img1)
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.subplot(224)
plt.bar(range(256), hist_img2)
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.show()


# In[58]:


def histogram_equalization(A, no_levels):
    hist = histogram(A, no_levels)
    histC = np.zeros(no_levels).astype(int)
    histC[0] = hist[0]
    for i in range(1, no_levels): # scan from intensity 1 up to no_levels-1
        histC[i] = hist[i] + histC[i-1]
        
    hist_transform = np.zeros(no_levels).astype(np.uint8)
    
    N, M = A.shape
    
    A_eq = np.zeros([N,M]).astype(np.uint8) #equalized "new" image
    
    for z in range(no_levels):
        scale = ((no_levels-1)/float(N*M))
        s = scale * histC[z]
        A_eq[np.where(A==z)] = s
        
        hist_transform[z] = s
        
    return (A_eq, hist_transform)


# In[59]:


img1_eq, img1_transf = histogram_equalization(img1, 256)
img2_eq, img2_transf = histogram_equalization(img2, 256)


# In[60]:


histeq_img1 = histogram(img1_eq, 256)
histeq_img2 = histogram(img2_eq, 256)

# visualization
plt.figure(figsize=(14,14))
plt.subplot(3,2,1)
plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.subplot(3,2,2)
plt.bar(range(256), hist_img1)
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.subplot(3,2,4)
plt.plot(range(256), img1_transf)
plt.xlabel('input pixel value (r)')
plt.ylabel('output pixel value (s)')

plt.subplot(3,2,5)
plt.imshow(img1_eq, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.subplot(3,2,6)
plt.bar(range(256), histeq_img1)
plt.xlabel('Intensity')
plt.ylabel('Frequency')


# In[61]:


# visualization
plt.figure(figsize=(14,14))
plt.subplot(3,2,1)
plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.subplot(3,2,2)
plt.bar(range(256), hist_img2)
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.subplot(3,2,4)
plt.plot(range(256), img2_transf)
plt.xlabel('input pixel value (r)')
plt.ylabel('output pixel value (s)')

plt.subplot(3,2,5)
plt.imshow(img2_eq, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.subplot(3,2,6)
plt.bar(range(256), histeq_img2)
plt.xlabel('Intensity')
plt.ylabel('Frequency')


# In[62]:


#Create a random image to [0,9] span, with (5x5) size
img_f = np.array(np.random.randint(0,10,[5,5]))
print(img_f)


# In[63]:


#Create a 3x3 filter
filter_w = np.array([[0.125, 0.25, 0.], [0.125, 0.5, 0.], [0., 0., 0.]])
print(filter_w)


# In[64]:


#This function helps to calculate the size of the filtered image
def calculate_target_size(img_size: int, filter_size: int) -> int:
    num_pixels = 0
    
    for i in range(img_size):
        sum_pixels = i + filter_size
        
        if (sum_pixels <= img_size):
            num_pixels += 1
            
    return num_pixels


# In[65]:


#This function is used to achieve the convolution between an img and a filter
def image_conv(img_f: np.array, filter_w: np.array) -> np.array:
    
    #Calculate the sizes
    conv_size = calculate_target_size(
        img_size=img_f.shape[0],
        filter_size=filter_w.shape[0])
    
    filter_size = filter_w.shape[0]
    
    img_size = img_f.shape[0]
    
    #Initiate the 2d convolved image with zeros
    convolved_img = np.zeros(shape=(conv_size, conv_size))
    
    k = l = 0
    for m in range(conv_size):
        
        for n in range(conv_size):
            
            sum_all = 0
            
            for i in range(filter_size):
                sum_rows = 0
                for j in range(filter_size):
                    mult = img_f[i+k][j+l]*filter_w[i][j]
                    sum_rows += mult
                if(j == filter_size-1):
                    sum_all += sum_rows
                    
            convolved_img[m][n] = sum_all
            if(l == (conv_size-1)):
                l=0
                k+=1
                    
            else:
                sum_all = 0
                l+=1

    return convolved_img


# In[66]:


#This function is used for zero padding, the result will make a black frame around the picture
def padding(img_f: np.array) -> np.array:
    
    new_size = img_f.shape[0] + 2
    padded_img = np.array(np.zeros(shape=(new_size, new_size)))
    
    for i in range(new_size):
        for j in range(new_size):
            if(i == 0 or j == 0 or (i == new_size-1) or (j == new_size-1)):
                padded_img[i][j] = 0
            else:
                padded_img[i][j] = img_f[i-1][j-1]
    return padded_img


# In[67]:


padded_img = padding(np.array(img_f)).astype(np.uint8)

print(img_f)
print("\n-------------------------------\n")
print(padded_img)


# In[68]:


#Implemantation of the filtering

#Create a flipped filter in the x and y axis
flip_x_y = np.flip(np.flip(filter_w, 0), 1)

#Convolve the padded image with the flipped filter
img_filtered_w = image_conv(img_f=np.array(padded_img), filter_w=np.array(flip_x_y)).astype(np.uint8)

print(img_f)
print(img_f.shape[0])
print("\n-------------------------------\n")
print(img_filtered_w)
print(img_filtered_w.shape[0])


# In[69]:


#Gradient noise image implementation

#Load the image
img_f_2 = imageio.imread('images/gradient_noise.png')

#Pad the image
padded_img_2 = padding(np.array(img_f_2)).astype(np.uint8)

#Create the averaging filter
averaging_filter = np.array([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]])

#Convolve the image
img_avg_filtered = image_conv(img_f=np.array(padded_img_2), filter_w=np.array(averaging_filter)).astype(np.uint8)

print(img_f_2)
print(img_f_2.shape[0])
print("\n-------------------------------\n")
print(img_avg_filtered)
print(img_avg_filtered.shape[0])


# In[70]:


#Gradient noise image visualization
plt.figure(figsize=(14,14))
plt.subplot(221)
plt.imshow(img_f_2, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.subplot(222)
plt.imshow(img_avg_filtered, cmap='gray', vmin=0, vmax=255)
plt.axis('off')