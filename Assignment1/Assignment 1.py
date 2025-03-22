#!/usr/bin/env python
# coding: utf-8

#KAPAKOS GEORGIOS 03165

# In[2]:


import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt


# In[25]:

#calculate scale
scale = 255/(np.log2(256))

#plot the log2 function
r = np.arange(0,256)
y = scale*np.log2(1+r)
plt.plot(r, y)


# In[26]:

#initialize the array to all zeros
img2_log2 = np.zeros(shape = (size2[0], size2[1]))

#transform and scale
img2_log2 = scale * np.log2(img2+1)

#make int the final values of the array
img2_log2 = np.array(img2_log2, dtype = np.uint8)


# In[27]:

#img2
plt.figure(figsize = (15, 15))
plt.subplot(121)
plt.imshow(img2, cmap = "gray")
plt.axis('off')

#log2(img2)
plt.subplot(122)
plt.imshow(img2_log2, cmap = "gray")
plt.axis('off')


# In[28]:

#Plot the Sigmoid function
x = np.linspace(-10, 10, 100)
z = 1/(1 + np.exp(-x))
  
plt.plot(x, z)
plt.xlabel("x")
plt.ylabel("Sigmoid(X)")
  
plt.show()


# In[36]:

#initialize img2_sigmoid array
img2_sigmoid = np.zeros(shape = (size2[0], size2[1]))

#select a scale to divide img2
img2_scaled = img2/255

#transformation
img2_sigmoid = 1/(1 + np.exp(-1 * (img2_scaled + 1))*(100))

#make the array int andincrease  the span to [0, 255]
img2_sigmoid = np.array(img2_sigmoid * 255, dtype = np.uint8)


# In[37]:

#img2
plt.figure(figsize = (15, 15))
plt.subplot(121)
plt.imshow(img2, cmap = "gray")
plt.axis('off')

#img2 simgoid
plt.subplot(122)
plt.imshow(img2_sigmoid, cmap = "gray")
plt.axis('off')


# In[38]:


print(img2_sigmoid)


# In[ ]:




