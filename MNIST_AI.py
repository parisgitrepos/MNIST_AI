#!/usr/bin/env python
# coding: utf-8

# In[2]:


from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


# In[3]:


(train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()


# In[4]:


train_x = train_x/255
test_x = test_x/255


# In[5]:


train_x = train_x.reshape(60000, 28, 28, 1)
test_x = test_x.reshape(10000, 28, 28, 1)


# In[6]:


model = keras.Sequential()


# In[7]:


conv1 = keras.layers.Conv2D(input_shape = (28, 28, 1), kernel_size = 3, strides = 2, filters = 5, data_format = 'channels_last', activation = 'relu')
pool1 = keras.layers.MaxPool2D(pool_size = 3, strides = 2)
flatten = keras.layers.Flatten()
dense1 = keras.layers.Dense(units = 128, activation = 'relu')
dense2 = keras.layers.Dense(units = 64, activation = 'relu')
dense3 = keras.layers.Dense(units = 32, activation = 'relu')
dense4 = keras.layers.Dense(units = 16, activation = 'relu')
dense5 = keras.layers.Dense(units = 10, activation = 'softmax')


# In[8]:


model.add(conv1)
model.add(pool1)
model.add(flatten)
model.add(dense1)
model.add(dense2)
model.add(dense3)
model.add(dense4)
model.add(dense5)


# In[9]:


model.compile(optimizer = keras.optimizers.Adam(lr = 0.001), loss = 'sparse_categorical_crossentropy')


# In[12]:


model.fit(x = train_x, y = train_y, epochs = 5, batch_size = 16)


# In[237]:


plt.imshow(test_x[39].reshape(28, 28)*255)


# In[238]:


testx = test_x[39].reshape(1, 28, 28, 1)


# In[239]:


a = model.predict(testx)


# In[240]:


print(a[0])


# In[13]:


model.save('MNIST_Number_AI.h5')


# In[ ]:




