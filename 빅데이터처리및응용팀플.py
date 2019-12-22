
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from tensorflow.examples.tutorials.mnist import input_data


# In[2]:


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In[3]:


X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 2])            #2개의 출력 노드


# In[4]:


W1 = tf.Variable(tf.random_normal([5, 5, 1, 20], stddev=0.01)) 
# 5*5 20개 convolution 필터 -> (?, 28, 28, 20)
L1 = tf.nn.conv2d(X_img, W1, strides = [1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# pooling -> (?, 14, 14, 20)


# In[5]:


print(L1) # 2번째 컨볼루션의 입력으로 사용


# In[6]:


W2 = tf.Variable(tf.random_normal([5, 5, 20, 40], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides = [1, 1, 1, 1], padding='SAME') #14*14*40
L2 = tf.nn.relu(L2)
L2 = tf.nn.avg_pool(L2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
# 4*4*40


# In[7]:


print(L2)


# In[8]:


W3 = tf.Variable(tf.random_normal([5, 5, 40, 60], stddev=0.01))
L3 = tf.nn.conv2d(L2, W3, strides = [1, 1, 1, 1], padding = 'SAME') #4*4*60
L3 = - tf.nn.relu(L3)
L3 = - tf.nn.max_pool(L3, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='SAME')


# In[9]:


print(L3)  #4차원


# In[10]:


L3 = tf.reshape(L3, [-1, 4*4*60]) #2차원


# In[11]:


W4 = tf.get_variable("W4", shape=[4*4*60, 1000],initializer = tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([1000]))

L4 = tf.matmul(L3, W4) + b4
L4 = tf.nn.relu(L4)
L4 = tf.nn.dropout(L4, keep_prob = 0.5)


# In[12]:


print(L4)


# In[13]:


W5 = tf.get_variable("W5", shape=[1000, 100], initializer = tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([100]))
L5 = tf.matmul(L4, W5) + b5
L5 = tf.nn.relu(L5)
L5 = tf.nn.dropout(L5, keep_prob = 0.7)


# In[14]:


print(L5)


# In[15]:


W6 = tf.get_variable("W6", shape=[100, 2], initializer = tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([2]))
L6 = tf.matmul(L5, W6) + b6


# In[16]:


print(L6)


# In[17]:


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=L6, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)


# In[18]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[19]:


print('Learning started. It takes sometimes')
for epoch in range(10):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / 200)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(200)
        batch_ys_ = np.ones([len(batch_ys),2])
        for i in range(len(batch_ys)): 
            if 1 in batch_ys[i][0:5]:
                batch_ys_[i] = [1,0]          #good
            else:
                batch_ys_[i] = [0,1]          #bad
        feed_dict = {X: batch_xs, Y: batch_ys_}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
    print('Epoch: ', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))


# In[20]:


print('Learning Finished')


# In[21]:


correct_prediction = tf.equal(tf.argmax(L6, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
mnist.test.labels_ = np.ones([len(mnist.test.labels),2])
for i in range(len(mnist.test.labels)): 
    if 1 in mnist.test.labels[i][0:5]:
        mnist.test.labels_[i] = [1,0]      #good
    else:
        mnist.test.labels_[i] = [0,1]      #bad
    feed_dict = {X: mnist.test.images, Y: mnist.test.labels_}
mnist.test.labels_list = mnist.test.labels_.tolist()

print('Accuarcy:', sess.run(accuracy, feed_dict=feed_dict))
print('Good:', (mnist.test.labels_list.count([1,0])/len(mnist.test.labels_list))*100, '%')
print('Bad:',(mnist.test.labels_list.count([0,1])/len(mnist.test.labels_list))*100, '%' )

