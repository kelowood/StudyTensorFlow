#!/usr/bin/env python
# coding: utf-8

# In[1]:


# matplotlib 라이브러리는 시각화를 위해서 그래프를 쉽게 그릴 수 있도록 해주는 파이썬 라이브러리이다.
# 여기에서는 이 라이브러리를 통해 학습결과를 손글씨 이미지로 확인해보는 예제를 만들어보자.


# In[10]:


import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data as mnist_input_data

# 추가로 matplotlib의 pyplot 모듈을 임포트한다.
import matplotlib.pyplot as plt



mnist = mnist_input_data.read_data_sets("./mnist/data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784]) # 784 픽셀
Y = tf.placeholder(tf.float32, [None, 10]) # 10종류의 숫자


# In[15]:


# 신경망 설정

dropoutRate = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
b1 = tf.Variable(tf.zeros([256]))
L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))
L1 = tf.nn.dropout(L1, dropoutRate)

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
b2 = tf.Variable(tf.zeros([256]))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2))
L2 = tf.nn.dropout(L2, dropoutRate)

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
b3 = tf.Variable(tf.zeros([10]))
model = tf.add(tf.matmul(L2, W3), b3)


# In[16]:


# 비용함수, 최적화 함수 설정
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=model,
        labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


# In[17]:


# 세션 시작
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(30):
    total_cost = 0
    
    for i in range(total_batch):
        
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        
        _, cost_val = sess.run(
            [optimizer, cost], 
            feed_dict={X: batch_xs, Y: batch_ys, dropoutRate: 0.8})
        
        total_cost += cost_val
    
    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))
    
print('최적화 완료!')


# In[18]:


# 텍스트 결과 확인

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accurary = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:', sess.run(
    accurary, 
    feed_dict={
        X: mnist.test.images, 
        Y: mnist.test.labels,
        dropoutRate: 1}))


# In[19]:


# matplotlib 로 결과 확인

# 모델을 수행한 결과를 labels에 넣는다.
labels = sess.run(
    model,
    feed_dict={
        X: mnist.test.images,
        Y: mnist.test.labels,
        dropoutRate: 1
    })

# 손글씨를 출력할 그래프를 준비한다.
fig = plt.figure()

# 테스트 데이터의 첫 번째부터 열번째까지의 이미지와 예측 값을 출력한다.
for i in range(10):
    # 2행 5열의 그래프를 만들고, i + 1번째에 숫자 이미지를 출력한다.
    subplot = fig.add_subplot(2, 5, i + 1)
    
    # 이미지를 깨끗하게 출력하기 위해 x와 y의 눈금은 출력하지 않는다.
    subplot.set_xticks([])
    subplot.set_yticks([])
    
    # 출력한 이미지 위에 예측한 숫자를 출력한다.
    # np.argmax는 tf.argmax와 같은 기능의 함수이다.
    # 결과값인 labels의 i번째 요소가 원-핫 인코딩 형식으로 되어 있으므로,
    # 해당 배열에서 가장 높은 값을 가진 인덱스를 예측한 숫자로 출력한다.
    subplot.set_title('%d' % np.argmax(labels[i]))
    
    # 1차원 배열로 되어있는 i번째 이미지 데이터를 
    # 28*28 형식의 2차원 배열로 변형하여 이미지 형태로 출력한다.
    # cmap 파라미터를 통해 이미지를 그레이 스케일로 출력한다.
    subplot.imshow(
        mnist.test.images[i].reshape((28, 28)),
        cmap=plt.cm.gray_r)
    
# 마지막으로 그래프를 화면에 표시한다.
plt.show()



sess.close()


# In[ ]:




