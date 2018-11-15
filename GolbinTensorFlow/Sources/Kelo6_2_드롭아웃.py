#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 과적합(Overfitting)
# 과적합이란 학습 데이터를 사용해서 합습했을시 결과가 잘 맞지만, 
# 학습 데이터에만 너무 맞춰져있어서 그 외의 데이터에는 잘 맞지 않는 현상을 뜻한다.
# 쉽게말해 학습 데이터들에 대해서만 예측을 잘 하고, 정작 실제 데이터는 예측을 못하는 것을 말한다.

# 드롭아웃(Dropout)이란?
# 과적합 현상을 해결하기 위한 방법론이다.
# 방법은 상당히 단순한 편으로 학습할때 전체 신경망중에서 일부만을 사용하도록 하는 것이다.
# 또한 학습 회차마다 신경망을 다르게 설정하도록 한다.


# In[2]:


# 앞의 6.1에서 작업하였던 코드에 적용하여 보자.
# 신경망을 만들기 전까지는 이전 코드와 같다.

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_input_data

# 이 코드에서 mnist 정보를 다운로드 받고 레이블 데이터를 원-핫 인코딩 방식으로 읽어들인다.
mnist = mnist_input_data.read_data_sets("./mnist/data/", one_hot=True)

# MNIST 의 손글씨 이미지는 28*28 픽셀(784)로 이루어져 있다.
# 그리고 레이블은 0부터 9까지의 숫자이므로 10개의 분류로 나눌 수 있다.
# 그러므로 입력과 출력 플레이스 홀더는 아래와 같이 구성할 수 있다.

X = tf.placeholder(tf.float32, [None, 784]) # 784 픽셀
Y = tf.placeholder(tf.float32, [None, 10]) # 10종류의 숫자


# In[3]:


# 784 (특징 갯수) => 256 (첫번째 은닉층 뉴런 갯수) => 256 (두번째 은닉층 뉴런 갯수) => 10 (결과값 분류 갯수)
# 이제 여기서 중요한 점이 dropout 이라는 함수를 추가로 사용하였다는 점이다.

dropoutRate = tf.placeholder(tf.bool)

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

# tf.nn.dropout(L1, dropoutRate) 에서 dropoutRate은 사용할 뉴런의 비율을 뜻한다. 
# 만약 dropoutRate가 0.8 이었다면 80%의 뉴런을 사용하는 것이다.

# 여기서 dropoutRate라는 플레이스 홀더를 사용한 이유는
# 학습이 끝나고 값 예측을 할때에는 신경망 전체를 사용해야 하기 때문이다.
# 그런고로 학습할때에는 0.8 값을 넣고, 예측을 할때는 1을 넣도록 한다.


# In[4]:


cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        logits=model,
        labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


# In[10]:


# 세션 시작
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

# 여기에서 드롭아웃 기법을 적용한 뒤 학습을 진행하면 학습이 느리게 진행된다.
# 그렇기 때문에 에포크를 2배인 30으로 늘려서 더 많이 학습해보도록 하자.

for epoch in range(30):
    total_cost = 0
    
    for i in range(total_batch):
        # 반복문 안에서 배치 사이즈 만큼의 배치를 가져온다.
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        
        _, cost_val = sess.run(
            [optimizer, cost], 
            feed_dict={X: batch_xs, Y: batch_ys, dropoutRate: 0.8})
        
        total_cost += cost_val
    
    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))
    
print('최적화 완료!')


# In[11]:


# 이제 학습결과가 잘 나오는지 확인해볼 시간이다.

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accurary = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:', sess.run(
    accurary, 
    feed_dict={
        X: mnist.test.images, 
        Y: mnist.test.labels,
        dropoutRate: 1}))

sess.close()


# In[ ]:




