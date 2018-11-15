#!/usr/bin/env python
# coding: utf-8

# In[6]:


# 7.2장에서 만들었던 CNN에 대하여 layers 모듈을 이용하여 더 쉽게 구현할 수 있다.
# 우선 플레이스 홀더 정의까지는 이전장과 똑같이 구현한다.

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

# [이미지갯수, X축, Y축, 채널갯수]
# 채널갯수란 색상 갯수를 뜻하며, 1인 이유는 색상이 딱 한개 (흑색) 들어가기 때문이다.
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])
is_training = tf.placeholder(tf.bool)


# In[7]:


# 컨볼루션 계층 + 풀링 게층의 구현 부분이다.

# 본래 7.2장에서는 아래의 코드였다.
# W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
# L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
# L1 = tf.nn.relu(L1)
# L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 위의 것을 아래 형태로 간단하게 바꿀 수 있다.
L1 = tf.layers.conv2d(
    X, 
    32, 
    [3, 3], 
    activation=tf.nn.relu, 
    padding='SAME')
print ("L1(After Convolution) : ", L1)

L1 = tf.layers.max_pooling2d(
    L1,
    [2, 2],
    [2, 2],
    padding='SAME')
print ("L1(After MaxPooling)  : ", L1)

L1 = tf.layers.dropout(L1, 0.7, is_training)
print ("L1(After Dropout)     : ", L1)

# 위의 함수에 대하여 하나하나 설명해보자

# tf.layers.conv2d()
# 2차원 컨볼루션 계층을 만드는 텐서
# 매개변수
# inputs : 텐서 입력
# filters : 컨볼루션 계층을 수행할 필터의 갯수.
# kernal_size : 커널(필터)의 사이즈 int 값 두개 형태로 [높이, 너비]로 구성된 배열 값을 넣는다.
# strides : 스트라이드 값 [높이, 너비] 형태의 배열값을 넣으며, 해당 값 만큼 윈도우를 움직인다.
# activation : 활성화 함수. 적용할 활성화 함수를 명시한다.
# padding : tf.nn.conv2d()의 padding 매개변수와 동일.
# 리턴값 : 컨볼루션 계층 수행 텐서

# tf.layers.max_pooling2d()
# 2차원 최대값 풀링을 수행하는 텐서
# 매개변수
# inputs : 입력 텐서. 4랭크를 가진 텐서여야 한다.
# pool_size : 풀의 사이즈 [높이, 너비] 형태의 int 배열로 구성된다.
# strides : 스트라이드값 [높이이동값, 너비이동값] 형태의 int 배열로 구성된다.
# padding : tf.nn.max_pool()의 padding 매개변수와 동일
# 리턴값 : 풀링 계층 수행 텐서

# 위의 형태와 같이 편하게 코드 작성이 가능하다.

L2 = tf.layers.conv2d(
    L1, 
    64, 
    [3, 3], 
    activation=tf.nn.relu,
    padding='SAME')
print ("L2(After Convolution) : ", L2)

L2 = tf.layers.max_pooling2d(
    L2,
    [2, 2],
    [2, 2],
    padding='SAME')
print ("L2(After MaxPooling)  : ", L2)

L2 = tf.layers.dropout(L2, 0.7, is_training)
print ("L2(After Dropout)     : ", L2)


# In[9]:


# 완전 연결 계층 부분의 구현 부분이다.

# 기존코드는 아래와 같다.
# W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 256], stddev=0.01))
# L3 = tf.reshape(L2, [-1, 7 * 7 * 64])
# L3 = tf.matmul(L3, W3)
# L3 = tf.nn.relu(L3)
# L3 = tf.nn.dropout(L3, keep_prob)

# 위의 것을 아래처럼 간단히 바꿀 수 있다.
L3 = tf.contrib.layers.flatten(L2)
print ("L3(After flatten)     : ", L3)

L3 = tf.layers.dense(L3, 256, activation=tf.nn.relu)
print ("L3(After dense)       : ", L3)

L3 = tf.layers.dropout(L3, 0.5, is_training)
print ("L3(After Dropout)     : ", L3)

# tf.contrib.layers.flatten()
# 텐서를 평평하게 만드는 함수.
# 쉽게 말하자면 [배치갯수, k]의 2랭크 텐서로 Reshape해주는 함수다.
# 매개변수
# inputs : 입력 텐서. [배치크기, ...] 의 형태로 이루어져 있어야 한다.

# tf.layers.dense()
# 완전 연결 계층을 수행하는 함수
# 매개변수
# inputs : 텐서 입력
# units : integer 혹은 long 값. 출력되는 뉴런 갯수
# activation : 활성화 함수


# In[11]:


# 마지막으로 최종 출력 계층이다.
# tf.layers.dense()를 활용하자.

model = tf.layers.dense(L3, 10, activation=None)
print ("model                 : ", model)


# In[13]:


# 나머지는 7.2때와 거의 동일하다.

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        logits=model,
        labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


# In[15]:


# 세션 시작
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(15):
    total_cost = 0
    
    for i in range(total_batch):
        # 반복문 안에서 배치 사이즈 만큼의 배치를 가져온다.
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        
        # 여기서 입력값의 셰이프는 [배치갯수, 28 * 28] 이었던 것을
        # [배치갯수, 28, 28, 1]로 바꿔주어야 한다.
        # 참고로 batch_xs.reshape()는 텐서가 아니므로 즉시 실행된다.
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)
        
        _, cost_val = sess.run(
            [optimizer, cost], 
            feed_dict={X: batch_xs, Y: batch_ys, is_training: True})
        
        total_cost += cost_val
    
    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))
    
print('최적화 완료!')


# In[16]:


# 학습결과를 확인해보자.

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accurary = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# 여기에서도 테스트 이미지에 대하여 reshape를 해주어야 한다.
print('정확도:', sess.run(
    accurary, 
    feed_dict={
        X: mnist.test.images.reshape(-1, 28, 28, 1), 
        Y: mnist.test.labels,
        is_training: False}))

sess.close()

