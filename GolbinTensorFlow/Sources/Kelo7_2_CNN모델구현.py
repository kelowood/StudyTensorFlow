#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


# In[3]:


# [이미지갯수, X축, Y축, 채널갯수]
# 채널갯수란 색상 갯수를 뜻하며, 1인 이유는 색상이 딱 한개 (흑색) 들어가기 때문이다.
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)


# In[4]:


# CNN 계층 구성

# tt.nn.conv2d() 의 매개변수
# 이 함수는 2D 컨볼루션 연산을 한다.
# 변수1 : input -> 입력값으로 4차원 텐서값이다.
#         [배치, 높이값, 너비값, 채널값]
# 변수2 : filter -> 필터(혹은 커널) 값으로 4차원 텐서값이다.
#         [필터높이값, 필터너비값, 입력채널값, 출력채널값]
#         여기서 출력채널값은 해당 갯수의 커널을 가진 컨볼루션 계층을 만들겠다는 얘기다.
# 변수3 : strides -> 1차원 4길이의 int 리스트. 스트라이드 값의 설정이다.
#         [1, 가로움직임량, 세로움직임량, 1]
# 변수4 : padding -> SAME으로 설정하면 커널 슬라이딩시 이미지의 가장 외곽에서 한칸 밖으로 움직인다.

# 먼저 3*3 크기의 32개의 커널을 가진 컨볼루션 계층을 만든다.
# 또한 relu 활성화 함수를 적용시킨다.
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)

print('W1              : ', W1)
print('L1(Before pool) : ', L1)

# tt.nn.max_pool()의 매개변수
# 입력값에 대한 최대값 풀링을 적용시킨다. 커널 사이즈 내에서 가장 큰값이 적용된다.
# 변수1 : value -> 입력값으로 4차원 텐서이다. 기본적으로 NHWC 방식이 디폴트이다.
#         [배치, 높이값, 너비값, 채널값]
# 변수2 : ksize -> 1차원 4길이의 int 리스트. 윈도우 크기를 지정한다.
# 변수3 : strides -> 1차원 4길이의 int 리스트, 스트라이드 값의 설정이다.
# 변수4 : padding -> SAME으로 설정하면 커널 슬라이딩시 이미지의 가장 외곽에서 한칸 밖으로 움직인다.

# 풀링 계층
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

print('L1(After pool)  : ', L1)

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)

print('W2              : ', W2)
print('L2(Before pool) : ', L2)

L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

print('L2(After pool)  : ', L2)

# 출력되는 값의 셰이프를 잘 확인해 보자.
# 최종 출력 셰이프는 [배치크기, 7, 7, 64] 가 된다.


# In[5]:


# 세번째 부터는 추출한 가중치 적용을 하면서 특징의 차원을 줄이는 것을 해보도록 한다.

# tf.reshape()
# 입력값에 해당하는 텐서의 셰이프를 재구성하는 함수
# 매개변수1 : tensor -> 셰이프를 재구성하고자 하는 텐서
# 매개변수2 : shape -> 재구성하고자 하는 Shape 값
#             만약 -1값이 들어갈 시에는 기존 텐서 셰이프의 차원의 값으로 들어간다.

W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 256], stddev=0.01))
L3 = tf.reshape(L2, [-1, 7 * 7 * 64])

print('W3                : ', W3)
print('L3(After reshape) : ', L3)

L3 = tf.matmul(L3, W3)
L3 = tf.nn.relu(L3)
L3 = tf.nn.dropout(L3, keep_prob)

print('L3(After dropout) : ', L3)

# [배치크기, 7, 7, 64] 셰이프였던 텐서가 reshape를 거치면서 [배치크기, 7 * 7 * 64]로 2차원 텐서으로 바뀌었다.
# 즉 1차원 계층으로 줄은 것이다.
# 즉 256개의 뉴런으로 연결하는 신경망이 구축된 것이다. 
# 위 같이 모든 뉴런과 상호연결된 계층을 완전 연결 계층 (fully connected layer)라고 한다.
# 드롭아웃까지 거치고 나서는 [배치크기, 256] 으로 바뀌게 된다.


# In[7]:


# 마지막 계층은 은닉층 L3를 최종 출력값 10개로 만드는 작업을 한다.
W4 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L3, W4)

print('W4                : ', W4)
print('model             : ', model)


# In[8]:


# 이제 손실함수와 AdamOptimizer 최적화 함수를 정의해보자

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=model,
        labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


# In[9]:


# 세션 시작 후 학습 처리는 6.2때와 매우 흡사하나 다른점이 있다.

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
            feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})
        
        total_cost += cost_val
    
    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))
    
print('최적화 완료!')


# In[12]:


# 학습결과를 확인해보자.

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accurary = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# 여기에서도 테스트 이미지에 대하여 reshape를 해주어야 한다.
print('정확도:', sess.run(
    accurary, 
    feed_dict={
        X: mnist.test.images.reshape(-1, 28, 28, 1), 
        Y: mnist.test.labels,
        keep_prob: 1}))

sess.close()


# In[ ]:




