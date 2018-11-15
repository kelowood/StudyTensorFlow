#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf

# 텐서플로우 예제 mnist 튜토리얼에서 input_data 라는 클래스를 가져온다.
from tensorflow.examples.tutorials.mnist import input_data as mnist_input_data

# 이 코드에서 mnist 정보를 다운로드 받고 레이블 데이터를 원-핫 인코딩 방식으로 읽어들인다.
mnist = mnist_input_data.read_data_sets("./mnist/data/", one_hot=True)


# In[4]:


# MNIST 의 손글씨 이미지는 28*28 픽셀(784)로 이루어져 있다.
# 그리고 레이블은 0부터 9까지의 숫자이므로 10개의 분류로 나눌 수 있다.
# 그러므로 입력과 출력 플레이스 홀더는 아래와 같이 구성할 수 있다.

X = tf.placeholder(tf.float32, [None, 784]) # 784 픽셀
Y = tf.placeholder(tf.float32, [None, 10]) # 10종류의 숫자

# 보통 데이터를 적당한 크기로 나누어서 학습 시키는 것을 우리는 미니배치(minibatch)라 부른다.
# X, Y 텐서의 첫번째 차원이 None으로 지정되어 있다. 
# 이 자리에는 한번에 학습시킬 MNIST 이미지의 개수를 지정하는 값이 들어간다. 즉 배치크기가 지정되는 것이다.
# 원하는 크기를 명시해주는 방법도 있지만, 학습할 데이터를 바꿔가면서 실험을 할때는 None으로 해주면
# 텐서플로우가 알아서 계산한다.


# In[5]:


# 이제 아래와 같은 형태의 신경망을 만들어보고자 한다.
# 784 (특징 갯수) =>
# 256 (첫번째 은닉층 뉴런 갯수) =>
# 256 (두번째 은닉층 뉴런 갯수) =>
# 10 (결과값 분류 갯수)

W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
b1 = tf.Variable(tf.zeros([256]))
L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
b2 = tf.Variable(tf.zeros([256]))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2))

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
b3 = tf.Variable(tf.zeros([10]))
model = tf.add(tf.matmul(L2, W3), b3)

# 여기서 책과는 다르게 편향도 추가해 보았다.


# In[6]:


# 이제 손실값을 처리해보자

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        logits=model,
        labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


# In[7]:


# 세션 시작
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# In[8]:


# 이제 학습 진행 로직을 진행해보자

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)
# mnist에서는 데이터 수가 매우 크기 땜누에 학습에 미니배치를 사용한다.
# 미니 배치의 크기는 100이고, 학습 데이터 총 갯수인 mnist.train.num_examples를 배치 크기로 나누면
# 미니배치가 총 몇개인지를 알수 있다.

# MNIST 데이터 전체를 학습하는 일을 총 15번 반복한다.
# 여기에서 학습데이터 전체를 한바퀴 도는 것을 에포크(epoch)라 부른다.
for epoch in range(15):
    total_cost = 0
    
    for i in range(total_batch):
        # 반복문 안에서 배치 사이즈 만큼의 배치를 가져온다.
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        
        _, cost_val = sess.run(
            [optimizer, cost], 
            feed_dict={X: batch_xs, Y: batch_ys})
        
        total_cost += cost_val
    
    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))
    
print('최적화 완료!')


# In[9]:


# 이제 학습결과가 잘 나오는지 확인해볼 시간이다.

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accurary = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:', sess.run(accurary, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

