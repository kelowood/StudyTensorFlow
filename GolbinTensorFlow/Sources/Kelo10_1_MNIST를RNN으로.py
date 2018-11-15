#!/usr/bin/env python
# coding: utf-8

# In[1]:


# RNN(Recurrent Neural Network : 순환신경망)
# 상태가 고정된 데이터 (이미지같은)를 처리하는 타 신경망보다
# 자연어 혹은 음성처리같은 순서가 있는 데이터를 처리하는데 강점을 가진 신경망이다.
# 이번 코드는 MNIST를 RNN 방식으로 학습하고 예측하는 모델을 만들어 보자.


# In[24]:


# 28*28의 이미지에 위에서 아래 순서대로 28픽셀 한줄씩을 내려가면서 데이터를 입력받도록 한다.

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

tf.reset_default_graph()


# In[25]:


hp_leaning_rate = 0.001
hp_total_epoch = 30
hp_batch_size = 128

hp_n_input = 28
hp_n_step = 28
hp_n_hidden = 128
hp_n_class = 10

# 플레이스 홀더의 정의.
# 여기서의 입력값 X에서 hp_n_step 이란 차원을 하나 더 추가하였다.
# RNN은 순서가 있는 데이터를 다루기 때문에 한번에 입력 받을 개수와 
# 총 몇단계로 이루어진 데이터를 받을지를 설정해야 한다.
# 이때문에 가로 픽셀수를 hp_n_input으로, 세로 픽셀수를 단계수인 hp_n_step으로 설정하였다.
X = tf.placeholder(tf.float32, [None, hp_n_step, hp_n_input])
Y = tf.placeholder(tf.float32, [None, hp_n_class])

W = tf.Variable(tf.random_normal([hp_n_hidden, hp_n_class]))
b = tf.Variable(tf.random_normal([hp_n_class]))


# In[26]:


# hp_n_hidden (은닉층 수)의 출력값을 갖는 RNN 셀을 생성한다.
# 텐서플로우에서는 RNN 셀을 아래와 같이 쉽게 생성할 수 있다.
cell = tf.nn.rnn_cell.BasicRNNCell(hp_n_hidden)

# 셀 생성 함수는 여러 종류가 있는데 
# 위의 BasicRNNCell뿐만 아니라 BasicLSTMCell, GRUCell 등의 다양한 방식이 존재한다.

# 다음으로는 dynsmic_rnn 함수를 이용하여 RNN 신경망을 만든다.
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
# 위와 같이 RNN셀과 입력값, 입력값의 자료형만 넣어주면 간단하게 신경망을 만들수 있다.
# (원래대로 코드를 저수준으로 구현해야 한다면 상당히 복잡하다.)

# states = tf.zeros(hp_batch_size)
# for i in range(hp_n_step)
#     outputs, states = cell(X[[:, i]], states)
# 위 코드와 같이 한단계 학습하고 상태를 저장한 후, 그 상태를 다음 단계의 입력 상태로 해서 다시 학습한다.
# 이렇게 주어진 단계를 반폭하여 상태를 전파해가면서 출력값을 만드는 구조이다.
print(outputs)


# In[27]:


# 이제는 RNN에서 나온 출력 값을 가지고 최종 출력을 만들 차례다.
# 결과값은 One-hot 인코딩 형태이므로 손실 함수로 tf.nn.softmax_cross_entropy_with_logits를 사용하도록 한다.
# 근데 RNN에서 나온 출력값은 [hp_batch_size, hp_n_step, hp_n_hidden]의 형태이다.

# 그러므로 아래같은 형태로 행렬을 바꾸고자 한다.
# [hp_batch_size, hp_n_step, hp_n_hidden] -> [hp_n_step, hp_batch_size, hp_n_hidden]
outputs = tf.transpose(outputs, [1, 0, 2])
print(outputs)

# 그리고 맨앞의 hp_n_step 차원을 제거하고, 마지막 단계의 결과값만 가져오도록 한다.
# [hp_n_step, hp_batch_size, hp_n_hidden] -> [hp_batch_size, hp_n_hidden]
outputs = outputs[-1]
print(outputs)

# 이제 인공신경망의 기본 수식인 y = X * W + b  (dense)를 이용하여 최종 결과를 만들자
model = tf.matmul(outputs, W) + b
print(model)


# In[28]:


# 손실 함수를 작성한다.

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        logits=model,
        labels=Y))
optimizer = tf.train.AdamOptimizer(hp_leaning_rate).minimize(cost)


# In[29]:


# 세션을 만들고 학습을 수행한다.
sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples / hp_batch_size)

for epoch in range(hp_total_epoch):
    total_cost = 0
    
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(hp_batch_size)
        batch_xs = batch_xs.reshape(hp_batch_size, hp_n_step, hp_n_input)
        
        _, cost_val = sess.run(
            [optimizer, cost],
            feed_dict={X: batch_xs, Y: batch_ys})
        
        total_cost += cost_val
        
    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))
    
print('최적화 완료!')


# In[31]:


# 학습결과를 확인해보자.

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

test_batch_size = len(mnist.test.images)
test_xs = mnist.test.images.reshape(test_batch_size, hp_n_step, hp_n_input)
test_ys = mnist.test.labels

print('정확도:', sess.run(accuracy, feed_dict={X: test_xs, Y: test_ys}))

sess.close()

