#!/usr/bin/env python
# coding: utf-8

# In[6]:


# 이번에는 신경망을 둘 이상으로 구성한 심층 신경망, 즉, 딥러닝을 구현해보고자 한다.
# 이번에 만드는 코드는 4.2 코드와 매우 유사하나 가중치와 편향이 하나 더 추가된다.

import tensorflow as tf
import numpy as np

# 데이터 구성은 4.2와 동일하다.

# 학습에 사용할 데이터를 정의한다.
# [털, 날개]
x_data = np.array([
    [0, 0], 
    [1, 0], 
    [1, 1], 
    [0, 0], 
    [0, 0], 
    [0, 1]
])

# 기타 = [1, 0, 0]
# 포유류 = [0, 1, 0]
# 조류 = [0, 0, 1]

y_data = np.array([
    [1, 0, 0], # 기타
    [0, 1, 0], # 포유류
    [0, 0, 1], # 조류
    [1, 0, 0], # 기타
    [1, 0, 0], # 기타
    [0, 0, 1]  # 조류
])

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


# In[9]:


# 가중치와 편향은 2개를 만든다.
# 근데 또 다른 부분중 하나가 변수 생성의 셰이프 수가 다른걸 알 수 있다.
# 첫번째 신경망의 셰이프가 [2, 10]이다. 특징의 수가 2이고, 10은 바로 은닉층의 갯수를 의미한다.
# 은닉층은 하이퍼파라미터이며, 개발자가 적절한 수를 정할 수 있다. 이 값은 신경망 성능에 영향을 미친다.

# 신경망 제1층
W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.)) # [특징수, 은닉층뉴런수]
b1 = tf.Variable(tf.zeros([10])) # [은닉층뉴런수]

# 신경망 제2층
W2 = tf.Variable(tf.random_uniform([10, 3], -1., 1.)) # [은닉층뉴런수, 분류수]
b2 = tf.Variable(tf.zeros([3])) # [분류수]

# 신경망 제 1층 : [6, 2] * [2, 10] => [6, 10] + [10] => [6, 10]
# 6은 입력 갯수
# 2는 특징 갯수
# 10은 은닉층 갯수
 
# 신경망 제 2층 : [6, 10] * [10, 3] => [6, 3] + [3] => [6, 3]
# 6은 입력 갯수
# 10은 은닉층 갯수
# 3은 분류 갯수


# In[10]:


# 이제 신경망1층을 정의한다.
# 신경망 1층에서는 활성화 함수를 적용하였다.
L1 = tf.add(tf.matmul(X, W1), b1)
L1 = tf.nn.relu(L1)

# 신경망 2층을 정의하여 출력층을 만들도록 한다.
model = tf.add(tf.matmul(L1, W2), b2)

# 4.2의 신경망모델에서는 출력층에 활성화 함수 (ReLU)를 적용하였다.
# 사실 보통은 출력층에 활성화 함수를 적용하지 않는다. 
# 하이퍼 파라미터와 마찬가지로 은닉층, 출력층에서 활성화 함수를 적용할지 말지, 
# 또한 어떤 활성화 함수를 적용할지의 결정은 경험, 실험적 요소로 결정되는 부분이다.


# In[12]:


# 이번에는 손실함수를 텐서플로우가 기본적으로 제공하는 교차 엔트로피 함수를 이용해 보도록 하겠다.
# 하지만 각 요소의 평균은 직접 내야 한다.
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        labels=Y, 
        logits=model))


# In[13]:


# 최적화 함수로는 이번에는 AdamOptimizer를 사용해본다.
# AdamOptimizer는 앞서 사용하였던 GradientDescentOptimizer보다 보편적으로는 성능이 좋다고 알려져 있다.
# 하지만 항상 그런것은 아니니 여러가지로 시도해 보아야 할 것이다.

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)


# In[14]:


# 이제 지난번과 동일한 방식으로 세션을 만들고 실행해보도로 하자.

# 텐서플로우의 세션을 초기화한다.
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 이제 100번을 반복해야 한다.
for step in range(100):
    # 학습 연산을 수행해보자
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})
    
    # 학습 도중 10번에 한번씩 손실 값을 출력해 본다.
    if (step + 1) % 10 == 0:
        print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))


# In[15]:


# 이제 학습된 결과를 확인해 보는 코드를 작성해보자
prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y, axis=1)

# tf.argmax()는 axis에 해당하는 차원의 요소들 중 가장 큰 값의 인덱스를 찾아준다.
print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
print('실제값:', sess.run(target, feed_dict={Y: y_data}))


# In[16]:


# 마지막으로 정확도를 출력해보자
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))

# tf.equal()은 두개의 값이 같은지 다른지의 여부를 true, false값으로 리턴하는 함수이고
# tf.cast()는 true, false 값을 1, 0 으로 바꾸는 함수이다.
# 이렇게 나온 값을 평균내어 정확도를 구하게 된다.


# In[ ]:


# 정확도가 100퍼센트가 나왔다!

