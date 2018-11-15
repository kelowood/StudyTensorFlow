#!/usr/bin/env python
# coding: utf-8

# In[17]:


# 인공뉴런
# 인공신경망 개념은 뇌를 구성하는 신경세포인 뉴런(Neuron)의 동작원리에 기초한다.
# 인공뉴런의 개념은 책 53페이지 참조
# 인공뉴런을 수식으로 표현하자면 아래와 같다.
# y = Sigmoid(x * W + b)
# y : 출력
# Signoid : 활성화 함수
# x : 입력
# W : 가중치
# b : 편향
# 여기서 활성화 함수(activation function)란 인공신경망을 통과해온 값을 최종적으로 어떤 값으로 만들지를 결정하는 역활을 한다.
# 활성화 함수는 Sigmoid, ReLU, tanh 등등이 존재한다.


# In[18]:


import tensorflow as tf
import numpy as np

# numpy는 유명한 수치해석용 파이썬 라이브러리이다. 행렬조작 및 연산에 긴밀하게 이용된다.

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

# 데이터는 원-핫 인코딩(one-hot encoding)이라는 특수한 형태로 구성한다.
# 이것은 데이터가 가질 수 있는 값들을 일렬로 나열한 배열로 만들고, 
# 그중 표현하려는 값을 뜻하는 인덱스 원소만 1로 표기하고 나머지는 모두 0으로 채우는 표기법이다.

# 아래 데이터는 기타, 포유류, 조류를 원-핫 인코딩으로 표현한 방식이다.
# 기타 = [1, 0, 0]
# 포유류 = [0, 1, 0]
# 조류 = [0, 0, 1]

# 위 방식에 따라 출력 데이터를 아래와 같이 만든다.
y_data = np.array([
    [1, 0, 0], # 기타
    [0, 1, 0], # 포유류
    [0, 0, 1], # 조류
    [1, 0, 0], # 기타
    [1, 0, 0], # 기타
    [0, 0, 1]  # 조류
])


# In[19]:


# 신경망 모델을 구성해보자
# 특징 X와 레이블 Y와의 관계를 알아내는 모델이다.
# X, Y에 실측값 (ground truth)을 넣어서 학습시킬 것이므로 X, Y를 플레이스 홀더로 정의한다.
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# tf.placeholder() 함수의 인자에 Shape 값이 들어가 있지 않더라도 feed_dict로 배열을 넣을 수 있는 모양이다.


# 다음으로는 가중치와 편향값의 설정이다.
W = tf.Variable(tf.random_uniform([2, 3], -1., 1.))
b = tf.Variable(tf.zeros([3]))
# W 값은 [-1.0, 1.0) 실수범위의 랜덤값으로 구성된 [2, 3] 셰이프의 텐서이다. => [[?, ?, ?], [?, ?, ?]]
# b 값은 0으로만 구성된 [3] 셰이프의 텐서이다. => [0, 0, 0]

# 이제 이 가중치 값을 곱하고 편향을 더한 결과를 활성화 함수인 ReLU에 적용한다.
L = tf.add(tf.matmul(X, W), b)
L = tf.nn.relu(L)

# 여기에서 타 블로그의 내용을 잠깐 가져온다.
# ---------------------------------------------------
# "그런데 예를들어 어떤 훈련 데이터에 대해서 마지막 레이어 뉴런의 출력 값이 
# (강아지, 고양이, 토끼) = (0.9, 0.8, 0.7) 이렇게 나왔고 
# 또 다른 훈련 데이터에 대해서는 (강아지, 고양이, 토끼) = (0.5, 0.2, 0.1) 와 같은 출력이 나왔다고 가정하면 
# 둘 중 어떤 데이터가 강아지에 가까운 것일까요? 
# 전자의 경우 세개의 클래스에 대한 확률이 거의 비슷하므로 어떤 이미지인지 잘 구분하기 어렵습니다. 
# 하지만 후자의 경우는 강아지인 절대 확률은 낮지만 고양이나 토끼 보다는 강아지에 더 가까운 것 같습니다. 
# 이렇게 여러개의 클래스를 구분할 경우 마지막 뉴런의 활성화 함수로 시그모이드를 사용하면 출력 값을 공정하게 평가하기 어렵습니다. 
# 그래서 뉴런의 출력 값을 정규화하는 소프트맥스(softmax) 함수를 주로 사용합니다."
# ---------------------------------------------------
# 즉, 원-핫-인코딩 형태의 값을 정규화 시켜서 어느 출력 데이터에 가까운지를 판단하기 위해 소프트맥스 함수를 사용하게 된다.
# tf.nn.softmax() 함수는 배열내의 결과값들을 전체 합이 1이 되도록 정규화시켜준다.

model = tf.nn.softmax(L)


# In[25]:


# 이번에는 손실함수를 작성해볼 차례다.
# 손실 함수는 원-핫-인코딩을 이용하는 모델에서는 대부분 교차 엔트로피(Cross-Entropy)라는 함수를 사용한다.
# 교차 엔트로피 값은 예측값과 실제값 사이의 확률 분포 차이를 계산한 것이다.
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))

# 계산방식은 책의 63페이지 참고
# 여기에서 tf.reduce_xxx() 계열 함수는 텐서의 차원을 줄여준다. axis 매개변수로 축소할 차원을 정하여 연산을 수행하고 차원을 줄여주게 된다.


# In[26]:


# 최적화 함수를 정의하여 학습을 어떻게 시킬지 정해보자
# 최적화 함수는 전에 했던 대로 경사하강법을 사용한다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

# 여기서 minimize()는 아래 역할을 수행하는 함수이다.
# 1) gradient 계산
# 2) gradient를 원하는 대로 처리
# 3) 처리된 gradient를 적용


# In[27]:


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


# In[28]:


# 이제 학습된 결과를 확인해 보는 코드를 작성해보자
prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y, axis=1)

# tf.argmax()는 axis에 해당하는 차원의 요소들 중 가장 큰 값의 인덱스를 찾아준다.
print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
print('실제값:', sess.run(target, feed_dict={Y: y_data}))


# In[29]:


# 마지막으로 정확도를 출력해보자
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))

# tf.equal()은 두개의 값이 같은지 다른지의 여부를 true, false값으로 리턴하는 함수이고
# tf.cast()는 true, false 값을 1, 0 으로 바꾸는 함수이다.
# 이렇게 나온 값을 평균내어 정확도를 구하게 된다.


# In[30]:


# 하지만 정확도를 썩 좋은 편은 아니다.
# 그리고 학습 횟수를 아무리 늘려도 정확도가 높아지지 않는다.
# 그 이유는 신경망이 딱 한층밖에 되지 않아서 그렇고 층을 하나 더 늘리면 해결될것 같다.
# 그 부분은 다음 장에서..

