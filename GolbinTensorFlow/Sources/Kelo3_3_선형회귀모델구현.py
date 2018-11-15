#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 선형 회귀는 간단하게 말하자면 주어진 x, y 값을 가지고 서로간의 관계를 파악하는 것이다.
# (책 44페이지 참조) 2차 선형 회귀는 2차 그래프의 직선을 파악하여 입력에 대한 출력을 산출해 낼 수 있는 것이다.


# In[10]:


import tensorflow as tf

# x와 y 데이터를 정의한다.
x_data = [1, 2, 3]
y_data = [1, 2, 3]

# x, y의 상관관계를 설명하기 위한 변수 W, b를 정의한다.
# 이 두 값은 스칼라 값이며, -1.0 ~ 1.0 범위의 균등 분포를 가진 무작위값으로 초기화된다.
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
print(W)
print(b)


# In[11]:


# 다음은 자료를 입력받기 위해 필요한 플레이스 홀더이다.
# 자료형은 float32의 스칼라 값이다.
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")
print(X)
print(Y)

# 여기서는 플레이스 홀더 정의시 name 매개변수에 값을 넣음으로써 이름을 설정하는 것을 알 수 있다.
# 위와 같이 이름을 넣어줄 시 어떤 텐서가 어떻게 사용되고 있는지 쉽게 알 수 있다. 
# 그리고 차후에 사용할 디버깅도구인 텐서보드에서도 이 이름을 통해 디버깅을 수월하게 할 수 있다.
# 이름은 플레이스 홀더 뿐만 아니라 변수, 연산 및 연산함수에도 지정할 수 있다.


# In[12]:


# X, Y의 상관관계를 분석하기 위한 수식을 작성한다.
hypothesis = W * X + b
print(hypothesis)

# 이 수식은 W와의 곱과 b와의 합을 통하여 X, Y의 관계를 설명하겠다는 뜻이다.
# 즉, 주어진 X, Y가 있을때 이것에 적합한 W, b 값을 찾아내겠다는 의미이기도 하다.
# 여기서 곱연산을 하는 W는 가중치(Weight)라 하고, b는 편향(bias)라고 부른다.


# In[13]:


# 다음은 손실함수에 대한 작성이다.
# 손실함수(loss function)는 데이터에 대한 손실값을 나타내는 함수이다.
# 손실값 : X에 대한 실제값(Y)과 모델로 예측한 값이 얼마나 차이가 나는지를 나타내는 값
# 손실값이 작을수록 X에 대한 Y값을 정확히 예측할 수 있게 된다.
# 이 손실값을 전체 데이터에 대해 구한 경우 이것을 비용(cost)이라고 부른다.
# 즉 학습 이라 함은. 다양한 값들을 넣어봄으로써 손실값을 최소화하는 W, b 값을 구하는 것을 말한다.

# 손실값은 '예측값과 실제값의 거리'를 가장 많이 쓴다.
# 이 값은 예측 값에서 실제 값을 뺀 다음 제곱하여 산출한다.
# 그리고 비용 값은 모든 데이터들에 대한 손실값의 평균을 내어서 구한다.
cost = tf.reduce_mean(tf.square(hypothesis - Y))
print(cost)

# tf.square()는 각 요소의 제곱을 구하는 함수이다.
# tf.reduce_mean(x)는 x의 각 요소에 대한 평균 값을 구하는 함수이다.


# In[14]:


# 최적화 함수란 가중치와 편향 값을 변경해 가면서 손실값을 최소화하는 최적의 두 값을 찾아주는 함수이다.
# 이 두 값을 구할때 값들을 단순히 무작위로 변경한다면 시간이 너무 오래 걸리고 학습시간도 예측하기가 어렵다.
# 그렇기 때문에 빠르게 최적화하기 위한 다양한 방법을 사용하게 된다.
# 아래 사용하게 될 경사하강법(Gradient descent) 최적화 함수는 가장 기본적인 알고리즘으로 
# 함수의 기울기가 낮은쪽으로 계속 이동시키면서 최족의 값을 찾아나가는 방법이다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(cost)
print(optimizer)
print(train_op)

# 여기서 쓰인 학습률(learning_rate)은 학습을 얼마나 급하게 할것인지 설정하는 값이다.
# 이 값이 너무 크면 최적의 손실값을 찾지 못하고 지나쳐 버린다.
# 이 값이 너무 작으면 학습 속도가 느려진다.
# 이와 같이 학습을 진행하는 과정에 영향을 미치는 변수를 하이퍼파라미터(hyperparameter)라고 부른다.
# 런닝머신은 이 하이퍼 파라미터를 어떻게 튜닝하냐에 따라 성능이 크게 틀려 질수 있다.


# In[17]:


with tf.Session() as sess:
    # 전 장과 마찬가지로 세션 생성 후 변수들을 초기화한다.
    sess.run(tf.global_variables_initializer())
    
    # 아래 반복문을 통하여 최적화를 수행하는 train_op 값을 구한다.
    # 이 작업을 하면서 변경되는 W, b, 비용(cost)값 또한 같이 뽑아내어 출력해보도록 한다.
    for step in range(100):
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})  
        print(step, cost_val, sess.run(W), sess.run(b))
        
    # train_op를 반복적으로 실행시켜 적절한 W, b를 구해내었다면 이제 X값을 넣어보고 적절한 Y값 결과를 확인해 보도록 하자.
    print("X: 5, Y:", sess.run(hypothesis, feed_dict={X: 5}))
    print("X: 2.5, Y:", sess.run(hypothesis, feed_dict={X: 2.5}))
    
    # 결과를 보다시피 예상된 값과 굉장히 근접한 값이 나옴을 확인할 수 있다.


# In[ ]:




