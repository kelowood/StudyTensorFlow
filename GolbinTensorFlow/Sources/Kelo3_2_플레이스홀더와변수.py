#!/usr/bin/env python
# coding: utf-8

# In[27]:


import tensorflow as tf

# 플레이스 홀더란 그래프에 사용할 입력값을 실행시에 받는 매개변수이다.
# 즉, 플레이스 홀더를 정의하였다면 Session을 만들고 실행(run)하는 시점에서 값을 넣어주어야 하는 것이다.
# 아래 의미는 float32 타입의 임의행-3열의 행렬형 플레이스 홀더를 정의하는 것을 의미한다.
X = tf.placeholder(tf.float32, [None, 3])
print(X)


# In[28]:


# 위 텐서는 플레이스홀더형 텐서이며, 랭크는 2, 셰이프는 [?, 3]이다.

# 실행시에 플레이스 홀더에 넣어줄 값을 지금 먼저 정의해보자
x_data = [[1, 2, 3], [4, 5, 6]]
# [2, 3] 셰이프의 값이므로 위의 형태와 맞게 된다.

# 아래는 변수(Variable)의 정의이다.
# 텐서플로에서의 변수는 그래프를 최적화하는 용도로 쓰이는 값으로, 학습 함수들이 학습한 결과를 갱신하기 위해 필요한 값이다.
W = tf.Variable(tf.random_normal([3, 2]))
b = tf.Variable(tf.random_normal([2, 1]))

# 단, 현재 알아둬야 하는건 지금 이 프로그램의 W, b 는 학습을 할수 있는 형태가 아니다.
# random_normal() 함수를 통하여 무작위 값을 가져오게 되는 형태이다.
print(W)
print(b)


# In[29]:


# 입력값과 변수들을 계산할 수식을 작성한다.
# X와 W는 행렬이므로 tf.matmul() 행렬곱 함수를 사용한다.
# 행렬이 아닐때는 tf.mul() 혹은 단순히 곱셈 연산자를 사용해도 된다.
expr = tf.matmul(X, W) + b
print(expr)

# X는 ? * 3 행렬이고, W는 3행 2열이므로 tf.matmul(X, W)의 결과는 ? * 2 행렬이 나온다.
# 여기서 2 * 1 행렬 b를 더할시 행렬곱 결과가 2 * 2 행렬이었을때 덧셈이 성립하고 b의 각 요소들을 더하게 된다.


# In[30]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())

# tf.global_variables_initializer() 는 앞에서 정의한 변수 (위에서 random_normal()로 정의한것)를 초기화하는 함수이다.
# 기존 학습 값들을 가져와서 사용하는 것이 아닌 첫 실행이라면 연산 실행전 변수를 초기화 하여야 한다.

print("=== x_data ===")
print(x_data)
print("=== W ===")
print(sess.run(W))
print("=== b ===")
print(sess.run(b))


# In[31]:


# 플레이스 홀더가 포함된 그래프를 실행시 feed_dict을 통해서 입력값을 넣어 주어야 한다.
# 그렇지 않으면 에러가 난다.
print("=== expr ===")
print(sess.run(expr, feed_dict={X: x_data}))

sess.close()


# In[ ]:




