#!/usr/bin/env python
# coding: utf-8

# In[2]:


# 텐서플로우 라이브러리 임포트
import tensorflow as tf

# 상수 텐서 정의. 스트링으로 된 상수이며, hello 변수에 할당함.
hello = tf.constant('Hello, TensorFlow!')
# 정의한 텐서를 출력해본다.
print(hello)


# In[5]:


# hello의 자료형은 tensor이며 이것은 상수를 담고 있다.
# 랭크는 0이며, 셰이프는 []

a = tf.constant(10)
b = tf.constant(32)
c = tf.add(a, b)
print(a)
print(b)
print(c)


# In[7]:


# 변수 a, b는 int32 타입의 상수를 담은 텐서이며,
# 변수 c는 42가 아닌 add라는 형태의 텐서이다.
# 이 텐서 연산들이 모여서 그래프가 된다. 그래프 == 텐서들의 연산 모음
# 텐서플로에서는 텐서들의 연산을 정의하여 그래프를 만들고, 이후 필요할때 실제 연산을 수행하는 코드를 넣는다.
# 이러한 수행 방식을 지연 실행(lazy evaluation)이라고 부른다.
# 아래 코드에서 위에 정의한 그래프를 실행해보도록 한다.

# 그래프의 실행은 Session 내에서 이루어진다.
sess = tf.Session()

# hello 텐서의 수행
print(sess.run(hello))
# a, b, c 텐서의 수행 (배열로 각각 수행한다)
print(sess.run([a, b, c]))

sess.close()


# In[ ]:


# 정상적으로 실행된 그래프를 확인 할 수 있다.

