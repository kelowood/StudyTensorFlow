#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np

# 입력값은 각 글자에 해당하는 인덱스를 원-핫 인코딩으로 표현한 값을 사용할 것이다.

char_arr = [
    'a', 'b', 'c', 'd', 'e', 'f', 
    'g', 'h', 'i', 'j', 'k', 'l', 
    'm', 'n', 'o', 'p', 'q', 'r',
    's', 't', 'u', 'v', 'w', 'x',
    'y', 'z']

alpabet_dic = {n: i for i, n in enumerate(char_arr)}
print(alpabet_dic)

# 여기서 파이썬을 모르는 나에게 이해가 필요할 것 같다.
# enumerate() 는 순서가 있는 자료형(리스트, 튜플, 문자열)을 입력으로 받아서
# 인덱스 값을 포함하는 enumerate 객체를 리턴한다.

# 그리고 for문의 형태는 아래와 같다.
# for i in ??? -> 이것은 <리스트, 튜플, 문자열> 등의 자료형의 요소들을 반복하겠다는 것이다.

# enumerate()는 인자로 들어가는 리스트형 자료형의 인덱스를 추가로 넣어주어 튜플형태로 반환한다.
# enumerate(char_arr)를 받아서 반복적으로 출력하면 아래와 같이 나온다.
# for iterTP in enumerate(char_arr):
#     print(iterTP)
# 결과 : (0, 'a')
#        (1, 'b')
#        (2, 'c')
#       ...

# 즉, 위의 구문에서 for i, n in enumerate(char_arr) 이 구문이 들어가고 n : i가 들어갔다는 것은
# n값 즉, 알파벳값은 key값으로 들어가고 value값은 인덱스 값으로 넣는 딕셔너리를 만들겠다는 것이다.
# 위 딕셔너리의 결과는 대략 이와 같다.
# {'a': 0, 'b': 1, 'c', 2, ...}

dic_len = len(alpabet_dic)

# 그 다음 학습에 사용될 단어들을 배열로 저장한다.
seq_data = ['word', 'wood', 'deep', 'dive', 'cold', 'cool', 'load', 'love', 'kiss', 'kind']


# In[3]:


# 이제 아래 같은 형태로 함수를 작성할 것이다.
# 매개변수 값을 가공하여 학습 입력-출력값을 리턴하고 이것들을 학습에 쓰게 될 것이다.
# 1) 입력값용으로 단어의 처음 세 글자의 알파벳 인덱스를 구한 배열을 만든다. : [3, 4, 4]
# 2) 출력값용으로, 마지막 글자의 알파벳 인덱스를 구한다. : 15
# 3) 입력값을 원-핫 인코딩으로 변환한다.
# : [[0, 0, 0, 1, 0, ... 0]
#    [0, 0, 0, 0, 1, ... 0]
#    [0, 0, 0, 0 , 1, ... 0]]
# 단 여기서 출력값은 원-핫 인코딩을 사용하지 않고 15값 그대로 출력할 것이다.
# 그 이유는 손실함수로 사용하던 softmax_cross_entropy_with_logits를 쓰지 않고
# sparse_softmax_cross_entropy_with_logits을 사용할 것이기 때문이다.
# 이 함수는 실측값, 즉 label 값에 원-핫 인코딩을 사용하지 않아도 자동으로 변환해서 계산해준다.

def MakeBatch(_seqData):
    input_batch = []
    target_batch = []
    
    for seq in seq_data:
        input = [alpabet_dic[n] for n in seq[:-1]]
        target = alpabet_dic[seq[-1]]
        # np.eye()는 매개변수 n*n 크기의 정방 항등행렬을 만들어주는 함수이다.
        # dic_len이 알파벳의 크기이므로 25*25 크기의 항등행렬이 만들어진다.
        # 여기에서 input 값의 행 요소를 가져온다는 것은 바로 input값의 원-핫 인코딩 값을 가져온다는 것이 된다.
        # 이 값을 input_batch에 넣는다.
        input_batch.append(np.eye(dic_len)[input])
        # target 값은 그대로 target_batch에 넣는다.
        target_batch.append(target)
    
    # 입력 배치 값과 출력 배치값을 리턴한다.
    return input_batch, target_batch


# In[4]:


# 이제 신경 모델망을 구성해보자

hp_learning_rate = 0.01
hp_n_hidden = 128
hp_n_epoch = 30

hp_n_step = 3 
hp_n_input = hp_n_class = dic_len

# 처음 3글자를 단계적으로 학습할 것이므로 hp_n_step 3이 된다.
# 또한 sparse_softmax_cross_entropy_with_logits을 사용한다 하더라도 
# 예측 모델의 출력값은 원-핫 인코딩이 되어야 한다.

X = tf.placeholder(tf.float32, [None, hp_n_step, hp_n_input])
Y = tf.placeholder(tf.int32, [None])

print('X:', X)
print('Y:', Y)

# 실측값 플레이스홀더 Y값은 하나의 차원만 존재한다.
# 원-핫 인코딩이 아니라 인덱스 숫자를 그대로 쓰기 때문에 값이 하나뿐인 1차원 배열을 입력으로 받게 된다.

W = tf.Variable(tf.random_normal([hp_n_hidden, hp_n_class]))
b = tf.Variable(tf.random_normal([hp_n_class]))


# In[5]:


# 그 다음으로 두개의 RNN 셀을 생성한다.
# 여러셀을 조합하여 심층 신경망을 만들기 위해서다.
# 또한 DropoutWrapper 함수를 사용하여 RNN에서도 과적합 방지를 위한 드롭아웃 기법을 적용시킬 수 있다.
cell1 = tf.nn.rnn_cell.BasicLSTMCell(hp_n_hidden)
cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)
cell2 = tf.nn.rnn_cell.BasicLSTMCell(hp_n_hidden)

# 그 다음 MultiRNNCell 함수를 사용하여 셀들을 조합하고
# dynamic_rnn 함수를 사용하여 심층순환신경망(Deep RNN)을 만든다.
multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)

# 10.1 때의 MNIST 예측모델과 마찬가지로 출력층을 만든다.
outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]
model = tf.matmul(outputs, W) + b

# 손실함수는 sparse_softmax_cross_entropy_with_logits를 이용한다.
# 최적화 함수는 AdamOptimizer를 사용한다.
cost = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=model,
        labels=Y))
optimizer = tf.train.AdamOptimizer(hp_learning_rate).minimize(cost)


# In[6]:


# 세션을 만들고 학습을 수행한다.
sess = tf.Session()
sess.run(tf.global_variables_initializer())

input_batch, target_batch = MakeBatch(seq_data)

for epoch in range(hp_n_epoch):
    _, loss = sess.run(
        [optimizer, cost],
        feed_dict={X: input_batch, Y: target_batch})
        
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    
print('최적화 완료!')


# In[8]:


# 결과값으로 예측한 단어를 정확도와 함꼐 출력해보도록 하자.

print("model:", model)

prediction = tf.cast(tf.argmax(model, 1), tf.int32)
prediction_check = tf.equal(prediction, Y)
accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

print("model:", model)
print("prediction:", prediction)
print("prediction_check:", prediction_check)


# 여기서는 실측값을 원-핫 인코딩이 아닌 인덱스를 그대로 사용하므로 Y는 일반 정수가 된다.
# 그렇기 때문에 argmax로 변환한 예측값도 정수로 변환시켜줘야 한다.

input_batch, target_batch = MakeBatch(seq_data)

predict, accuracy_val = sess.run(
    [prediction, accuracy],
    feed_dict={X: input_batch, Y: target_batch})


# In[9]:


# 마지막으로 모델이 예측한 결과값들을 가지고, 
# 각각의 값에 해당하는 인덱스의 알파벡을 가져와서 예측한 단어를 출력해보자

predict_words = []
for idx, val in enumerate(seq_data):
    # last_char : 
    last_char = char_arr[predict[idx]]
    predict_words.append(val[:3] + last_char)
    
print('\n=== 예측 결과 ===')
print('입력값:', [w[:3] + ' ' for w in seq_data])
print('예측값:', predict_words)
print('정확도:', accuracy_val)


# In[ ]:




