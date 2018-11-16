#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Sequence to Sequence <Seq2Seq>
# RNN 신경망과 출력하는 신경망을 조합한 모델
# 번역이나 챗봇등에서 많이 사용된다.
# 구글 기계번역에서도 이것을 사용하고 있다.

# Seq2Seq 모델은 크게 두가지로 구성된다.
# 1) 인코더 : 입력을 위한 신경망. 원문을 입력으로 받음.
# 2) 디코더 : 출력을 위한 신경망. 인코더가 번역한 결과물을 입력으로 받음.


# In[11]:


# Seq2Seq 심볼은 특수한 심볼이 몇개 필요하다. 
# (골빈해커책의 195 페이지 참조)
# S 심볼 : 디코더에 입력이 시작됨을 알려주는 심볼
# E 심볼 : 디코더의 출력이 끝났음을 알려주는 심볼
# P 심볼 : 빈 데이터를 채울때 사용하는 무의미 심볼

import tensorflow as tf
import numpy as np

tf.reset_default_graph()

# 이제 데이터를 만들어보자. 먼저 영어알파벳과 한글들을 나열한 뒤에 한글자씩 배열에 집어넣어보자.


char_arr = [charcter for charcter in 'SEPabcdefghijklmnopqrstuvwxyz단어나무놀이소녀키스사랑']
num_dic = {n: i for i , n in enumerate(char_arr)}
dic_len = len(num_dic)

seq_data = [
    ['word', '단어'],
    ['wood', '나무'],
    ['game', '놀이'],
    ['girl', '소녀'],
    ['kiss', '키스'],
    ['love', '사랑']
]


# In[27]:


# 그다음 입력 단어와 출력 단어를 한글자씩 떼어내서 배열로 만든 다음,
# 원-핫 인코딩 형태로 바꿔주는 함수를 만든다.
# 리턴 데이터는 아래와 같이 3가지로 구성된다.
# 1) 인코더의 입력값
# 2) 디코더의 입력값
# 3) 디코더의 출력값

def MakeBatch(_seqData):
    input_batch = []
    output_batch = []
    target_batch = []
    
    for iterSeq in _seqData:
        
        print('MakeBatch() Process<iterSeq>:', iterSeq, ' ------------')
        
        # 1) 인코더 입력값 : 입력단어('word')를 한글자씩 떼어서 인덱스 배열로 만든다.
        inputData = [num_dic[n] for n in iterSeq[0]]
        # 2) 디코더 입력값 : 출력단어('단어') 앞에 디코더 시작을 나타내는 S 글자를 붙인다.
        # 그 후 한글자씩 떼어서 인덱스 배열로 만든다.
        outputData = [num_dic[n] for n in ('S' + iterSeq[1])]
        # 3) 디코더 출력값 : 출력단어('단어') 마지막에 디코더 끝을 알리는 심볼 'E' 글자를 붙인다.
        # 그 후 한글자씩 떼어서 인덱스 배열로 만든다.
        targetData = [num_dic[n] for n in (iterSeq[1] + 'E')]
        
        print('MakeBatch() inputData:', inputData)
        print('MakeBatch() outputData:', outputData)
        print('MakeBatch() targetData:', targetData)
        print('----------------------------------------------------')
        
        # 인코더 입력값, 디코더 입력값은 원-핫 인코딩 형태로 인코딩하여 리스트에 추가하고,
        # 디코더 출력값만 정수 인덱스 값 그대로 리스트에 추가한다.
        input_batch.append(np.eye(dic_len)[inputData])
        output_batch.append(np.eye(dic_len)[outputData])
        target_batch.append(targetData)
    
    return input_batch, output_batch, target_batch       


# In[13]:


# 하이퍼 파라미터의 정의

hp_learning_rate = 0.01
hp_n_hidden = 128
hp_total_epoch = 100

hp_n_class = hp_n_input = dic_len

# 이제 플레이스홀더를 만들어보고자 한다.
# 여기에서 인코더 및 디코더의 입력값 형식은 아래와 같다.
# [batchSize, timeSteps, inputSize]
# 여기에서 inputSize는 hp_n_input 즉 딕셔너리의 길이, char_arr로 정의한 글자들의 갯수를 뜻한다.

# 그리고 디코더 출력값의 형식은 아래와 같다.
# [batchSize, timeSteps]

enc_input = tf.placeholder(tf.float32, [None, None, hp_n_input])
dec_input = tf.placeholder(tf.float32, [None, None, hp_n_input])
targets = tf.placeholder(tf.int64, [None, None])

# 잘 이해가 가지 않는 부분이다.
# 여기에서 timeSteps는 글자수를 의미하는 것이라고 한다. 즉, 'word'면 4글자이고, '단어'라면 두글자가 될 것이다.
# 다음 부분에서 알아봐야 할 것 같다.


# In[14]:


# RNN 모델을 위한 셀 구성 부분이다.
# 인코더 셀과 디코더 셀 이렇게 두 종류의 셀이 구성된다.

with tf.variable_scope('encode'):
    enc_cell = tf.nn.rnn_cell.BasicRNNCell(hp_n_hidden)
    enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5)
    
    outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input, dtype=tf.float32)

# 여기서 중요한점이 디코더를 만들때 초기 상태값으로 인코더의 최종 상태값을 넣어주어야 한다는 것이다.
# seq2seq의 핵심중 하나는 인코더에서 계산된 상태를 디코더로 전파하는 것이다.
with tf.variable_scope('decode'):
    dec_cell = tf.nn.rnn_cell.BasicRNNCell(hp_n_hidden)
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)
    
    outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input, initial_state=enc_states, dtype=tf.float32)


# In[17]:


# 다음으로 출력층을 만들고 손실함수, 최적화 함수를 구성해보자
model = tf.layers.dense(outputs, hp_n_class, activation=None)

cost = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=model, labels=targets))

optimizer = tf.train.AdamOptimizer(hp_learning_rate).minimize(cost)


# In[23]:


# 세션을 시작하기전에 여기서 먼저 seq_data로부터 나오는 배치 데이터들이 어떤형태로 나오는지 봐보자.
input_batch, output_batch, target_batch = MakeBatch(seq_data)


# In[21]:


# 세션을 실행해보자
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(hp_total_epoch):
    _, loss = sess.run(
        [optimizer, cost],
        feed_dict=
        {
            enc_input: input_batch,
            dec_input: output_batch,
            targets: target_batch
        })
    
    print('Eopch:', '%04d' % (epoch + 1),
          'cost = ', '{:.6f}'.format(loss))
    
print('최적화 완료')


# In[26]:


# 결과를 확인하기 위해서 단어를 입력받아 번역 단어를 예측하는 함수를 만들어보자
def Translate(_word):
    seqData = [_word, 'P' * len(_word)]
    
    input_batch, output_batch, target_batch = MakeBatch([seqData])
    
    # 여기에서 _word 인자로 'kiss'를 받았다면 seqData는 ['kiss', 'PPPP']로 구성될 것이다.
    # 사전에 설명했다시피 'P' 글자는 빈 데이터를 채울때 사용하는 무의미 심볼을 의미한다.
    # input_batch는 ['k', 'i', 's', 's'], output_batch는 ['P', 'P', 'P', 'P'] 글자들의 인덱스를 원-핫인코딩한 것들이다.
    # target_batch는 ['P', 'P', 'P', 'P']
    
    # 예측 모델을 돌려보자
    # print('model:', model)
    # model의 출력 형태는 [batchSize, timeSteps, inputSize] 형태인데 여기서 세번째 차원 inputSize의 argmax값을 사용한다.
    prediction = tf.argmax(model, 2)

    result = sess.run(
        prediction,
        feed_dict=
        {
            enc_input: input_batch,
            dec_input: output_batch,
            targets: target_batch
        })
    
    # 예측 결과는 글자의 인덱스를 뜻하는 숫자이므로 각 숫자에 해당하는 글자를 가져와서 배열을 만든다.
    # 그리고 출력의 끝을 의미하는 E 이후의 글자들을 제거하고 문자열로 만든다. 
    # 출력은 디코더의 입력 크기(time steps) 만큼 출력값이 나오므로 최종결과는 ['사', '랑', 'E', 'E'] 처럼 나오기 때문이다.
    decoded = [char_arr[i] for i in result[0]]
    
    end = decoded.index('E')
    translated = ''.join(decoded[:end])
    
    return translated


# In[28]:


# 실제 번역테스트를 수행해보도록 하자.
print('\n=== 번역 테스트 ===')

trans_word = Translate('word')
trans_wodr = Translate('wodr')
trans_love = Translate('love')
trans_loev = Translate('loev')
trans_abcd = Translate('abcd')

print('word ->', trans_word)
print('wodr ->', trans_wodr)
print('love ->', trans_love)
print('loev ->', trans_loev)
print('abcd ->', trans_abcd)


# In[ ]:




