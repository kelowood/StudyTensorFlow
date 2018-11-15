#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 이번에는 숫자를 무작위로 생성하지 않고 원하는 숫자를 지정하여 생성하는 모델을 만들어 보자


# In[2]:


# tensorflow, matplotlib.pyplot, numpy, minst 튜토리얼 데이터를 임포트한다.

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

# 하이퍼 파라미터 설정하기
hp_total_epoch = 100
hp_batch_size = 100
hp_n_hidden = 256
hp_n_input = 28*28
hp_n_noise = 128
hp_n_class = 10 # 숫자의 갯수

# 여기서 플레이스 홀더에 Y값을 추가한다.
# 이 Y값은 결과값 판정용은 아니다.
# 노이즈와 실제 이미지에 각각에 해당하는 숫자를 힌트로 넣어주는 용도로 사용할 것이다.

X = tf.placeholder(tf.float32, [None, hp_n_input])
Y = tf.placeholder(tf.float32, [None, hp_n_class])
Z = tf.placeholder(tf.float32, [None, hp_n_noise])


# In[3]:


# 이번에는 신경망 구성을 다른방식으로 적용할텐데 변수들을 선언하지 않고 tf.layers를 사용해보고자 한다.
# 이전 장에서는 생성자와 구분자를 동시에 학습시켜야 했고, 학습시 각 신경망 변수를 따로따로 학습시켜야 했다.
# 그러나 tf.layers를 사용하면 변수를 선언하지 않고 tf.variable_scope()를 이용하여 스코프를 지정해줄 수 있다.

# tf.concat() : 텐서들을 하나의 차원에서 연결시키는 메서드

# 생성자에 대하여 신경망을 만들어 보자.
def Generator(noise, labels):
    
    with tf.variable_scope('Generator'):
        
        inputs = tf.concat([noise, labels], 1)
        hidden = tf.layers.dense(inputs, hp_n_hidden, activation=tf.nn.relu)    
        output = tf.layers.dense(hidden, hp_n_input, activation=tf.nn.sigmoid)
        
    return output

# 구분자 신경망을 만들어 보자.
# 여기서 _isReuse 매개변수를 사용하여, 
# scope.reuse_variables() 함수를 호출시켜 이전에 사용한 변수를 재사용하도록 한다.
# 그 이유는 진짜 이미지를 판별할 때와 가짜 이미지를 판별할때 똑같은 변수를 사용해야 하기 때문이다.
def Discriminator(inputs, labels, reuse=None):
    
    with tf.variable_scope('Discriminator') as scope:
        if reuse:
            scope.reuse_variables()
        
        inputs = tf.concat([inputs, labels], 1)
        hidden = tf.layers.dense(inputs, hp_n_hidden, activation=tf.nn.relu)
        output = tf.layers.dense(hidden, 1, activation=None)
        
    return output


# In[4]:


# 그리고 노이즈 생성 유틸리티 함수에서 이번에는 노이즈를 균등분포로 생성하도록 작성한다.
def GenNoise(_batchSize, _numNoise):
    return np.random.uniform(-1., 1., size=[_batchSize, _numNoise])

G = Generator(Z, Y)
D_real = Discriminator(X, Y)
D_gene = Discriminator(G, Y, True)


# In[5]:


# 이제는 손실함수를 만들 차례다.

# 1) 구분자의 손실함수
# 이전과 같이 진짜 이미지를 판별하는 D_real 값은 1에 가까워지게 만들고,
# D_gene 값은 0에 가까워지도록 해야 한다.
# 하지만 9.1장과 다르게 이번에는 tf.nn.sigmoid_cross_entropy_with_logits()라는 함수를 만들어보자.
loss_D_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_real,
        labels=tf.ones_like(D_real)))
loss_D_gene = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_gene,
        labels=tf.zeros_like(D_gene)))

loss_D = loss_D_real + loss_D_gene

# 2) 생성자의 손실함수
# D_gene 값은 1에 가까워지게 해야 한다.
# 마찬가지로 tf.nn.sigmoid_cross_entropy_with_logits()를 이용해 보자
loss_G = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_gene,
        labels=tf.ones_like(D_gene)))


# In[6]:


# 이제 학습모델을 구성하자.
# tf.get_collection()를 이용하여 Discriminator와 Generator 스코프에서 사용된 변수들을 가져온다.
# 그 후 이 변수들을 최적화에 사용할 각각의 손실함수와 함께 최적화 함수에 넣도록 한다.
vars_D = tf.get_collection(
    tf.GraphKeys.TRAINABLE_VARIABLES,
    scope='Discriminator')
vars_G = tf.get_collection(
    tf.GraphKeys.TRAINABLE_VARIABLES,
    scope='Generator')

train_D = tf.train.AdamOptimizer().minimize(loss_D, var_list=vars_D)
train_G = tf.train.AdamOptimizer().minimize(loss_G, var_list=vars_G)


# In[7]:


# 이제 세션을 시작해보자
# 이전장과 내용이 거의 비슷하지만 Y 입력값으로 batch_ys가 들어가게 된다.
sess = tf.Session()
sess.run(tf.global_variables_initializer())

num_total_batch = int(mnist.train.num_examples / hp_batch_size)
res_loss_val_D, res_loss_val_G = 0, 0

for epoch in range(hp_total_epoch):
    for i in range(num_total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(hp_batch_size)
        noise = GenNoise(hp_batch_size, hp_n_noise)
        
        _, res_loss_val_D = sess.run(
            [train_D, loss_D],
            feed_dict={X: batch_xs, Y: batch_ys, Z: noise})
        _, res_loss_val_G = sess.run(
            [train_G, loss_G],
            feed_dict={Y:batch_ys, Z: noise})
        
    print(
        'Epoch:', '%04d' % epoch,
        'D loss: {:.4}'.format(res_loss_val_D),
        'G loss: {:.4}'.format(res_loss_val_G))
        
    # 확인용 이미지를 만들어보도록 하자.
    # 0, 9. 19, 29, ...번째 에포크마다 생성기로 이미지를 생성하여 눈으로 직접 확인해보도록 하자.
    if epoch == 0 or (epoch + 1) % 10 == 0:
        sample_size = 10
        noise = GenNoise(sample_size, hp_n_noise)
        samples = sess.run(G, feed_dict={Y: mnist.test.labels[:sample_size], Z: noise})
        
        fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))
        
        for i in range(sample_size):
            ax[0][i].set_axis_off()
            ax[1][i].set_axis_off()
            
            ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            ax[1][i].imshow(np.reshape(samples[i], (28, 28)))
            
        plt.savefig(
            'samples2/{}.png'.format(str(epoch).zfill(3)),
            bbox_inches='tight')
        
        plt.close(fig)

print('최적화 완료.')

