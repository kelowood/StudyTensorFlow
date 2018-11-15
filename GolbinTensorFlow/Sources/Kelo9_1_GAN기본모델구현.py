#!/usr/bin/env python
# coding: utf-8

# In[1]:


# GAN (Generative Adversarial Network)
# 오토 인코더와 마찬가지로 결과물을 생성하는 생성 모델 중 하나이다.
# 서로 대립(adversarial)하는 두 신경망을 경쟁시켜가면서 결과물 생성방법을 학습하는 모델

# 구분자와 생성자를 잘 이해해야 한다.
# 위조지폐범과 정상지폐를 구분하는 경찰과의 관계를 참고하자 (책. 148페이지 참조)

# 구분자 (Discriminator)
# 경찰 역활. 이미지를 주고 이것이 진짜인지 가짜인지 판별하게 하는 존재.
# 가장 먼저는 실제 이미지를 준다음 이것이 진짜임을 판단하게 해야 한다.
# 그 다음 생성자를 통해 노이즈로부터 만들어진 임의의 이미지를 만들고 
# 이것을 해당 구분자를 통해 가짜임을 판별하도록 한다.
# 생성자의 이미지가 진짜인지 가짜인지 잘 분류할 수 있도록 학습하는 것이 주 목표

# 생성자 (Generator)
# 위조지폐범 역활. 구분자가 진짜라고 판단할 정도로 이미지를 만들어내도록 훈련
# 구분자가 구분을 못하도록 흡사하게 이미지를 만들어내도록 학습하는 것이 주 목표

# 구분자와 생성자의 경쟁을 통해 
# 결과적으로 생성자는 실제 이미지와 상당히 비슷한 결과를 만들어내게 된다.

# 이 코드는 MNIST 손글씨 숫자를 무작위로 생성하는 간단한 예제를 만들어본다.


# In[2]:


# tensorflow, matplotlib.pyplot, numpy, minst 튜토리얼 데이터를 임포트한다.

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


# In[3]:


# 하이퍼 파라미터 설정하기
hp_total_epoch = 100
hp_batch_size = 100
hp_learning_rate = 0.0002
hp_n_hidden = 256
hp_n_input = 28*28
hp_n_noise = 128  # 생성자의 입력값으로 사용할 노이즈의 크기

# 플레이스 홀더 설정
# GAN도 비지도 학습이므로 Y값이 존재하지 않는다.
# 하지만 구분자에 넣을 이미지가 실제 이미지와 생성된 가짜이미지 두개이며,
# 가짜 이미지는 노이즈에서 생성할 예정이므로 노이즈를 입력할 플레이스홀더 Z를 추가한다.
X = tf.placeholder(tf.float32, [None, hp_n_input])
Z = tf.placeholder(tf.float32, [None, hp_n_noise])


# In[4]:


# 변수 설정

# 첫번째로 생성자(Generator) 신경망에 사용할 변수를 설정한다.
# 노이즈입력 -> 은닉층 -> 출력층으로 가는 형태이다.
# 생성자의 출력층 갯수는 구분자에서의 당연히 원본 이미지 픽셀수와 같아야 한다.
VGen_W1 = tf.Variable(tf.random_normal([hp_n_noise, hp_n_hidden], stddev=0.01))
VGen_b1 = tf.Variable(tf.zeros([hp_n_hidden]))
VGen_W2 = tf.Variable(tf.random_normal([hp_n_hidden, hp_n_input]))
VGen_b2 = tf.Variable(tf.zeros([hp_n_input]))

# 그 다음으로 구분자(Discriminator) 신경망에 사용할 변수를 설정한다.
# 은닉층 갯수는 생성자 때와 동일하다. 
# 구분자는 출력층 갯수로 딱 하나로써 진짜와 얼마나 가까운가를 판단하는 값이 된다.
VDis_W1 = tf.Variable(tf.random_normal([hp_n_input, hp_n_hidden], stddev=0.01))
VDis_b1 = tf.Variable(tf.zeros([hp_n_hidden]))
VDis_W2 = tf.Variable(tf.random_normal([hp_n_hidden, 1], stddev=0.01))
VDis_b2 = tf.Variable(tf.zeros([1]))


# In[5]:


# 이제 생성자와 구분자 신경망을 구성해보자

# 생성자 신경망 함수이다.
# 활성화 함수는 sigmoid를 적용하였다.
def Generator(noise_z):
    hidden = tf.nn.relu(
        tf.matmul(noise_z, VGen_W1) + VGen_b1)
    output = tf.nn.sigmoid(
        tf.matmul(hidden, VGen_W2) + VGen_b2)
    
    return output

# 구분자 신경망 함수이다.
# 역시나 활성화 함수는 sigmoid가 들어갔다.
def Discriminator(inputs):
    hidden = tf.nn.relu(
        tf.matmul(inputs, VDis_W1) + VDis_b1)
    output = tf.nn.sigmoid(
        tf.matmul(hidden, VDis_W2) + VDis_b2)
    
    return output


# In[6]:


# 이번엔 무작위 노이즈를 만들어주는 간단한 유틸리티 함수를 만들자
def GenerateNoise(_batchSize, _numNoise):
    return np.random.normal(size=(_batchSize, _numNoise))

# 그 다음 생성자를 호출시켜 가짜 이미지를 만들고, 
# 가짜 이미지와 원본 이미지를 구분자에 넣어 진위여부를 판단하도록 한다.
G = Generator(Z)
D_gene = Discriminator(G)
D_real = Discriminator(X)


# In[7]:


# 이제는 손실값을 구해야 하는데 이번에는 두개가 필요하다.

# 1) 생성자가 만든 이미지를 구분자가 가짜라고 판단하도록 하는 손실값 (경찰 학습용)
# 2) 생성자가 만든 이미지를 진짜라고 판단하도록 하는 손실값 (위조지폐범 학습용)

# 1)의 경우는 D_real이 1에 가까워져야 하고, D_gene가 0에 가까워져야 한다.
loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_gene))
# 위의 한줄을 간단히 설명하자면 (D_real)과 (1 - D_gene)을 각각 로그처리하여 더한 값을 손실값으로 한다는 것이다.

# 2)의 경우는 가짜 이미지 판별값 D_gene을 1에 가깝게 만들기만 하면 된다.
loss_G = tf.reduce_mean(tf.log(D_gene))

# 결국 GAN에서의 학습은 loss_D와 loss_G를 최대화 시키는 것이다.
# 하지만, loss_D와 loss_G는 서로 연관되어 있으므로 두 손실값이 항상 같이 증가하지는 않을 것이다.
# loss_D가 증가하면 loss_G는 감소할 것이고, 반대로 loss_G가 증가하면 loss_D는 감소하게 될 것이다.


# In[8]:


# 이제 손실값들을 이용하여 학습시키는 일이 남았다
# 근데 여기서 주의점이 있다.
# loss_D를 구할때는 구분자 신경망에서 사용되는 변수들만 사용하고, 
# loss_G를 구할때는 생성자 신경망에서 사용되는 변수들만 사용해서 최적화해야 한다.
# 그렇게 해야 loss_D를 학습할때 생성자가 변하지 않고,
# loss_G를 학습할때는 구분자가 변하지 않기 때문이다.
D_var_list = [VDis_W1, VDis_b1, VDis_W2, VDis_b2]
G_var_list = [VGen_W1, VGen_b1, VGen_W2, VGen_b2]

# 이제 최적화 함수를 정의해보자.
# GAN 논문에 따르면 loss를 최대화 해야 하는 것이지만 
# 최적화에 쓸수 있는 함수는 minimize()밖에 없으므로.
# 최적화 하려는 loss_D와 loss_G에 음수를 붙여주도록 한다.
train_D = tf.train.AdamOptimizer(hp_learning_rate).minimize(-loss_D, var_list=D_var_list)
train_G = tf.train.AdamOptimizer(hp_learning_rate).minimize(-loss_G, var_list=G_var_list)

# minimize()내에서 var_list 매개변수의 인자를 주게 되면 해당 
# 변수 리스트내 요소들만 학습하게 된다는 것을 알수있다.


# In[14]:


# 이제 그래프는 모두 구성했고 세션을 만들어서 실행해보자.
# 이번에는 두개의 손실값을 학습해야 하므로 코드가 좀 틀릴 수 있다.
sess = tf.Session()
sess.run(tf.global_variables_initializer())

num_total_batch = int(mnist.train.num_examples / hp_batch_size)
res_loss_val_D, res_loss_val_G = 0, 0

# 미니배치로 학습을 반복한다.
for epoch in range(hp_total_epoch):
    for i in range(num_total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(hp_batch_size)
        noise = GenerateNoise(hp_batch_size, hp_n_noise)
        
        _, res_loss_val_D = sess.run(
            [train_D, loss_D],
            feed_dict={X: batch_xs, Z: noise})
        _, res_loss_val_G = sess.run(
            [train_G, loss_G],
            feed_dict={Z: noise})
        
    print(
        'Epoch:', '%04d' % epoch,
        'D loss: {:.4}'.format(res_loss_val_D),
        'G loss: {:.4}'.format(res_loss_val_G))
        
    # 확인용 이미지를 만들어보도록 하자.
    # 0, 9. 19, 29, ...번째 에포크마다 생성기로 이미지를 생성하여 눈으로 직접 확인해보도록 하자.
    if epoch == 0 or (epoch + 1) % 10 == 0:
        sample_size = 10
        noise = GenerateNoise(sample_size, hp_n_noise)
        samples = sess.run(G, feed_dict={Z: noise})
        
        fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))
        
        for i in range(sample_size):
            ax[0][i].set_axis_off()
            ax[1][i].set_axis_off()
            
            ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            ax[1][i].imshow(np.reshape(samples[i], (28, 28)))
            
        plt.savefig(
            'samples/{}.png'.format(str(epoch).zfill(3)),
            bbox_inches='tight')
        
        plt.close(fig)

print('최적화 완료.')


# In[ ]:




