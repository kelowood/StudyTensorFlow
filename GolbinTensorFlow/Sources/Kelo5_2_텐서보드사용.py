#!/usr/bin/env python
# coding: utf-8

# In[10]:


# 텐서보드를 사용해보도록 하자.
# 일단 플레이스 홀더 정의까지는 이전 5.1 장 코드와 완전 동일하다.

import tensorflow as tf
import numpy as np

tf.reset_default_graph()

data = np.loadtxt(
    './data.csv',
    delimiter=',', 
    unpack=True, 
    dtype='float32')

x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])

global_step = tf.Variable(
    0, 
    trainable=False, 
    name='global_step')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


# In[11]:


# 이제 신경망 그래프 모델 정의를 하는 부분인데 아래와 이 작성한다.

# 제1계층
with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.), name='W1')
    L1 = tf.nn.relu(tf.matmul(X, W1))

# with tf.name_scope로 묶은 블록은 텐서보드에서 한계층 내부를 표현해주는 것이다.
# 또한 W1 변수 뒤에 name값을 W1 스트링 값으로 이름을 지정함으로써 
# 텐서보드에서 해당 이름의 변수가 어디에서 사용되는지 쉽게 알 수 있다.
# 다른 부분도 처리해보도록 하겠다.

# 제2계층
with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.), name='W2')
    L2 = tf.nn.relu(tf.matmul(L1, W2))

# 출력층
with tf.name_scope('output'):
    W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.), name='W3')
    model = tf.matmul(L2, W3)

# 최적화 부분
with tf.name_scope('optimizer'):
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=Y, 
            logits=model))
    
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(cost, global_step=global_step)
    
    # 다음으로 손실값을 추적하기 위해 수집할 값을 지정하는 코드를 작성한다.
    # 아래 tf.summary.scalar()는 값이 하나인 텐서를 수집할때 사용된다.
    tf.summary.scalar('cost', cost)


# In[12]:


# 신경망 모델 학습
sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())

ckpt = tf.train.get_checkpoint_state('model')

if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())


# In[13]:


# 이제 tf.summary.merge_all() 함수를 통해 앞서 지정된 텐서들을 수집한다.
# 단, 이것도 텐서이므로 sess.run()을 통해야 실제 실행이 가능하다.
# 그다음 tf.summary.FileWriter() 함수를 통해 그래프와 텐서 값을 저장할 디렉토리를 설정한다.
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('logs', sess.graph)


# In[14]:


# 그 다음 최적화 실행 코드를 이전과 마찬가지로 작성한다. (단, 5.1과는 틀리게 100번 돌린다.)
for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})
    
    print('Step: %d, ' % sess.run(global_step),
          'Cost: %3f' % sess.run(cost, feed_dict={X: x_data, Y: y_data}))
    
    # 여기에서 앞서 merged로 모아둔 텐서의 값들을 계산하여 수집한 뒤, 
    # writer.add_summary()를 통하여 해당 값들을 지정 디렉터리에 저장하게 된다.
    summary = sess.run(merged, feed_dict={X: x_data, Y: y_data})
    writer.add_summary(summary, global_step=sess.run(global_step))


# In[15]:


# 모델 저장과 예측 부분은 이전 장과 동일하다.
saver.save(sess, 'model\dnn.ckpt', global_step=global_step)

prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
print('실제값:', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))


# In[16]:


# 이 프로그램을 실행하면 logs라는 폴더가 생성되며, 내부에 파일이 생성되게 된다.
# 명령 프롬프트에서 아래와 같이 명령을 실행하도록 하자
# tensorboard --logdir=logs
# 이후 //피씨이름:6006 이라는 주소가 뜨는데 이것을 확인하고 아래 주소로 접속하면 텐서보드를 확인할 수 있다.
# http://localhost:6006

# Scalar 탭에서는 tf.summary.scalar('cost', cost)로 수집한 손실값의 변화를 그래프로 확인할 수 있다.
# Graphs 탭에서는 with tf.name_scope로 그룹핑한 결과들을 그래프로 확인해 볼 수 있다.


# In[ ]:




