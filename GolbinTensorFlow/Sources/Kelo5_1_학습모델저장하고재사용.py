#!/usr/bin/env python
# coding: utf-8

# In[68]:


import tensorflow as tf
import numpy as np

# 책에도 없는 내용! 그래프 중복 문제로 인해 체크포인트를 불러오지 못하는 문제의 해결책.
tf.reset_default_graph()

# 현재 data.csv는 아래와 같이 구성되어있다.
# 0, 0, 1, 0, 0
# 1, 0, 0, 1, 0
# 1, 1, 0, 0, 1
# 0, 0, 1, 0, 0
# 0, 0, 1, 0, 0
# 0, 1, 0, 0, 1

# 첫번째~두번째 열은 털, 날개 로 이루어진 특징값이고,
# 세번째~다섯번째 열은 기타, 포유류, 조류로 이루어진 분류 값이다.

data = np.loadtxt(
    './data.csv',
    delimiter=',', 
    unpack=True, 
    dtype='float32')
print(data)

# np.loadtxt()를 통해서 csv 파일을 불러오게 되는데, 
# 여기서 중요한 점은 unpack 값이 True로 하게 되면,
# 불러온 행렬 데이터를 "전치행렬" 시킨다는 점이다.
# 즉, data를 출력하면 아래와 같이 된다.
# [[ 0.  1.  1.  0.  0.  0.]
#  [ 0.  0.  1.  0.  0.  1.]
#  [ 1.  0.  0.  1.  1.  0.]
#  [ 0.  1.  0.  0.  0.  0.]
#  [ 0.  0.  1.  0.  0.  1.]]

# 즉, data의 0~1행이 특징값으로 바뀌었으며,
# 2~4행이 분류 값으로 바뀌게 되었다.

print(data[0:2])
print(data[2:])

x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])

# data[0:2]는 인덱스 0값의 행부터 시작하여 두개의 행값을 가져오는 것이다.
# [[ 0.  1.  1.  0.  0.  0.]
#  [ 0.  0.  1.  0.  0.  1.]]
# 즉 특징값만을 가져온다.(털, 날개)

# data[2:]는 인덱스 2값의 행부터 시작하여 나머지 모두의 행값을 가져오는 것이다.
# [[ 1.  0.  0.  1.  1.  0.]
#  [ 0.  1.  0.  0.  0.  0.]
#  [ 0.  0.  1.  0.  0.  1.]]
# 즉 분류값만을 가져온다.(기타, 포유류, 조류)

# np.transpose()는 인자 행렬을 전치시키는 역활을 한다.
print(" ")
print("x_data")
print(x_data)
print("y_data")
print(y_data)

# 결과를 보면 알다시피 4.3 코드때와 같은 값이 나오게 된다.

# 이제 신경망 모델을 정의해보도록 하자.
# 그전에 먼저 모델을 저장할때 쓸 변수 하나를 만든다.
# 이 변수는 학습에 직접 사용되지 않고, 학습 횟수를 카운트하는 역활을 한다.
# 이것을 위하여 변수를 정의할때 trainable 값을 False로 주어야 한다.
global_step = tf.Variable(
    0, 
    trainable=False, 
    name='global_step_3')

# 이번 신경망은 4.3 때의 것과 비교하자면 편향 없이 가중치만 사용하고, 계층을 하나 더 늘린다.
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
L1 = tf.nn.relu(tf.matmul(X, W1))

W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.))
L2 = tf.nn.relu(tf.matmul(L1, W2))

W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.))
model = tf.matmul(L2, W3)

# 1계층 이후 은닉층 뉴런 개수는 10개이고, 2계층 이후 은닉층 뉴런 개수는 20개이다.
# 이렇게 신경망의 계층수와 은닉층 뉴런수를 늘리면 복잡도가 높은 문제를 해결하는데 도움이 된다.
# 하지만 이렇게 한다고 모든 문제가 해결되는것은 아니며, 지나치면 오히러 과적합이라는 다른 문제가 생길 수 있다.
# 즉, 신경망 모델 구성시 계층 및 뉴런수를 적절하게 최적화하는것이 핵심이다.

# 다음은 비용함수 및 최적화 함수의 정의이다.
# 전때와 마찬가지로 최적화 함수는 AdamOptimizer를 사용한다.

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost, global_step=global_step)

# 여기서 최적화 함수의 minimize() 사용시 global_step 매개변수가 있고 여기에 g_step 값을 넣어주었다.
# 이렇게 하면 최적화를 한번 수행할때마다 g_step 변수 값이 1개씩 늘어나게 된다.

# 이제 세션을 열고 실행해보자
sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())

# tf.global_variables()는 앞서 정의한 변수들을 가져오는 함수이다.
# 이 함수를 통해 앞에서 정의한 변수들을 모두 가져온다.
# 그후 Saver에 넣음으로써 이 변수들을 파일에 저장하거나, 이전에 학습한 결과를 불러와 담을 수 있게 된다.



# 다음은 ./model 디렉토리에 기존에 학습해둔 모델이 있는지 확인해 본다.
# 만약 모델이 있다면 saver.restore 함수를 사용하여 학습된 값들을 불러온다.
# 만약 모델이 없다면 변수를 새로 초기화한다.
# 학습된 모델을 저장한 파일을 체크포인트 파일(Checkpoint file)이라 부른다.

ckpt = tf.train.get_checkpoint_state("model\\")

if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print(ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())
    
# 최적화를 수행해보도록 하자.
# 지난번과는 다르게 step값을 쓰지 않고 윗부분에서 정의한 global_step 이용하여
# 학습을 몇번째 진행하고 있는지를 출력한다.
# global_step 텐서 변수이므로 sess.run()을 이용해야 한다.
for step in range(2):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})
    
    print('Step: %d, ' % sess.run(global_step),
          'Cost: %3f' % sess.run(cost, feed_dict={X: x_data, Y: y_data}))
    
# 최적화가 끝난 학습된 변수들을 체크포인트 파일로 저장하도록 한다.
saver.save(sess, "model\dnn.ckpt", global_step=global_step)

# 지난번과 마찬가지로 정확도를 측정하는 코드를 작성한다.
prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
print('실제값:', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))


# In[ ]:




