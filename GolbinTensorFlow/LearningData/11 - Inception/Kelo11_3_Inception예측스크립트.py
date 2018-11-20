#!/usr/bin/env python
# coding: utf-8

# In[3]:


# 이 스크립트를 작성 및 실행전 아래와 같은 작업을 하였다.
# 
# 1) 자료 다운로드 
# http://download.tensorflow.org/example_images/flowers_photos.tgz 를 받고 압축을 푼다.
#
# 2) 디렉토리 및 파일 위치 맞추기
# 현재 기본 디렉토리 (이 파일이 있는 디렉토리)에 workspace라는 디렉토리를 생성한다.
# 다운로드된 파일의 압축을 풀고 아래와 같이 파일과 디렉토리를 맞춘다.
# /기본디렉토리
#     /retrain.py
#     /workspace
#         /flower_photos
#             /daisy
#             /dandelion
#             /roses
#             /sunflowers
#             /tulips
#
# 3) retrain.py 실행 -> 꽃사진 학습
# 기본 디렉토리 폴더에서 아래와 같이 retrain.py 파이썬 명령을 실행하였다.
# C:\...\ python retrain.py \
#         --bottleneck_dir=workspace\bottlenecks \         학습할 사진을 인셉션용 학습 데이터로 변환해서 저장해둘 디렉토리
#         --model_dir=workspace\inception \                인셉션 모델을 내려받을 경로
#         --output_graph=workspace\flowers_graph.pb \      학습된 모델(.pb)을 저장할 경로
#         --output_labels=workspace\flowers_labels.txt \   레이블 이름들을 저장해둘 파일 경로
#         --image_dir workspace\flower_photos \            원본 이미지 경로
#         --how_many_training_steps 1000                   반복 학습 횟수
# 


# In[2]:


import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys


# In[3]:


# 텐서플로우의 유용한 모듈인 app.flags를 사용하여 스크립트에서 받을 옵션과 기본값을 설정한다.
# tf.app 모듈을 사용하면 터미널이나 명령 프롬프트에서 입력받는 옵션을 쉽게 처리할 수 있다.
tf.app.flags.DEFINE_string(
    "output_graph",
    "./workspace/flowers_graph.pb",
    "학습된 신경망이 저장된 위치")
tf.app.flags.DEFINE_string(
    "output_labels",
    "./workspace/flowers_labels.txt",
    "학습할 레이블 데이터 파일")
tf.app.flags.DEFINE_boolean(
    "show_image",
    True,
    "이미지 추론 후 이미지를 보여줍니다.")
FLAGS = tf.app.flags.FLAGS
# 이제 FLAGS.output_graph, FLAGS.output_labels, FLAGS.show_image 로 해당 값에 접근할 수가 있다.


# In[4]:


# retrain.py 를 이용하여 학습을 진행하면 workspace 디렉토리의 flowers_labels.txt 파일에 꽃의 이름을 전부 저장해두게 된다.
# 한줄에 하나씩 들어가 있고, 줄번호를 해당 꽃 이름의 인덱스로 하여 학습을 진행하였다.
# flowers_labels.txt
# daisy
# dandelion
# roses
# sunflowers
# tulips


# In[ ]:


# 그리고 예측 후 출력되는 값은 아래와 같이 모든 인덱스에 대하여 총합이 1인 확률을 나열한 softmax값이 된다.

# [ 0.65098846, 0.1175265,  0.01479027, 0.20992769, 0.00676709]
#   daisy       dandelion   roses       sunflowers  tulips


# In[ ]:


def main(_):
    # 테스트 결과를 확인할 때 사용하기 위해 파일에 담긴 꽃 이름들을 가져와 배열로 저장해 둔다.
    # 이름을 출력할때 사용할 것이다.
    labels = [line.rstrip() for line in tf.gfile.GFile(FLAGS.output_labels)]
    
    # 아까 전 retrain.py를 통한 학습으로 flowers_graph.pb 파일이 생성되었었다.
    # 이것은 학습 결과를 프로토콜 버퍼(Protocol Buffers, pb) 데이터 형식으로 저장해둔 파일이다.
    # 꽃 사진 예측을 위해 이 파일을 읽어서 신경망 그래프를 생성할 것이다.
    # 텐서플로우를 이용하면 아래처럼 매우 쉽게 처리할 수 있다.
    with tf.gfile.FastGFile(FLAGS.output_graph, 'rb') as fp:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fp.read())
        tf.import_graph_def(graph_def, name='')
    
    with tf.Session() as sess:
        # 읽어들인 신경망 모델에서 예측에 사용할 텐서를 지정한다.
        # 저장되어 있는 모델에서 최종 출력층은 'final_result:0'라는 이름의 텐서이다.
        logits = sess.graph.get_tensor_by_name('final_result:0')
        
        # 예측 스크립트를 실행할 때 주어진 이름의 이미지 파일을 읽어들인뒤, 그 이미지를 예측 모델에 넣어 예측을 실행한다.
        # 아래의 DecodeJpeg/contents:0은 이미지 데이터를 입력값으로 넣을 플레이스 홀더 이름이다.
        image = tf.gfile.FastGFile(sys.argv[1], 'rb').read()
        prediction = sess.run(logits, {'DecodeJpeg/contents:0': image})
        
    # 프로토콜 버퍼(pb) 방식으로 저장되어 있는 타 모델들도 이런 방식으로 쉽게 읽어들여서 사용할 수 있다.
    
    # 이제 예측 결과를 출력하는 코드를 작성해 보자.
    # 다음 코드로 앞에서 읽어온 레이블 (꽃 이름)들에 해당하는 모든 예측 결과를 출력한다.
    print('=== 예측 결과 ===')
    for i in range(len(labels)):
        name = labels[i]
        score = prediction[0][i]
        print('%s (%.2f%%)' % (name, score * 100))
        
    # 그리고 주어진 이름의 이미지 파일을 matplotlib 모듈을 이용하여 출력한다.
    if FLAGS.show_image:
        img = mpimg.imread(sys.argv[1])
        plt.imshow(img)
        plt.show()

# 마지막으로 스크립트 실행시 주어진 옵션들과 함께 main() 함수를 실행하는 코드를 작성해 준다.
if __name__ == "__main__":
    tf.app.run()
    
# 이것을 py 파일로 만들어서 실행해 보도록 하자. 이 파일은 명령프롬프트에서 아래와 같이 열면 된다.
# > python [파이썬파일명] workspace/flower_photos/[원하는꽃_디렉토리]/[이미지파일명].jpg

