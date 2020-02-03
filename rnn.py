import pandas as pd # 데이터 전처리
import numpy as np # 데이터 전처리
import random #데이터 전처리
from pandas import DataFrame #데이터 전처리
from collections import Counter #데이터 전처리

from tqdm import tqdm #시간 측정용

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer # model setting
from sklearn.model_selection import train_test_split  # model setting

from sklearn.naive_bayes import MultinomialNB  # model 관련
from sklearn.metrics import roc_auc_score  # model 성능 확인


train = pd.read_csv("train_smishing.csv", engine='python') #해당 14th data의 csv 파일 중 train.csv 불러오기
train.head(2)
train['text'] = train['text'].str.replace('[0-9]', ' ').str.replace('[A-z]',' ').str.replace('[-=+,./\?~!@#$^&*|\\\(\)\"\'\;:ㅡㅇ%]',' ')
#train['text'] = train['text'].str.replace('[0-9]', ' ').str.replace('X',' ').str.replace('x',' ').str.replace('[-=+,./\?~!@#$^&*|\\\(\)\"\'\;:ㅡㅇ%]',' ')

test = pd.read_csv("public_test_smishing.csv", engine='python')
test.head(2)

Counter(train['smishing'])

random.seed(2019) #반복 수행시에도 동일한 결과 나올 수 있도록 시드 번호 지정
train_nsm_list=list(train[train['smishing']!=1].index)
train_nsmishing=random.sample(train_nsm_list, 11750 )

random.seed(2019)
train_sm_list=list(train[train['smishing']==1].index)
train_smishing=random.sample(train_sm_list, 850 ) #0.066과 제일 비슷하게 나올 수 있도록  train data under sampling

train_xx=train.iloc[train_smishing+train_nsmishing,:].reset_index(drop=True) #under sampling으로 나온 index들로 train data 선별

train_yy=DataFrame(train['smishing'],columns=['smishing']) 
train_yyy=train_yy.iloc[train_smishing+train_nsmishing,:].reset_index(drop=True)


test['smishing']=2 #train data와 동일한 형태 생성을 위해 임의의 숫자를 추가 #이후 스미싱 여부 확률 값으로 덮어 씌워짐
test_xx=DataFrame(test['text'])
test_yyy=DataFrame(test['smishing'])

train_xx.shape,train_yyy.shape,test_xx.shape,test_yyy.shape

# 토크나이즈 단계
import konlpy
from konlpy.tag import Mecab
tokenizer = Mecab()

train_doc = [ ( tokenizer.pos(x), y ) for x, y in tqdm( zip( train_xx['text'], train_yyy['smishing'] ) )  ] # Mecab를 활용하여 text를 토큰화 시킴
test_doc = [ ( tokenizer.pos(x), y ) for x, y in tqdm( zip( test_xx['text'], test_yyy['smishing'] ) )  ]

# 불용어처리 단계
stopwords = ['은행', '광고', '상품', '대출','사장','무료', '수신', '거부',
             '상담', '기록', '님', '고객', '고객님', '리브', 'Liiv', '최대', '카톡', '친구', '여신', '금리', '거부', '어플', '다운', '거부',
             '을', '를', '이', '가', '은', '는', '께', '서' ,'에', '와', '과', '고', '의', '보다', '한테', '되도록', '처럼', '에', '위해', 
             '들', '에', '게']

def get_couple(_words): #필요없는 단어들 없애는 함수
    global stopwords
    _words = [x for x in _words if x[0] not in stopwords]
    l = len(_words)
    for i in range(l):
        yield _words[i][0]
        

X_train, Y_train = [], []   # X_train : 토크나이즈 과정을 거친 단어 모음. Y_train : 스미싱여부 원 핫 인코드(러닝을 위한 답안지)
for lwords in train_doc:
    Y_train.append(lwords[1])
    temp = []
    for x in get_couple(lwords[0]):
        temp.append("{}".format(x))
    X_train.append(" ".join(temp))

    
X_test = []   # 테스트 데이터 셋
for lwords in test_doc:
    temp = []
    for x in get_couple(lwords[0]):
        temp.append("{}".format(x))
    X_test.append(" ".join(temp))
    

# 모델 적용단계
import tensorflow as tf

import tensorflow.keras
import tensorflow.keras.layers

v=TfidfVectorizer()
v.fit(X_train)
vec_x_train= v.transform(X_train).toarray()
vec_x_test= v.transform(X_test).toarray()

train_input, test_input, train_label, test_label = train_test_split(vec_x_train, Y_train, test_size=0.3) 
VOCAB_SIZE = vec_x_train.shape[0] + 1
WORD_EMBEDDING_DIM = 64
BUFFER_SIZE = 10000
BATCH_SIZE = 16
train_dataset = tf.data.Dataset.from_tensor_slices((train_input, train_label)) 
test_dataset = tf.data.Dataset.from_tensor_slices((test_input, test_label)) 
train_dataset = train_dataset.shuffle(BUFFER_SIZE) 
train_dataset = train_dataset.batch(BATCH_SIZE) 
test_dataset = test_dataset.batch(BATCH_SIZE) 
model=tensorflow.keras.Sequential(name='embedding')
model.add(tensorflow.keras.layers.Embedding(VOCAB_SIZE, WORD_EMBEDDING_DIM)) 
model.add(tensorflow.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)))
model.add(tensorflow.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)))
model.add(tensorflow.keras.layers.Dense(64, activation='relu'))
model.add(tensorflow.keras.layers.Dropout(rate=0.2))
model.add(tensorflow.keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # 모델 훈련 
history = model.fit(train_dataset, epochs=10,
                    validation_data=test_dataset )


# CV
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits=5, shuffle=True, random_state=0) 
results = cross_val_score(model,vec_x_train, Y_train, cv=kfold, scoring = 'accuracy')

y_train_pred1=model.predict_proba(vec_x_train)
y_train_pred1_one= [ i[1]  for i in y_train_pred1]

y_test_pred1=model.predict_proba(vec_x_test)
y_test_pred1_one= [ i[1]  for i in y_test_pred1] 

# 답안지 작성과정.
submission=pd.read_csv("submission_제출양식.csv")
submission.head(2)

submission['smishing'] = y_test_pred1_one
submission.to_csv("14th_baseline_multi rnn.csv",index=False) #현재 결과물인 output2를 구글 드라이브에 submission_test라는 이름으로 저장