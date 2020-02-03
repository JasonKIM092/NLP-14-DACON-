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

test = pd.read_csv("public_test_smishing.csv", engine='python')
test.head(2)

Counter(train['smishing'])

random.seed(2019) #반복 수행시에도 동일한 결과 나올 수 있도록 시드 번호 지정
train_nsm_list=list(train[train['smishing']!=1].index)

random.seed(2019)
train_sm_list=list(train[train['smishing']==1].index)

train_xx=train.iloc[train_nsm_list+train_sm_list,:].reset_index(drop=True) #under sampling으로 나온 index들로 train data 선별

train_yy=DataFrame(train['smishing'],columns=['smishing']) 
train_yyy=train_yy.iloc[train_nsm_list+train_sm_list,:].reset_index(drop=True)


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
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Activation
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical

v=TfidfVectorizer()
v.fit(X_train)
vec_x_train= v.transform(X_train).toarray()
vec_x_test= v.transform(X_test).toarray()

train_input, test_input, train_label, test_label = train_test_split(vec_x_train, Y_train, test_size=0.3) 

train_input = np.array(train_input).reshape((train_input.shape[0], train_input.shape[1], 1))
test_input = np.array(test_input).reshape((test_input.shape[0], test_input.shape[1], 1))

y_data = np.concatenate((train_label, test_label))
y_data = to_categorical(y_data)

train_label = y_data[:8820]
test_label = y_data[8820:]

# 데이터의 모양 출력하기
print(train_input.shape)
print(test_input.shape)
print(train_label.shape)
print(test_label.shape)

# 기본 RNN 모델을 구현하기 위한 함수
def stacked_vanilla_rnn():
    model = Sequential()
    model.add(SimpleRNN(10, input_shape = (10894,1), return_sequences = True))   # return_sequences parameter has to be set True to stack
    model.add(SimpleRNN(10, return_sequences = False))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    
    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    
    return model

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss') # 조기종료 콜백함수 정의
model = KerasClassifier(build_fn = stacked_vanilla_rnn, epochs = 3, batch_size = 100, verbose = 1, callbacks=[early_stopping])
model.fit(train_input, train_label)

vec_x_train = np.array(vec_x_train).reshape((vec_x_train.shape[0], vec_x_train.shape[1], 1))
vec_x_test = np.array(vec_x_test).reshape((vec_x_test.shape[0], vec_x_test.shape[1], 1))


y_train_pred1=model.predict_proba(vec_x_train)
y_train_pred1_one= [ i[1]  for i in y_train_pred1]

y_test_pred1=model.predict_proba(vec_x_test)
y_test_pred1_one= [ i[1]  for i in y_test_pred1] 

# 답안지 작성과정.
submission=pd.read_csv("submission_제출양식.csv")
submission.head(2)

submission['smishing'] = y_test_pred1_one
submission.to_csv("14th_baseline_multi rnn.csv",index=False) #현재 결과물인 output2를 구글 드라이브에 submission_test라는 이름으로 저장