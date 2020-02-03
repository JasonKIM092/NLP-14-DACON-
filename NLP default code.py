import pandas as pd # 데이터 전처리
import numpy as np # 데이터 전처리
import random #데이터 전처리
from pandas import DataFrame #데이터 전처리
from collections import Counter #데이터 전처리

from tqdm import tqdm #시간 측정용

from sklearn.feature_extraction.text import CountVectorizer # model setting
from sklearn.model_selection import train_test_split  # model setting

from sklearn.naive_bayes import MultinomialNB  # model 관련
from sklearn.metrics import roc_auc_score  # model 성능 확인


train = pd.read_csv("train_smishing.csv", engine='python') #해당 14th data의 csv 파일 중 train.csv 불러오기
train.head(2)
train['text'] = train['text'].str.replace('[0-9]', ' ').str.replace('[A-z]',' ').str.replace('[-=+,./\?~!@#$^&*|\\\(\)\"\'\;:ㅡ%]',' ')

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
okt?
train_doc = [ ( tokenizer.pos(x) ) for x in tqdm(  train_xx['text']  )  ] # Mecab를 활용하여 text를 토큰화 시킴
test_doc = [ ( tokenizer.pos(x) ) for x in tqdm(  test_xx['text']  )  ]

import konlpy
from konlpy.tag import Okt
okt = Okt()

train_doc = [ ( okt.pos(x, stem=True), y ) for x, y in tqdm( zip( train_doc, train_yyy['smishing'] ) )  ]
test_doc = [ ( okt.pos(x, stem=True), y ) for x, y in tqdm( zip( test_doc, test_yyy['smishing'] ) )  ]

# 불용어처리 단계
stopwords = ['을', '를', '이', '가', '은', '는', '뿐', '만', '께', '께서', '님', '와', '과', '고', '의', '보다', '한테', '도', '처럼', '에', '꼭', '꼬옥', '위해', '중', '중인', '들', '분', '에', '에게']


def get_couple(_words): #필요없는 단어들 없애는 함수
    global stopwords
    _words = [x for x in _words if x[0] not in stopwords]
    l = len(_words)
    for i in range(l-1):
        yield _words[i][0], _words[i+1][0]
        

        
X_train, Y_train = [], []   # X_trai : 토크나이즈 과정을 거친 단어 모음. Y_train : 스미싱여부 원 핫 인코드(러닝을 위한 답안지)
for lwords in train_doc:
    Y_train.append(lwords[1])
    
    temp = []
    for x, y in get_couple(lwords[0]):
        temp.append("{}.{}".format(x, y))
    
    X_train.append(" ".join(temp))
    
    X_test = []
    
X_test = []   # 테스트 데이터 셋
for lwords in test_doc:
    
    temp = []
    for x, y in get_couple(lwords[0]):
        temp.append("{}.{}".format(x, y))
    
    X_test.append(" ".join(temp))
    
# CountVecotrizer를 사용하여서 학습 데이터를 바탕으로 벡터화를 시키고 이를 test data에 적용한다   
# MultinomialNB 모델의 경우, 데이터를 벡터화하여 적용을 한다. 타 예제코드(영자신문 분석)에서도 해당과정을 적용한다. 이유는 모르겠지만, 한개의 전처리 과정임!!!
v=CountVectorizer()

v.fit(X_train)

vec_x_train= v.transform(X_train).toarray()
vec_x_test= v.transform(X_test).toarray()

# 모델 적용단계
m1= MultinomialNB()   # 자연어 나이브 베이즈 모델중 하나.
m1.fit(vec_x_train,Y_train)

y_train_pred1=m1.predict_proba(vec_x_train)
y_train_pred1_one= [ i[1]  for i in y_train_pred1]

y_test_pred1=m1.predict_proba(vec_x_test)
y_test_pred1_one= [ i[1]  for i in y_test_pred1]

# 답안지 작성과정.
submission=pd.read_csv("submission_제출양식.csv")
submission.head(2)

submission['smishing'] = y_test_pred1_one
submission.to_csv("14th_baseline_multi.csv",index=False) #현재 결과물인 output2를 구글 드라이브에 submission_test라는 이름으로 저장