import pandas as pd # 데이터 전처리
import numpy as np # 데이터 전처리
import random #데이터 전처리
from pandas import DataFrame #데이터 전처리
from collections import Counter #데이터 전처리

from tqdm import tqdm #시간 측정용

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer # model setting
from sklearn.model_selection import train_test_split  # model setting

from sklearn.naive_bayes import MultinomialNB  # model 관련
from sklearn.metrics import roc_auc_score  # model 성능 확인


train = pd.read_csv("train_smishing.csv", engine='python') #해당 14th data의 csv 파일 중 train.csv 불러오기
train.head(2)
train['text'] = train['text'].str.replace('[0-9]', ' ').str.replace('[A-z]',' ').str.replace('[-=+,./\?~!@#$^&*|\\\(\)\"\'\;:ㅡㅇ%]',' ')

#test = pd.read_csv("public_test_smishing.csv", engine='python')
#test.head(2)

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

train_xx.shape,train_yyy.shape

# 토크나이즈 단계
import konlpy
from konlpy.tag import Mecab
tokenizer = Mecab()

train_doc = [ ( tokenizer.pos(x), y ) for x, y in tqdm( zip( train_xx['text'], train_yyy['smishing'] ) )  ] # Mecab를 활용하여 text를 토큰화 시킴

# 불용어처리 단계
stopwords = ['은행', '광고', '상품', '대출',                   
             '상담', '기록', '님', '고객', '고객님', '리브', 'Liiv', '최대', '카톡', '친구', '여신', '금리', '거부', '어플', '다운', '거부',
             '을', '를', '이', '가', '은', '는', '께', '서' ,'에', '와', '과', '고', '의', '보다', '되도록', '처럼', '에', '위해', 
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

    
# CountVecotrizer를 사용하여서 학습 데이터를 바탕으로 벡터화를 시키고 이를 test data에 적용한다   
# MultinomialNB 모델의 경우, 데이터를 벡터화하여 적용을 한다. 타 예제코드(영자신문 분석)에서도 해당과정을 적용한다. 이유는 모르겠지만, 한개의 전처리 과정임!!!
v=CountVectorizer()

v.fit(X_train)

vec_x_train= v.transform(X_train).toarray()

# 모델 적용단계

from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier

# CV
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, random_state=0)
print("훈련 세트의 크기: {}   테스트 세트의 크기: {}".format(len(x_train), len(x_test)))

best_score = 0
for n_estimators in [2,3,4,5,6,7,8,9,10, 20, 30, 40]:
    ensemble = Pipeline([('vect', TfidfVectorizer(min_df = 0, sublinear_tf = True)),
                         ('ensemble',BaggingClassifier(base_estimator=MultinomialNB(), n_estimators=n_estimators)),
                         ])
    ensemble.fit(x_train, y_train)
    score = ensemble.score(x_test, y_test)
    if score > best_score:
        best_score = score
        best_parameters = {'n_estimators': n_estimators}

print("최고 점수: {:.2f}".format(best_score))
print("최적 파라미터: {}".format(best_parameters))


x_trainval, x_test, y_trainval, y_test = train_test_split(X_train, Y_train)
# 훈련+검증 세트를 훈련 세트와 검증 세트로 분할
x_train, x_valid, y_train, y_valid = train_test_split(x_trainval, y_trainval, random_state=1)
print("훈련 세트의 크기: {}   검증 세트의 크기: {}   테스트 세트의 크기:"
      " {}\n".format(len(x_train), len(x_valid), len(x_test)))

best_score = 0
for n_estimators in [2,3,4,5,6,7,8,9,10, 20, 30, 40]:
    ensemble = Pipeline([('vect', TfidfVectorizer(min_df = 0, sublinear_tf = True)),
                         ('ensemble',BaggingClassifier(base_estimator=MultinomialNB(), n_estimators=n_estimators)),
                         ])
    ensemble.fit(x_train, y_train)
        # 검증 세트로 SVC를 평가합니다
    score = ensemble.score(x_valid, y_valid)
        # 점수가 더 높으면 매개변수와 함께 기록합니다
    if score > best_score:
        best_score = score
        best_parameters = {'n_estimators' : n_estimators}

# 훈련 세트와 검증 세트를 합쳐 모델을 다시 만든 후
# 테스트 세트를 사용해 평가합니다
ensemble = Pipeline([('vect', TfidfVectorizer(min_df = 0, sublinear_tf = True)), ('ensemble',BaggingClassifier(base_estimator=MultinomialNB(), n_estimators=n_estimators)), ])
ensemble.fit(x_trainval, y_trainval)
test_score = ensemble.score(x_test, y_test)
print("검증 세트에서 최고 점수: {:.2f}".format(best_score))
print("최적 파라미터: ", best_parameters)
print("최적 파라미터에서 테스트 세트 점수: {:.2f}".format(test_score))