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

train_doc = [ ( tokenizer.nouns(x), y ) for x, y in tqdm( zip( train_xx['text'], train_yyy['smishing'] ) )  ] # Mecab를 활용하여 text를 토큰화 시킴
test_doc = [ ( tokenizer.nouns(x), y ) for x, y in tqdm( zip( test_xx['text'], test_yyy['smishing'] ) )  ]

# 불용어처리 단계

stopwords = ['은행', '광고', '상품', '대출', '사장', '무료', '수신', '거부', '수수료', '안내', '영업부', '년', '정부', '지원', '이자',
             '상담', '기록', '님', '고객', '고객님', '리브', 'Liiv', '최대', '카톡', '친구', '여신', '금리', '거부', '어플', '다운', '거부']

def get_couple(_words): #필요없는 단어들 없애는 함수
    global stopwords
    _words = [x for x in _words if x[0] not in stopwords]
    l = len(_words)
    for i in range(l-1):
        yield _words[i][0], _words[i+1][0]

X_train, Y_train = [], []

for lwords in train_doc:
    Y_train.append(lwords[1])
    temp = []
    for x in lwords[0]:
        temp.append(x)
    X_train.append(" ".join(temp))
   
X_test = []

for lwords in test_doc:   
    temp = []
    for x in lwords[0]:
        temp.append(x)
    X_test.append(" ".join(temp))


# 모델 적용단계
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
#from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier

ensemble = Pipeline([('vect', TfidfVectorizer(min_df =0, sublinear_tf = True, ngram_range = (1, 2), analyzer = 'word')),  # ngram_range = (1, 2), analyzer = 'word'
                     ('ensemble',BaggingClassifier(base_estimator=MultinomialNB(), n_estimators=100, max_samples = 0.5)),
                     ])
ensemble.fit(X_train, Y_train)

# CV
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits=5, shuffle=True, random_state=0) 
results = cross_val_score(ensemble,X_train, Y_train, cv=kfold, scoring = 'accuracy')


y_train_pred1=ensemble.predict_proba(X_train)
y_train_pred1_one= [ i[1]  for i in y_train_pred1]

y_test_pred1=ensemble.predict_proba(X_test)
y_test_pred1_one= [ i[1]  for i in y_test_pred1] 

# 답안지 작성과정.
submission=pd.read_csv("submission_제출양식.csv")
submission.head(2)

submission['smishing'] = y_test_pred1_one
submission.to_csv("14th_baseline_multi pipeline x tfidf bag.csv",index=False) #현재 결과물인 output2를 구글 드라이브에 submission_test라는 이름으로 저장