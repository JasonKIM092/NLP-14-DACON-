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

# train data set
train = pd.read_csv("train_smishing.csv", engine='python') #해당 14th data의 csv 파일 중 train.csv 불러오기
train.head(2)
train['text'] = train['text'].str.replace('[0-9]', ' ').str.replace('X',' ').str.replace('x',' ').str.replace('[-=+,./\?~!@#$^&*|\\\(\)\"\'\;:ㅡㅇ%]',' ')

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

train_xx.shape,train_yyy.shape

# 토크나이즈 단계
import konlpy
from konlpy.tag import Mecab
tokenizer = Mecab()

train_doc = [ ( tokenizer.pos(x), y ) for x, y in tqdm( zip( train_xx['text'], train_yyy['smishing'] ) )  ] # Mecab를 활용하여 text를 토큰화 시킴

# 불용어처리 단계
stopwords = ['은행', '광고', '상품', '대출',                   
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

# 토크나이즈 단계
import konlpy
from konlpy.tag import Mecab
tokenizer = Mecab()

train_doc = [ ( tokenizer.pos(x), y ) for x, y in tqdm( zip( train_xx['text'], train_yyy['smishing'] ) )  ] # Mecab를 활용하여 text를 토큰화 시킴


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


# 모델 적용단계
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
#from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier


clf_1 = Pipeline([('vect', CountVectorizer()),
                  ('clf', MultinomialNB(alpha=3.0)),
                  ])

clf_2 = Pipeline([('vect', TfidfVectorizer()),
                  ('clf', MultinomialNB(alpha=3.0)),
                  ])

clf_3 = Pipeline([('vect', CountVectorizer()),
                  ('clf', BernoulliNB(alpha=3.0)),
                  ])

clf_4 = Pipeline([('vect', TfidfVectorizer()),
                  ('clf', BernoulliNB(alpha=3.0)),
                  ])
    
clf_5 = Pipeline([('vect', CountVectorizer()),
                  ('clf', RandomForestClassifier(n_estimators=100)),
                  ])
    
clf_6 = Pipeline([('vect', TfidfVectorizer()),
                  ('clf', RandomForestClassifier(n_estimators=100)),
                  ])
#m1 = MultinomialNB(alpha = 3)
#m1.fit(vec_x_train,Y_train)
#m2 = BernoulliNB(alpha = 3)
# m2.fit(vec_x_train,Y_train)
#m3 = RandomForestClassifier(n_estimators=100)
#m3.fit(vec_x_train,Y_train)
# , max_features=0.001, max_depth = 10)

# ensemble
#ensemble = VotingClassifier(estimators=[('mnb', m1), ('bnb',m2), ('rf', m3)], voting='soft') # 86~87점
#ensemble.fit(X_train,Y_train)
clf_ensemble_1 = VotingClassifier(estimators=[('clf1', clf_1), ('clf2', clf_2), ('clf3', clf_3), ('clf4', clf_4), ('clf5', clf_5), ('clf6', clf_6)], voting='soft')   # voting = 'hard' 안댐 : predict_proba is not available when voting='hard'
clf_ensemble_2 = VotingClassifier(estimators=[('clf2', clf_2), ('clf4', clf_4), ('clf6', clf_6)], voting='soft')
clf_ensemble_3 = VotingClassifier(estimators=[('clf1', clf_1), ('clf3', clf_3), ('clf5', clf_5)], voting='soft')


ensemble1 = Pipeline([('vect', TfidfVectorizer(min_df = 6, max_df=len(X_train)*0.9,sublinear_tf = True)),
                     ('ensemble',BaggingClassifier(base_estimator=MultinomialNB(), n_estimators=1000, max_samples = 0.5)),
                     ])

#ensemble2 = Pipeline([('vect', TfidfVectorizer(min_df = 6, max_df=len(X_train)*0.9,sublinear_tf = True)),
#                     ('ensemble',BaggingClassifier(base_estimator=BernoulliNB(), n_estimators=1000, max_samples = 0.5)),
#                     ])

#ensemble3 = Pipeline([('vect', TfidfVectorizer(min_df = 6, max_df=len(X_train)*0.9,sublinear_tf = True)),
#                     ('ensemble',BaggingClassifier(base_estimator=MultinomialNB(), n_estimators=1000, max_samples = 0.6)),
#                     ])
#    
#ensemble4 = Pipeline([('vect', TfidfVectorizer(min_df = 6, max_df=len(X_train)*0.9,sublinear_tf = True)),
#                     ('ensemble',BaggingClassifier(base_estimator=BernoulliNB(), n_estimators=1000, max_samples = 0.6)),
#                     ])
#
ensemble5 = Pipeline([('vect', TfidfVectorizer(min_df = 6, max_df=len(X_train)*0.9,sublinear_tf = True)),
                     ('ensemble',BaggingClassifier(base_estimator=MultinomialNB(), n_estimators=1000, max_samples = 0.7)),
                     ])

from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
from scipy.stats import sem

def evaluate_cross_validation(clf, X, y, K):
    cv = KFold(n_splits = K, shuffle=True, random_state = 0)
    scores = cross_val_score(clf, X, y, cv=cv)
    print (scores)
#    print ("Name : {0:}  \n Mean score: {1:.3f} (+/-{2:.3f})". format (clf.__dict__, np.mean(scores), sem(scores)))
    print ("Mean score: {0:.3f} (+/-{1:.3f}) \n". format (np.mean(scores), sem(scores)))

clfs = [ensemble1, ensemble5]
for clf in clfs:
    evaluate_cross_validation(clf, X_train, Y_train, 5)   # 3, 5, 10