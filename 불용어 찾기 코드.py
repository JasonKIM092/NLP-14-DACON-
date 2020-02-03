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


import re
s1=train_xx['text']
train_xx.dtypes

s2=train_xx.loc[train_xx['smishing']==1,'text']

pat1 = re.compile('이자', flags=re.IGNORECASE)
sum( np.where(pd.isnull(s1.str.findall(pat1).str[0]),False,True) )  # 전체에서
sum( np.where(pd.isnull(s2.str.findall(pat1).str[0]),False,True) )  # 850 스미싱문자
sum( np.where(pd.isnull(s2.str.findall(pat1).str[0]),False,True) )/sum( np.where(pd.isnull(s1.str.findall(pat1).str[0]),False,True) )
