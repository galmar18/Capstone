# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:59:51 2019

@author: mg
"""


import pandas as pd
import numpy as np  

###find current working directory

import os
cwd = os.getcwd()
print(cwd)

#submission csv check

df_submission = pd.read_csv('submission_popular.csv' )
df_submission

df_submission.shape
df_submission.info(verbose=True)
#print(df_submission.columns)
df_submission.isnull().values.any()
df_submission.duplicated() 

df_submission.head()




##metadata csv check

df_metadata = pd.read_csv('item_metadata.csv' )
df_metadata

df_metadata.shape
df_metadata.info(verbose=True)
#print(df_submission.columns)
df_metadata.isnull().values.any()
df_metadata.duplicated() 

df_metadata.head()

##train csv check


df_train = pd.read_csv('train.csv' )
df_train

df_train.shape
df_train.info(verbose=True)
#print(df_submission.columns)
df_train.isnull().values.any()
df_train.isnull().sum()


df_train.duplicated() 

df_train.head()


df_train.sort_values(by=['user_id','timestamp'])

##unique

df_train['user_id'].nunique()
df_train['session_id'].nunique()


#df user-session grouped and sorted


df_train.groupby(['user_id','session_id'],sort=True)['session_id'].count()
df_session=df_train.groupby(['user_id','session_id'],sort=True)['session_id'].count()
df_session

#df test users
df_train.loc[(df_train['user_id'] == '0001VQMGUI65')]

df_train.loc[(df_train['user_id'] == '0008BO33KUQ0')]

df_train.loc[(df_train['user_id'] == '0006W0R5A5V8')]


#duration

df_train.loc[(df_train['user_id'] == '4HI37W7EA5FP')]

#df_train[['user_id','device']].groupby(['user_id','device'],sort=True)['user_id','device'].count()>1

#df_train[['user_id','session_id','device']].groupby(['user_id','session_id'],sort=True)['device'].min()

##df_train[['user_id','device']].groupby(['user_id','device'],sort=True)['user_id','device'].count()

#
#df_train[['user_id','timestamp']].groupby(['user_id','timestamp'],sort=True)['timestamp'].count()

#Devises per user more than one
#df_train[['user_id','device']].groupby(['user_id','device'],sort=True)['user_id'].count()

#df_train[['user_id','session_id','device']].groupby(['user_id','session_id','device'],sort=True)
#df_device.describe()

####device
df_train.groupby(['user_id','session_id','device'], as_index=False,sort=True).count()
df_device=df_train.groupby(['user_id','session_id','device'], as_index=False,sort=True).count()
df_device



##3 devices
df_train['device'].nunique()


##platform
#df_train.groupby(['user_id','session_id','platform'],sort=True)['platform'].count()
#df_platform=df_train.groupby(['user_id','session_id','platform'],sort=True)['platform'].count()
#df_platform

##55 platforms
df_train['platform'].nunique()


###action_type =10

df_train['action_type'].nunique()
df_train.groupby(['user_id','session_id','action_type'], as_index=False,sort=True).count()
df_action_type10=df_train.groupby(['user_id','session_id','action_type'],as_index=False,sort=True).count()
df_action_type10

df_action_type10.columns


df_action_type10[df_action_type10['action_type'] == 'clickout item']  


####all
df_train.groupby(['user_id','session_id','device','platform','action_type'], as_index=False,sort=True).count()
df_all=df_train.groupby(['user_id','session_id','device','platform','action_type'], as_index=False,sort=True).count()
df_all

df_all2=df_all.groupby(['user_id','session_id','device','platform'], as_index=False,sort=True).count()
df_all2.columns

df_all2

df_all2=df_all.groupby(['user_id','session_id','device','platform'], as_index=False,sort=True).count()
df_all2.columns

df_all2
##final

df30=df_all.head(30)
df30

df_final=df_all2[['user_id','session_id','device','platform']]
df_final


##device
df_final['mobile'] = ""
df_final['tablet'] = ""
df_final['desktop'] = ""



df_final

df_final['mobile'] = np.where(df_final['device'] =='mobile', 'yes','no')
df_final['tablet'] = np.where(df_final['device'] =='tablet', 'yes','no')
df_final['desktop'] = np.where(df_final['device'] =='desktop', 'yes','no')

##platform

df_final['platform_us'] = ""
df_final['platform_uk'] = ""
df_final['platform_de'] = ""
df_final['platform_au'] = ""
df_final['platform_ch'] = ""
df_final['platform_jp'] = ""

df_final['platform_us'] = np.where(df_final['platform'] =='US', 'yes','no')
df_final['platform_uk'] = np.where(df_final['platform'] =='UK', 'yes','no')
df_final['platform_de'] = np.where(df_final['platform'] =='DE', 'yes','no')
df_final['platform_au'] = np.where(df_final['platform'] =='AU', 'yes','no')
df_final['platform_ch'] = np.where(df_final['platform'] =='CH', 'yes','no')
df_final['platform_jp'] = np.where(df_final['platform'] =='JP', 'yes','no')

####add platforms

df_train.platform.unique()

df_final['platform_br'] = ""
df_final['platform_fi'] = ""
df_final['platform_mx'] = ""
df_final['platform_fr'] = ""
df_final['platform_it'] = ""
df_final['platform_at'] = ""
df_final['platform_hk'] = ""
df_final['platform_ru'] = ""

df_final['platform_br'] = np.where(df_final['platform'] =='BR', 'yes','no')
df_final['platform_fi'] = np.where(df_final['platform'] =='FI', 'yes','no')
df_final['platform_mx'] = np.where(df_final['platform'] =='MX', 'yes','no')
df_final['platform_fr'] = np.where(df_final['platform'] =='FR', 'yes','no')
df_final['platform_it'] = np.where(df_final['platform'] =='IT', 'yes','no')
df_final['platform_at'] = np.where(df_final['platform'] =='AT', 'yes','no')
df_final['platform_hk'] = np.where(df_final['platform'] =='HK', 'yes','no')
df_final['platform_ru'] = np.where(df_final['platform'] =='RU', 'yes','no')

df_final['platform_in'] = ""
df_final['platform_co'] = ""
df_final['platform_es'] = ""
df_final['platform_cl'] = ""
df_final['platform_be'] = ""
df_final['platform_ar'] = ""
df_final['platform_nl'] = ""
df_final['platform_ca'] = ""
df_final['platform_ie'] = ""

df_final['platform_in'] = np.where(df_final['platform'] =='IN', 'yes','no')
df_final['platform_co'] = np.where(df_final['platform'] =='CO', 'yes','no')
df_final['platform_es'] = np.where(df_final['platform'] =='ES', 'yes','no')
df_final['platform_cl'] = np.where(df_final['platform'] =='CL', 'yes','no')
df_final['platform_be'] = np.where(df_final['platform'] =='BE', 'yes','no')
df_final['platform_ar'] = np.where(df_final['platform'] =='AR', 'yes','no')
df_final['platform_nl'] = np.where(df_final['platform'] =='NL', 'yes','no')
df_final['platform_ca'] = np.where(df_final['platform'] =='CA', 'yes','no')
df_final['platform_ie'] = np.where(df_final['platform'] =='IE', 'yes','no')

df_final['platform_se'] = ""
df_final['platform_th'] = ""
df_final['platform_my'] = ""
df_final['platform_hu'] = ""
df_final['platform_ph'] = ""
df_final['platform_za'] = ""
df_final['platform_pe'] = ""
df_final['platform_id'] = ""
df_final['platform_nz'] = ""
df_final['platform_cz'] = ""

df_final['platform_se'] = np.where(df_final['platform'] =='SE', 'yes','no')
df_final['platform_th'] = np.where(df_final['platform'] =='TH', 'yes','no')
df_final['platform_my'] = np.where(df_final['platform'] =='MY', 'yes','no')
df_final['platform_hu'] = np.where(df_final['platform'] =='HU', 'yes','no')
df_final['platform_ph'] = np.where(df_final['platform'] =='PH', 'yes','no')
df_final['platform_za'] = np.where(df_final['platform'] =='ZA', 'yes','no')
df_final['platform_pe'] = np.where(df_final['platform'] =='PE', 'yes','no')
df_final['platform_id'] = np.where(df_final['platform'] =='ID', 'yes','no')
df_final['platform_nz'] = np.where(df_final['platform'] =='NZ', 'yes','no')
df_final['platform_cz'] = np.where(df_final['platform'] =='CZ', 'yes','no')

df_final['platform_kr'] = ""
df_final['platform_rs'] = ""
df_final['platform_bg'] = ""
df_final['platform_dk'] = ""
df_final['platform_hr'] = ""
df_final['platform_tr'] = ""
df_final['platform_il'] = ""
df_final['platform_sg'] = ""
df_final['platform_ec'] = ""
df_final['platform_sk'] = ""
df_final['platform_pl'] = ""


df_final['platform_kr'] = np.where(df_final['platform'] =='KR', 'yes','no')
df_final['platform_rs'] = np.where(df_final['platform'] =='RS', 'yes','no')
df_final['platform_bg'] = np.where(df_final['platform'] =='BG', 'yes','no')
df_final['platform_dk'] = np.where(df_final['platform'] =='DK', 'yes','no')
df_final['platform_hr'] = np.where(df_final['platform'] =='HR', 'yes','no')
df_final['platform_tr'] = np.where(df_final['platform'] =='TR', 'yes','no')
df_final['platform_il'] = np.where(df_final['platform'] =='IL', 'yes','no')
df_final['platform_sg'] = np.where(df_final['platform'] =='SG', 'yes','no')
df_final['platform_ec'] = np.where(df_final['platform'] =='EC', 'yes','no')
df_final['platform_sk'] = np.where(df_final['platform'] =='SK', 'yes','no')
df_final['platform_pl'] = np.where(df_final['platform'] =='PL', 'yes','no')

df_final['platform_no'] = ""
df_final['platform_aa'] = ""
df_final['platform_tw'] = ""
df_final['platform_pt'] = ""
df_final['platform_ro'] = ""
df_final['platform_uy'] = ""
df_final['platform_gr'] = ""
df_final['platform_ae'] = ""
df_final['platform_si'] = ""
df_final['platform_cn'] = ""
df_final['platform_vn'] = ""

df_final['platform_no'] = np.where(df_final['platform'] =='NO', 'yes','no')
df_final['platform_aa'] = np.where(df_final['platform'] =='AA', 'yes','no')
df_final['platform_tw'] = np.where(df_final['platform'] =='TW', 'yes','no')
df_final['platform_pt'] = np.where(df_final['platform'] =='PT', 'yes','no')
df_final['platform_ro'] = np.where(df_final['platform'] =='RO', 'yes','no')
df_final['platform_uy'] = np.where(df_final['platform'] =='UY', 'yes','no')
df_final['platform_gr'] = np.where(df_final['platform'] =='GR', 'yes','no')
df_final['platform_ae'] = np.where(df_final['platform'] =='AE', 'yes','no')
df_final['platform_si'] = np.where(df_final['platform'] =='SI', 'yes','no')
df_final['platform_cn'] = np.where(df_final['platform'] =='CN', 'yes','no')
df_final['platform_vn'] = np.where(df_final['platform'] =='VN', 'yes','no')


df_final.shape

list(df_final)



#####action om the last step
##df_train.sort_values('step', ascending=False).drop_duplicates(['Sp','Mt'])

#df_step=df_train[['user_id','session_id','step','action_type']]
#df_step
#df_step2=df_step.sort_values('step', ascending=False).drop_duplicates(['user_id','session_id',])
#df_step2
#df_step2.sort_values(['user_id', 'session_id'], ascending=[True, True])


df_last=df_train[['user_id', 'session_id','step','action_type']].sort_values(['user_id', 'session_id','step'], ascending=[True, True,True]).groupby(['user_id', 'session_id']).tail(1)
df_last



df_final['click'] = ""

df_final2 = pd.merge(df_final,df_last, on=['user_id','session_id'])
df_final2

df_final2['click'] = np.where(df_final2['action_type'] =='clickout item', 'yes','no')
df_final2

###actions per session
df_actions=df_all[['user_id', 'session_id','step','action_type']]
df_actions
df_pivoted = df_actions.pivot_table(index=['user_id','session_id'], columns='action_type', values='step').reset_index()

df_pivoted

df_pivoted2=df_pivoted.fillna(0)
df_pivoted2
df_pivoted3=df_pivoted2.round(0)

df_pivoted3


df_final3 = pd.merge(df_final2,df_pivoted3, on=['user_id','session_id'])
df_final3

df_final4=df_final3.drop(columns=['device','platform','action_type'])
df_final4


##duration

df_time_min=df_train[['user_id', 'session_id','timestamp']].sort_values(['user_id', 'session_id'], ascending=[True, True]).groupby(['user_id', 'session_id']).head(1)
df_time_min


df_time_max=df_train[['user_id', 'session_id','timestamp']].sort_values(['user_id', 'session_id'], ascending=[True, True]).groupby(['user_id', 'session_id']).tail(1)
df_time_max

#from datetime import datetime
##datetime.fromtimestamp(df_time_min['timestamp'])
##datetime.fromtimestamp(df_time_max['timestamp'])

df_time_min['timestamp']=pd.to_datetime(df_time_min['timestamp'], unit='s')
df_time_min

df_time_max['timestamp']=pd.to_datetime(df_time_max['timestamp'], unit='s')
df_time_max




df_time = pd.merge(df_time_min,df_time_max, on=['user_id','session_id'])
df_time

df_time['duration']=(df_time['timestamp_y']-df_time['timestamp_x'])

df_time['duration_sec']=df_time['duration'].dt.total_seconds()

df_time2=df_time.drop(columns=['duration','timestamp_x','timestamp_y'])
df_time2

df_final5 = pd.merge(df_final4,df_time2, on=['user_id','session_id'])
df_final5

#drop clickout
df_final6=df_final5.drop(columns=['clickout item'])
df_final6

df_final6.shape

list(df_final6)
####final list

df_final6['click_out']=df_final6['click']

df_final7=df_final6.drop(columns=['click'])
df_final7
#import datetime
#import time
#x = time.strptime('    ','%H:%M:%S')
#datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()


df_final7.shape

list(df_final7)

df_final7




#replace
df_final8=df_final7.replace({'yes': 1, 'no': 0})

##df_final8=df_final7.replace(to_replace=r'^yes^', value=1, regex=True)

#df_final8=df_final7.replace('no', 0)
#df_final8

#df_final9=df_final8.replace('yes', 1)
#df_final9
#dataframe to csv

#w['female'] = w['female'].map({'female': 1, 'male': 0})
df_final8

#df_final8.loc[(df_train['user_id'] == '0008BO33KUQ0')]

#df_final8.to_csv("final8_56.csv", index = False, sep=',')

df_final8.dtypes


df_final8['change of sort order']    = df_final8['change of sort order'].astype(int)
df_final8['filter selection']        = df_final8['filter selection'].astype(int)
df_final8['interaction item deals']  = df_final8['interaction item deals'].astype(int)
df_final8['interaction item image']  = df_final8['interaction item image'].astype(int)
df_final8['interaction item info']   = df_final8['interaction item info'].astype(int)
df_final8['interaction item rating'] = df_final8['interaction item rating'].astype(int)
df_final8['search for destination']  = df_final8['search for destination'].astype(int)
df_final8['search for item']         = df_final8['search for item'].astype(int)
df_final8['search for poi']          = df_final8['search for poi'].astype(int)
df_final8['duration_sec']            = df_final8['duration_sec'].astype(int)

df_final8.dtypes

df_final8

df_final8.to_csv("final8_56.csv", index = False, sep=',')
###


################
#drop user_id session id
import pandas as pd  
 

df_final9 = pd.read_csv("final8_56.csv") 
df_final9
cols = df_final9['duration_sec']
df_final9[df_final9['duration_sec'] <0]

df_final10=df_final9.drop(['user_id', 'session_id'], axis=1)
df_final10
df_final10.to_csv("final10_166.csv", index = False, sep=',')
