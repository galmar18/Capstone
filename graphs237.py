# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 13:18:59 2019

@author: mg
"""
import pandas as pd
import numpy as np  

###find current working directory

import os
cwd = os.getcwd()
print(cwd)

###plotly

import plotly
plotly.__version__

import plotly.graph_objs as go
##import plotly.plotly as py
###example
"""
plotly.offline.plot({
    "data": [go.Scatter(x=[1, 2, 3, 4], y=[4, 3, 2, 1])],
    "layout": go.Layout(title="hello world")
}, auto_open=True)

"""
#datasets

df_graphs = pd.read_csv('finalgraphs.csv' )

df_graphs.head(10)

##df_train.loc[(df_train['user_id'] == '00RL8Z82B2Z1')]


datag = pd.read_csv('final12_246.csv', header=0)
datag.head(10)
datag.columns
###click graph
not_click=(datag['click_out'] == 0).sum()
click=(datag['click_out'] == 1).sum()

plotly.offline.plot({
    "data": [go.Bar(x=['click', 'not_click'],y=[click, not_click])],
     "layout": go.Layout(title="Clicks")
}, auto_open=True)
###device graph

mobile=(datag['mobile'] == 1).sum()
mobile
tablet=(datag['tablet'] == 1).sum()
desktop=(datag['desktop'] == 1).sum()

plotly.offline.plot({
    "data": [go.Bar(x=['mobile', 'tablet', 'desktop'],y=[mobile, tablet, desktop])],
     "layout": go.Layout(title="Device")
}, auto_open=True)


#
"""
df_train.platform.value_counts()
df_platform_graph=df_train.platform.value_counts()
df_platform_graph.head(10)
df_platform=df_platform_graph.head(10)
df_platform=df_platform.sort_values()
df_platform
df_platform.plot(x='platform',kind='barh')



df_platform.plot(x='platform',kind='barh')

plotly.offline.plot({
    "data": [go.Bar(x=['mobile', 'tablet', 'desktop'],y=['platform'])],
     "layout": go.Layout(title="Platform")
}, auto_open=True)

ax=df_platform_graph.sort_values().plot(kind='barh')

"""
s = df_graphs['platform'].value_counts() 
s


new = pd.DataFrame({'platform':s.index, 'count':s.values})  
new.head(10)
df_platform2=new.head(10)
df_platform2=df_platform2.sort_values(by=['count'])



plotly.offline.plot({
    "data": [go.Bar(x=df_platform2['count'],y=df_platform2['platform'],orientation = 'h')],
     "layout": go.Layout(title="Sessions per Platform")
}, auto_open=True)


##duration


#data['duration_sec'].max()



##x = data['duration_sec']
##data = [go.Histogram(x=x)]


plotly.offline.plot({
    "data": [go.Histogram(x=datag['duration_sec'])],
     "layout": go.Layout(title="Duration" ,xaxis=dict(
        range=[0, 20000]
    ),
    yaxis=dict(
        range=[0, 15000]
))
}, auto_open=True)


##stack graph actions
"""
filter_selection1=(datag['filter selection'] != 0).sum()
filter_selection0=(datag['filter selection']).sum()-(datag['filter selection'] != 0).sum()

interaction_item_deals1=(datag['interaction item deals'] != 0).sum()
interaction_item_deals0=(datag['interaction item deals']).sum()-(datag['interaction item deals'] != 0).sum()

interaction_item_image1=((datag['interaction item image'] != 0 )& ((datag['click_out'] == 1)|(datag['click_out'] == 0))).sum()
interaction_item_image1
interaction_item_image0=(datag['interaction item image']).count()-(datag['interaction item image'] != 0).sum()

interaction_item_info1=(datag['interaction item info']!= 0).sum()
interaction_item_info0=(datag['interaction item info']).sum()-(datag['interaction item info']!= 0).sum()

interaction_item_rating1=(datag['interaction item rating'] != 0).sum()
interaction_item_rating0=(datag['interaction item rating'] == 0).sum()

search_for_item1=(datag['search for item'] != 0).sum()
search_for_item0=(datag['search for item'] == 0).sum()

search_for_destination1=(datag['search for destination'] != 0).sum()
search_for_destination0=(datag['search for destination'] == 0).sum()

search_for_poi1=(datag['search for poi'] != 0).sum()
search_for_poi0=(datag['search for poi'] == 0).sum()
"""
###### sosto

filter_selection1=(datag['filter selection'] != 0).sum()
filter_selection0=(datag['filter selection'] == 0).sum()

interaction_item_deals1=(datag['interaction item deals'] != 0).sum()
interaction_item_deals0=(datag['interaction item deals'] == 0).sum()

interaction_item_image1=(datag['interaction item image'] != 0).sum()
interaction_item_image0=(datag['interaction item image'] == 0).sum()

interaction_item_info1=(datag['interaction item info'] != 0).sum()
interaction_item_info0=(datag['interaction item info'] == 0).sum()

interaction_item_rating1=(datag['interaction item rating'] != 0).sum()
interaction_item_rating0=(datag['interaction item rating'] == 0).sum()

search_for_item1=(datag['search for item'] != 0).sum()
search_for_item0=(datag['search for item'] == 0).sum()

search_for_destination1=(datag['search for destination'] != 0).sum()
search_for_destination0=(datag['search for destination'] == 0).sum()

search_for_poi1=(datag['search for poi'] != 0).sum()
search_for_poi0=(datag['search for poi'] == 0).sum()


trace1 = go.Bar(
    x=['filter selection', 'interaction item deals', 'interaction item image',
       'interaction_item_info','interaction item rating','search for item',
       'search for destination','search for poi'],
    y=[filter_selection1, interaction_item_deals1, interaction_item_image1,
       interaction_item_info1,interaction_item_rating1,search_for_item1,
       search_for_destination1,search_for_poi1],
    name='Yes'
)
trace2 = go.Bar(
     x=['filter selection', 'interaction item deals', 'interaction item image',
       'interaction_item_info','interaction item rating','search for item',
       'search for destination','search for poi'],
    y=[filter_selection0, interaction_item_deals0, interaction_item_image0,
       interaction_item_info0,interaction_item_rating0,search_for_item0,
       search_for_destination0,search_for_poi0],
    name='No'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group',title="Actions During Session"
)
plotly.offline.plot({
    "data" : go.Figure(data=data, layout=layout)
}, auto_open=True)

    




##heatmap

  
data2 = datag[['filter selection','interaction item deals',
              'interaction item image','interaction item info','interaction item rating',
              'search for destination','search for item','search for poi','duration_sec','click_out']]
x=data2['interaction item image'].sum()
x


data3=data2.corr()

datah = [go.Heatmap( z=data3.values.tolist(),
                     x=['filter selection','interaction item deals',
              'interaction item image','interaction item info','interaction item rating',
              'search for destination','search for item','search for poi','duration_sec','click_out'], 
                     y=['filter selection','interaction item deals',
              'interaction item image','interaction item info','interaction item rating',
              'search for destination','search for item','search for poi','duration_sec','click_out'],
                colorscale='Viridis')]

plotly.offline.plot({
    "data": datah,
     "layout": go.Layout(title="Actions Correlation Table" )
}, auto_open=True)
    

##scatter matrix



#scatters

import plotly.express as px
#scatter1

##iris = px.data.iris()
fig = px.scatter_matrix(data2, 
    dimensions=['interaction item image','interaction item info',
             'duration_sec'],
    color='click_out')
##fig.show()
plotly.offline.plot({
    "data": fig,
     "layout": go.Layout(title="Scatter1" )
}, auto_open=True)

##scatter2
fig = px.scatter_matrix(data2, 
    dimensions=['filter selection','interaction item deals',
             'duration_sec'],
    color='click_out')
##fig.show()
plotly.offline.plot({
    "data": fig,
     "layout": go.Layout(title="Scatter2" )
}, auto_open=True)

##scatter3

fig = px.scatter_matrix(data2, 
    dimensions=['interaction item rating',
              'search for destination',
             'duration_sec'],
    color='click_out')
##fig.show()
plotly.offline.plot({
    "data": fig,
     "layout": go.Layout(title="Scatter3" )
}, auto_open=True)

##scatter4

fig = px.scatter_matrix(data2, 
    dimensions=['search for item','search for poi','duration_sec'],
    color='click_out')
##fig.show()
plotly.offline.plot({
    "data": fig,
     "layout": go.Layout(title="Scatter4" )
}, auto_open=True)

####bar
#datag.groupby(['click_out'])['filter selection'].sum()

#df.groupby(['Category','scale']).sum().groupby('Category').cumsum()
"""
filter_selection_c0=datag['filter selection'][(datag["click_out"] == 0) ].sum()
filter_selection_c0
filter_selection_c1=datag['filter selection'][(datag["click_out"] == 1) ].sum()
filter_selection_c1


datag.head(10)




data5=data2.groupby(['click_out']).sum().groupby('click_out').cumsum()
data5.columns
data5

data5=data2.groupby(['click_out']).sum()


data5=df = data5.drop('duration_sec', 1)

filter_selection_c0=data5['filter selection'].loc[0]
filter_selection_c1=data5['filter selection'].loc[1]

interaction_item_deals_c0=data5['interaction item deals'].loc[0]
interaction_item_deals_c1=data5['interaction item deals'].loc[1]

interaction_item_image_c0=data5['interaction item image'].loc[0]
interaction_item_image_c1=data5['interaction item image'].loc[1]

interaction_item_info_c0=data5['interaction item info'].loc[0]
interaction_item_info_c1=data5['interaction item info'].loc[1]

interaction_item_rating_c0=data5['interaction item rating'].loc[0]
interaction_item_rating_c1=data5['interaction item rating'].loc[1]

search_for_destination_c0=data5['search for destination'].loc[0]
search_for_destination_c1=data5['search for destination'].loc[1]

search_for_item_c0=data5['search for item'].loc[0]
search_for_item_c1=data5['search for item'].loc[1]

search_for_poi_c0=data5['search for poi'].loc[0]
search_for_poi_c1=data5['search for poi'].loc[1]

trace1 = go.Bar(
    x=['filter selection', 'interaction item deals', 'interaction item image',
       'interaction item info','interaction item rating','search for item',
       'search for destination','search for poi'],
    y=[filter_selection_c1, interaction_item_deals_c1, interaction_item_image_c1,
       interaction_item_info_c1,interaction_item_rating_c1,search_for_item_c1,
       search_for_destination_c1,search_for_poi_c1],
    name='Click'
)
trace2 = go.Bar(
     x=['filter selection', 'interaction item deals', 'interaction item image',
       'interaction item info','interaction item rating','search for item',
       'search for destination','search for poi'],
    y=[filter_selection_c0, interaction_item_deals_c0, interaction_item_image_c0,
       interaction_item_info_c0,interaction_item_rating_c0,search_for_item_c0,
       search_for_destination_c0,search_for_poi_c0],
    name='Not Click'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group',title="Actions in Relation to Clicks"
)
plotly.offline.plot({
    "data" : go.Figure(data=data, layout=layout)
}, auto_open=True)


filter_selection_c0=datag["filter selection"][(datag["click_out"] == 0) & (datag['filter selection'] != 0)].sum()

filter_selection_c1=datag["filter selection"][(datag["click_out"] == 1) ].sum()

interaction_item_deals_c0=datag["interaction item deals"][(datag["click_out"] == 0) ].sum()
interaction_item_deals_c1=datag["interaction item deals"][(datag["click_out"] == 1) ].sum()

interaction_item_image_c0=datag["interaction item image"][(datag['interaction item image'] != 0)].sum()
interaction_item_image_c0
interaction_item_image_c1=datag["interaction item image"][(datag["click_out"] == 1) ].sum()

interaction_item_info_c0=datag["interaction item info"][(datag["click_out"] == 0) ].sum()
interaction_item_info_c1=datag["interaction item info"][(datag["click_out"] == 1) ].sum()

interaction_item_rating_c0=datag["interaction item rating"][(datag["click_out"] == 0) ].sum()
interaction_item_rating_c1=datag["interaction item rating"][(datag["click_out"] == 1) ].sum()

search_for_item_c0=datag["search for item"][(datag["click_out"] == 0) ].sum()
search_for_item_c1=datag["search for item"][(datag["click_out"] == 1) ].sum()

search_for_destination_c0=datag["search for destination"][(datag["click_out"] == 0) ].sum()
search_for_destination_c1=datag["search for destination"][(datag["click_out"] == 1) ].sum()

search_for_poi_c0=datag["search for poi"][(datag["click_out"] == 0) ].sum()
search_for_poi_c1=datag["search for poi"][(datag["click_out"] == 1) ].sum()



trace1 = go.Bar(
    x=['filter selection', 'interaction item deals', 'interaction item image',
       'interaction item info','interaction item rating','search for item',
       'search for destination','search for poi'],
    y=[filter_selection_c1, interaction_item_deals_c1, interaction_item_image_c1,
       interaction_item_info_c1,interaction_item_rating_c1,search_for_item_c1,
       search_for_destination_c1,search_for_poi_c1],
    name='Click'
)
trace2 = go.Bar(
     x=['filter selection', 'interaction item deals', 'interaction item image',
       'interaction item info','interaction item rating','search for item',
       'search for destination','search for poi'],
    y=[filter_selection_c0, interaction_item_deals_c0, interaction_item_image_c0,
       interaction_item_info_c0,interaction_item_rating_c0,search_for_item_c0,
       search_for_destination_c0,search_for_poi_c0],
    name='Not Click'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group',title="Actions in Relation to Clicks"
)
plotly.offline.plot({
    "data" : go.Figure(data=data, layout=layout)
}, auto_open=True)



interaction_item_image1=(datag['interaction item image'] != 0).sum()
interaction_item_image1
interaction_item_image0=(datag['interaction item image'] == 0).sum()
interaction_item_image0.values.sum()

interaction_item_image3=(datag['interaction item image'] != 0).sum()
interaction_item_image3

interaction_item_image4=datag['interaction item image']
interaction_item_image4.values.sum()

interaction_item_image6=(datag['interaction item image'] != 0)
interaction_item_image6.sum()

interaction_item_image7=(datag['interaction item image'] == 0)
interaction_item_image7.sum()


##--------------------
filter_selectionc1=(((datag['interaction item image'] != 0) | (datag['interaction item image'] == 0) )& ((datag['click_out'] == 0))).sum()
c0=(((datag['interaction item image'] != 0) | (datag['interaction item image'] == 0) )& ((datag['click_out'] == 0))).sum()
c0
"""

filter_selectionc0=(((datag['filter selection'] != 0) ) & ((datag['click_out'] == 0))).sum()
filter_selectionc1=(((datag['filter selection'] != 0) ) & ((datag['click_out'] == 1))).sum()

interaction_item_dealsc0=(((datag['interaction item deals'] != 0) ) & ((datag['click_out'] == 0))).sum()
interaction_item_dealsc0
interaction_item_dealsc1=(((datag['interaction item deals'] != 0) ) & ((datag['click_out'] == 1))).sum()
interaction_item_dealsc1

interaction_item_imagec0=(((datag['interaction item image'] != 0) ) & ((datag['click_out'] == 0))).sum()
interaction_item_imagec0
interaction_item_imagec1=(((datag['interaction item image'] != 0) ) & ((datag['click_out'] == 1))).sum()
interaction_item_imagec1

interaction_item_infoc0=(((datag['interaction item info'] != 0) ) & ((datag['click_out'] == 0))).sum()
interaction_item_infoc1=(((datag['interaction item info'] != 0) ) & ((datag['click_out'] == 1))).sum()

interaction_item_ratingc0=(((datag['interaction item rating'] != 0) ) & ((datag['click_out'] == 0))).sum()
interaction_item_ratingc1=(((datag['interaction item rating'] != 0) ) & ((datag['click_out'] == 1))).sum()

search_for_itemc0=(((datag['search for item'] != 0) ) & ((datag['click_out'] == 0))).sum()
search_for_itemc1=(((datag['search for item'] != 0) ) & ((datag['click_out'] == 1))).sum()

search_for_destinationc0=(((datag['search for destination'] != 0) ) & ((datag['click_out'] == 0))).sum()
search_for_destinationc1=(((datag['search for destination'] != 0) ) & ((datag['click_out'] == 1))).sum()

search_for_poic0=(((datag['search for poi'] != 0) ) & ((datag['click_out'] == 0))).sum()
search_for_poic1=(((datag['search for poi'] != 0) ) & ((datag['click_out'] == 1))).sum()


trace1 = go.Bar(
    x=['filter selection', 'interaction item deals', 'interaction item image',
       'interaction item info','interaction item rating','search for item',
       'search for destination','search for poi'],
    y=[filter_selectionc1, interaction_item_dealsc1, interaction_item_imagec1,
       interaction_item_infoc1,interaction_item_ratingc1,search_for_itemc1,
       search_for_destinationc1,search_for_poic1],
    name='Click'
)
trace2 = go.Bar(
     x=['filter selection', 'interaction item deals', 'interaction item image',
       'interaction item info','interaction item rating','search for item',
       'search for destination','search for poi'],
    y=[filter_selectionc0, interaction_item_dealsc0, interaction_item_imagec0,
       interaction_item_infoc0,interaction_item_ratingc0,search_for_itemc0,
       search_for_destinationc0,search_for_poic0],
    name='Not Click'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group',title="Actions in Relation to Click Item"
)
plotly.offline.plot({
    "data" : go.Figure(data=data, layout=layout)
}, auto_open=True)
