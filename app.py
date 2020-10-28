# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 22:01:02 2020

@author: bhowku01
"""

#from pip._vendor import pkg_resources

#def get_version(package):
#    package = package.lower()
#    return next((p.version for p in pkg_resources.working_set if p.project_name.lower() == package), "No match")
#    get_version("io")

import streamlit as st
import pandas as pd
import numpy as np
#import os
import altair as alt

# Libraries needed for the tutorial


import requests
import io
    
# Downloading the csv file from your GitHub account

url = "https://raw.githubusercontent.com/kunalbhowmick/streamlit-forecast/main/DOW_2020-09-26.csv" # Make sure the url is the raw version of the file on GitHub
download = requests.get(url).content

# Reading the downloaded content and turning it into a pandas dataframe

df = pd.read_csv(io.StringIO(download.decode('utf-8')),parse_dates=['Date'])

# Printing out the first 5 rows of the dataframe

#print (df.head())


#os.chdir("C:\\Users\\bhowku01\\DL Forecast")
st.title('Love you Nupur')
st.title('Time series Analyser')
#df = pd.read_csv('DOW_2020-09-26.csv',parse_dates=['Date'])
if st.checkbox('Show dataframe'):
 st.write(df)
 
st.subheader('A Quick Plot of the Selected Series')
series=st.selectbox('Which series would you like to use',
('Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'))
st.write('You selected:', series)
chart0=alt.Chart(df).mark_line().encode(
    x='Date',
    y=series,tooltip = [alt.Tooltip(series),
               alt.Tooltip('Date')]).properties(
    width=700,
    height=300,
    autosize=alt.AutoSizeParams(
        type='fit',
        contains='padding'
    )
)
    
st.write(chart0)

st.title('Missing Value Analysis')
import pandas as pd
#import numpy as np

#df_dow=pd.read_csv('DOW_2020-09-26.csv',parse_dates=['Date']) #parse_dates=True, or parse_dates=['column name']
df_dow=df
df_dow.index=df_dow['Date']
# Looking at the data structure
type(df_dow) # To check the type of the object being imported
df_dow.info() # Understand the columns datatpe within the dataframe

import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt

#dummy data
start_of_range=str(df_dow.index.min())
end_of_range=str(df_dow.index.max())
date_range = pd.date_range(start_of_range, end_of_range, freq='B')

df = pd.DataFrame(np.random.randint(1, 20, (date_range.shape[0], 1)))
df.index = date_range  # set index
#df_missing = df.drop(df.between_time('00:12', '00:14').index)

#check for missing datetimeindex values based on reference index (with all values)
missing_dates = df.index[~df.index.isin(df_dow.index)]

st.write(missing_dates)
st.write(missing_dates.strftime('%A'))# 0-Monday 1-Tue 2-Wed ...... Check if there were holidays on the missing days

st.title("Change points in time series")

# Here for the change point algorithim we are using the Bottom Up segmentation
#https://centre-borelli.github.io/ruptures-docs/detection/bottomup.html
'''Bottom-up change point detection is used to perform fast signal segmentation 
and is implemented in ruptures.detection.BottomUp. 
It is a sequential approach. Contrary to binary segmentation, 
which is a greedy procedure, bottom-up segmentation is generous: 
it starts with many change points and successively deletes the less significant ones.
First, the signal is divided in many sub-signals along a regular grid. 
Then contiguous segments are successively merged according to a measure of how similar they are. 
See for instance [BUKCHP01] or [BUFry07] for an algorithmic analysis of ruptures.detection.BottomUp. 
The benefits of bottom-up segmentation includes low complexity (of the order of O(nlogn), 
where n is the number of samples), the fact that it can extend any single change point 
detection method to detect multiple changes points and that it can work whether the number 
of regimes is known beforehand or not.'''


import matplotlib.pyplot as plt
import ruptures as rpt
    
    # detection
algo = rpt.BottomUp(model="rbf").fit(df_dow[series].to_numpy())
result = algo.predict(pen=10)
# display
#change_point_graph=

rpt.show.display(df_dow[series].to_numpy(), result)


st.pyplot()

# Get the date corresponding to the Change points identified by the algorithim
change_point_dates=df_dow.index[[result[0:len(result)-1]]]
st.write(change_point_dates)




# Creating certain Features based on the dates
df_dow['weekday']=df_dow.index.weekday
df_dow['Month']=df_dow.index.month
df_dow['Day']=df_dow.index.day
df_dow['Year']=df_dow.index.year

date_feature = st.selectbox('Which date feature would you like to use?',
('weekday', 'Month', 'Day','Year'))
st.write('You selected:', date_feature)



#date_feature='weekday'
#series='Close'
group=df_dow[[date_feature,series]]
grp=round(group.groupby([date_feature]).sum(),0)
grp[date_feature]=grp.index
st.write(grp) # For Validation and can be used after transposing

chart1=alt.Chart(grp).mark_line().encode(
    x=date_feature,
    y=series)

st.write(chart1)        

st.title("LSTM Model for the time series")


#Structuring the data for LSTM
st.subheader('Just testing something')
x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)

