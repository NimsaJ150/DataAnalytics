#%%



#%% md

TODO: chapter numbers

# Data Analysis

1. 2019 vs 2020 differences
2. Prediction of Severity with traffic data [day, time, state??]

---
## 1 Imports
### 1.1 Libraries

#%%

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

#%% md

### 1.2 Data

#%%

file_path = './US_Accidents_Dec20_updated.csv'

# If file exists
if os.path.isfile(file_path):
    data_ori = pd.read_csv(file_path)
else:
    # TODO: insert code to download file
    pass
"""
PATH_HOME = os.getcwd()
PATH_DATA = os.path.join(PATH_HOME, "data")
PATH_OUTPUT = os.path.join(PATH_HOME, "output")
filename = os.path.join(PATH_DATA, "amazon_electronics.json")
if not os.path.exists(filename):
    display("Reading from URL..")
    df_amz = pd.read_json("https://www.dropbox.com/s/o9jxaeax4mascd3/Electronics_5.json?dl=1", lines = True)
    if not os.path.exists(PATH_DATA):
        os.mkdir(PATH_DATA)
    df_amz.to_pickle(filename)

else:
    display("Reading from hard drive..")
    df_amz = pd.read_pickle(filename)

df_amz.head()
"""

#%% md

---
## 2 Definitions

#%%
#all parameters in original data

column_list = [
    'ID',
    'Severity',
    'Start_Time',
    'End_Time',
    'Start_Lat',
    'Start_Lng',
    'End_Lat',
    'End_Lng',
    'Distance(mi)',
    'Description',
    'Number',
    'Street',
    'Side',
    'City',
    'County',
    'State',
    'Zipcode',
    'Country',
    'Timezone',
    'Airport_Code',
    'Weather_Timestamp',
    'Temperature(F)',
    'Wind_Chill(F)',
    'Humidity(%)',
    'Pressure(in)',
    'Visibility(mi)',
    'Wind_Direction',
    'Wind_Speed(mph)',
    'Precipitation(in)',
    'Weather_Condition',
    'Amenity',
    'Bump',
    'Crossing',
    'Give_Way',
    'Junction',
    'No_Exit',
    'Railway',
    'Roundabout',
    'Station',
    'Stop',
    'Traffic_Calming',
    'Traffic_Signal',
    'Turning_Loop',
    'Sunrise_Sunset',
    'Civil_Twilight',
    'Nautical_Twilight',
    'Astronomical_Twilight'
]

# defining all columns in original data with numeric values
numeric_columns = [
    'Severity',
    'Start_Lat',
    'Start_Lng',
    'End_Lat',
    'End_Lng',
    'Distance(mi)',
    'Number',
    'State',
    'Zipcode',
    'Temperature(F)',
    'Wind_Chill(F)',
    'Humidity(%)',
    'Pressure(in)',
    'Visibility(mi)',
    'Wind_Speed(mph)',
    'Precipitation(in)'
]

# defining wind values for value transformation (section 5.4.4)
wind_values = {
    'North': 'N',
    'South': 'S',
    'West': 'W',
    'East': 'E',
    'Calm': 'CALM',
    'Variable': 'VAR',
}

weather_values = {
    'Blowing': 'Blowing',
    'Windy': 'Windy',
    'Snow': 'Snow',
    'Drifting': 'Drifting',
    'Clear': 'Clear',
    'Cloudy': 'Cloudy',
    'Drizzle': 'Drizzle',
    'Fog': 'Fog',
    'Dust': 'Dust',
    'Whirls': 'Whirls',
    'Fair': 'Fair',
    'Freezing': 'Freezing',
    'Funnel Cloud': 'Funnel Cloud',
    'Rain': 'Rain',
    'Hail': 'Hail',
    'Haze': 'Haze',
    'Heavy': 'Heavy',
    'Light': 'Light',
    'Low': 'Light',  #
    'Mist': 'Mist',
    'Mostly Cloudy': 'Cloudy',
    'Overcast': 'Overcast',
    'Partial': 'Partial',
    'Partly': 'Partial',  #
    'Patches of Fog': 'Fog',  #
    'Sand': 'Sand',
    'Dusty': 'Dust',  #
    'Whirls Nearby': 'Whirlwinds',  #
    'Whirlwinds': 'Whirlwinds',
    'Ice': 'Ice Pellets',
    'Scattered Clouds': 'Cloudy',  #
    'Showers in the Vicinity': 'Rain',  #
    'Sleet': 'Sleet',
    'Small Hail': 'Hail',  #
    'Wintry Mix': 'Wintry Mix',
    'Thunder': 'Thunderstorm',  #
    'T-Storm': 'Thunderstorm',  #
    'Thunderstorm': 'Thunderstorm',
    'Thunderstorms': 'Thunderstorm',  #
    'Thunder in the Vicinity': 'Thunderstorm',  #
    'Tornado': 'Tornado',
    'Smoke': 'Smoke',
    'Shallow Fog': 'Fog',  #
    'Snow Grains': 'Snow Grains',
    'Squalls': 'Squalls',
    'Widespread Dust': 'Dust',  #
    'Volcanic Ash': 'Volcanic Ash',
    'None': 'None',
    'N/A Precipitation': 'None',  #
}

# defining day values binary encoding
day_dict = {
    'Day' : True,
    'Night': False
}

#%% md

---
## 3 Data Overview

#%%

# Shape of data
data_ori.shape

#%%

# Types
data_ori.dtypes


#%%

# Features
list(data_ori)  # What potential lies in the "Description"? (?@luke)

#%%

# Head
data_ori.head()

#%%

# Descriptions
data_ori.describe()

#%% md

---
## 5 Data Cleaning

Number, Temperature(F)	Wind_Chill(F)	Humidity(%)	Pressure(in)	Visibility(mi)	Wind_Speed(mph)	Precipitation(in)

#%%

# display all value counts -> length: counts unique values
for column in column_list:  # list of columns
    print(data_ori[column].value_counts().sort_index(), "\n")



#%% md

### 5.1 Drop columns
Dropping irrelevant columns.

Reasons:
Description - contains unstructured text data (with typos) which contains information such as address/ zipcode which
are already present in the data set. Other information in this column such as exact names, details of those involved
etc are unimportant for our current project.

Number, Precipitation - too many NaN values, others mostly 0. Weather data already included in another column.

Turning_Loop - all values are 'False'. Will not make any change to model.

Timezone - our analysis will be based on local time. Timezone does not have any effect on accidents.

Airport_Code - Location of accident already included in data set. Airport code unimportant.

Weather_Timestamp - shows us exact time of weather measurement which all match day of accident. Unimportant for now.

Wind_Chill(F) - We already have weather data. Wind chill is calculated using temperature and wind speed which we
already have in dataset. Affect of wind on skin is unimportant for accident rates.

End_Time - End time in this dataset is just Start_time + 6 hours. Doesn't have any significant meaning.

#%%

columns_to_drop = [
    'Description',
    'Number',
    'Precipitation(in)',
    'Turning_Loop',
    'Timezone',
    'Airport_Code',
    'Weather_Timestamp',
    'Wind_Chill(F)',
    'End_Time' #new addition** (irene)
]

data_ori.drop(columns=columns_to_drop, inplace=True)  # inplace -> no need to store result in new variable

#%% md

### 5.2 Drop missing values

checking for nan values in each column
todo : remaining cols

#%%

new_col_list = [] #39 cols
for col in column_list:
    if col not in columns_to_drop:
        new_col_list.append(col)

# 13 cols contain nan values
for col in new_col_list:
    nan_sum = data_ori[col].isnull().sum()
    if nan_sum:
        print(col, nan_sum)

#%%

# to confirm

# deleting 83 total rows - City, Sunrise_Sunset, Civil_Twilight, Nautical_Twilight, Astronomical_Twilight
print(len(data_ori)) #1516064
data_ori.dropna(subset = ["City", 'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight'], inplace=True)
print(len(data_ori)) #1515981

# to do : what about the rest (replace?)

#%% md

### 5.3 Drop incorrect values

Some values recorded in this dataset are clearly incorrect. We choose to not include them in future calculations.

These include:
1. Temperature(F) - contains extreme values of temperature outside the range of recorded temperature values in the US
from 2016 - 2020
2. Wind_Speed(mph) - contains extreme values of wind speed outside the range of recorded speed values in the US
from 2016 - 2020. Assuming vehicles involved in the accident were not literally inside a tornado/hurricane

#%%

# Extreme Temperature -> 5 rows dropped
data_ori.drop(data_ori[(data_ori['Temperature(F)'] >= 168.8) | (data_ori['Temperature(F)'] <= -77.8)].index,
              inplace=True)

# Extreme Wind_Speed -> 6 rows dropped
data_ori.drop(data_ori[data_ori['Wind_Speed(mph)'] >= 471.8].index, inplace=True)

#%% md

### 5.4 Value Transformation

#### 5.4.1 Zip Code

Formatting all zipcodes in dataset to contain 5 digits only - basic US zipcode format. The extended ZIP+4 code present
in a few of the rows is not necessary for our analysis.

#%%

# taking first 5 digits of zip code -> save it in Zipcode again
data_ori['Zipcode'] = data_ori['Zipcode'].str[:5]


#%% md

#### 5.4.2 Unit conversion to SI units

TODO: (Luke) fil End_lat and End_Lng by Start_Lat and Start_Lng (check prior what is the average difference between the two)
the point is: Even if we know, that the distance is 5km, the end of the accident/traffic jam can be in all cardinal
directions. Thus setting the end to the same coordinates is the best proxy.

#%%

data_ori["End_Lat"].fillna(data_ori["Start_Lat"])
data_ori["End_Lng"].fillna(data_ori["Start_Lng"])

#%%

# new columns for each SI unit created

# Distance miles -> kilometres
data_ori['Distance(km)'] = data_ori['Distance(mi)'] * 1.609

# Temperature F -> C
data_ori['Temperature(C)'] = (data_ori['Temperature(F)'] - 32) / 1.8

# Wind_Speed mi/h -> km/h
data_ori['Wind_Speed(kmh)'] = data_ori['Wind_Speed(mph)'] * 1.609

# Visibility mi -> km
data_ori['Visibility(km)'] = data_ori['Visibility(mi)'] * 1.609

# Pressure Pa -> in
data_ori['Pressure(Pa)'] = data_ori['Pressure(in)'] / 29.92

# dropping previous columns with american units
columns_to_drop = [
    'Distance(mi)',
    'Temperature(F)',
    'Wind_Speed(mph)',
    'Visibility(mi)',
    'Pressure(in)'
]

data_ori.drop(columns=columns_to_drop, inplace=True)

#%% md

#### 5.4.3 Timestamp transformation

#%%

#converting from string type to datetime
data_ori['Start_Time'] = pd.to_datetime(data_ori['Start_Time'])

# creating columns for Section 6 analysis.
data_ori['Year'] = data_ori['Start_Time'].dt.year
data_ori['Weekday'] = data_ori['Start_Time'].dt.dayofweek  # Monday = 0
data_ori['Month'] = data_ori['Start_Time'].dt.month
data_ori['Hour'] = data_ori['Start_Time'].dt.hour

#%% md

#### 5.4.4 Wind direction transformation
Converting overlapping values. For example: 'S' & 'South' mean the same thing so 'South' will be transformed to 'S'.
Transformations based on wind_values dict.

#%%

data_ori["Wind_Direction"].replace(wind_values, inplace=True)

#%% md

### 5.5 Set Data Types

#%%

data_prep = data_ori.copy(deep=True)

#%% md

---
## 6 Exploratory Data Analysis
### 6.1 Univariate Non-Graphical

describe data again

idea: What learnings can be drawn for your own driving behaviour? When is the most dangerous time? What is the most
what is the most dangerous weather? Which is the most dangerous state? How strong is the correlation between accidents
and the population density? What are the safest types of crossings? Where is the most dangerous place in the US?
difference weekday/weekend
differentiate between corona and pre-corona times; include apple mobility data https://covid19.apple.com/mobility

#%%

# display all value counts
for column in data_prep:  # list of columns
    print(data_prep[column].value_counts().sort_index(), "\n")

#%%

# display data types
data_prep.dtypes

#%%

# describe numerical columns
data_prep.describe()

#%%



#%%



#%%



#%%



#%%



#%% md

### 6.2 Univariate Graphical

#%%

# histogram of accidents of the biggest cities
data_prep.City.value_counts()[:20].plot(kind='bar', figsize=(12,6))

#%%

# histogram of accidents according to the weather condition (how to standardize?)
# preferably after feature engineering
data_prep.Weather_Condition.value_counts().plot(kind='bar', figsize=(12,6))

#%%

# ggplot with the develpment of accidents over time
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(12,6))
data_prep.groupby(['Month','Severity']).count()['ID'].unstack().plot(ax=ax)  # 'Year'
ax.set_xlabel('Month')
ax.set_ylabel('Number of Accidents')

#%%

# histogram of accidents according to time of day
hours = [hour for hour, df in data_prep.groupby('Hour')]
plt.plot(hours, data_prep.groupby(['Hour'])['ID'].count())
plt.xticks(hours)
plt.xlabel('Hour')
plt.ylabel('Number of accidents')
plt.grid(True)
plt.show()

#%%

# histogram of accidents according to day of the week
# TODO: Change numbers to actual days of the week (Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday)
days = [day for day, df in data_prep.groupby('Weekday')]
plt.plot(days, data_prep.groupby(['Weekday'])['ID'].count())
plt.xticks(days)
plt.xlabel('Weekday')
plt.ylabel('Number of accidents')
plt.grid(True)
plt.show()

#%%

# histogram of accidents filtered by state
data_prep.State.value_counts().plot(kind='bar', figsize=(12,6))

#%%

# pie diagram on severity
data_prep.Severity.value_counts().plot.pie()
# pie diagram on severity if the weather is poor (wind > threashold, rain > threshold);
# already dropped


#%% md

### 6.3 Multivariate Non-Graphical

#%%

# correlation matrices and PCA??
data_prep.corr()

#%% md

### 6.4 Multivariate Graphical

#%%

# to be adjusted:
fig=plt.gcf()
fig.set_size_inches(20,20)
fig=sns.heatmap(data_prep.corr(),annot=True,linewidths=1,linecolor='k',square=True,mask=False, vmin=-1, vmax=1,cbar_kws={"orientation": "vertical"},cbar=True)
sns.set(style='ticks')
plt.title("Spearman Correlation between all features")
sns.pairplot(data_prep)

#%%

# to be adjusted:
# US map simple: scatterplot based on latitude and longitude data, with correct alpha, to show densitiy
plt.figure(figsize = (15,9))
sns.scatterplot(x="Start_Lng", y="Start_Lat", data=data_prep, hue = "Severity", legend = "auto", s=15)
plt.title("Location of all accidents in the USA in the time from 2016 to 2020, distinguished by severtiy")
plt.show()

#%%

# ideas: A MAP of the US, showing the accident intensity for each place by colour
# https://runestone.academy/runestone/books/published/httlads/WorldFacts/cs1_graphing_infant_mortality.html


#%% md

---
## 7 Feature Engineering
### 7.1 Type Conversion
#### 7.1.1 Label Encoding

TODO: Jasmin @Irene -> is label encoding useful? Is assumes an order in the values, which is not given for county, state, cities.
    Wouldn't it be of higher relevance to OneHotEncode those as well?

(Update)
- attempt freq encoding for Counties, Cities
- one hot encoding for States
- binary encoding for Amenity, Bump, Crossing, Give_Way.......Astronomical twilight --done

duration
TMC: NA is an important information



#%%

data_encoding = data_prep.copy(deep=True)


#%% md

#### 7.1.2 'Binary' Encoding
Ordinal encoding for columns with Day/Night values to bool - Sunrise_Sunset, Civil_Twilight, Nautical_Twilight, Astronomical_Twilight

#%%


data_encoding['Sunrise_Sunset_isDay'] = data_encoding.Sunrise_Sunset.map(day_dict)
data_encoding['Civil_Twilight_isDay'] = data_encoding.Civil_Twilight.map(day_dict)
data_encoding['Nautical_Twilight_isDay'] = data_encoding.Nautical_Twilight.map(day_dict)
data_encoding['Astronomical_Twilight_isDay'] = data_encoding.Astronomical_Twilight.map(day_dict)

# drop previous columns without bool values
columns_to_drop = [
    'Sunrise_Sunset',
    'Civil_Twilight',
    'Nautical_Twilight',
    'Astronomical_Twilight'
]

data_encoding.drop(columns=columns_to_drop, inplace=True)
data_ori.head(10)
#%%
# to delete
data_encoding.head()
#%% md

#### 7.1.3 OneHot Encoding
##### TODO : Irene @Jasmin - just wanna double check if I did this right
For states

#%%
ohc = OneHotEncoder()
one_hot_encoded = ohc.fit_transform(data_encoding.State.values.reshape(-1,1)).toarray()

# generate array with correct column names
categories = ohc.categories_
column_names = []

for category in categories[0]:
    column_name = category
    column_names.append(column_name)

# set correct column names
one_hot_data = pd.DataFrame(one_hot_encoded, columns=column_names)

# delete one column to avoid the dummy variable trap
one_hot_data.drop(['AL'], axis= 1, inplace=True)

# combining ohc dataframe to previous df
data_encoding = pd.concat([data_encoding, one_hot_data], axis = 1)
data_encoding.head()
#%% md

#### 7.1.4 Manual Encoding

#%%

data_encoding['Weather_Condition'] = data_encoding['Weather_Condition'].fillna('None')

one_hot_encoder = OneHotEncoder()

data_one_hot = one_hot_encoder.fit_transform(data_encoding[['Weather_Condition']])
data_one_hot_array = data_one_hot.toarray()

# generate array with correct column names
categories = one_hot_encoder.categories_
column_names = []

for category in categories[0]:
    column_name = category
    column_names.append(column_name)

    # set correct column names
data_one_hot = pd.DataFrame(data_one_hot_array, columns=column_names)

# delete one column to avoid the dummy variable trap
data_one_hot.drop('None', axis=1, inplace=True) # drop last n rows

#%%

split_words= ['/', 'and', 'with', ' ']

def replace(value, split_value, split_index):
    if split_value in weather_values:
        column_name = 'weather_' + weather_values[split_value]
        if not column_name in data_encoding:
            data_encoding[column_name] = data_one_hot[value]
        else:
            data_encoding[column_name] += data_one_hot[value]

    else:
        try:
            if split_index < len(split_words):
                split_values = split_value.split(split_words[split_index])
                for split_value in split_values:
                    split_value=split_value.strip()
                    replace(value, split_value, split_index+1)
            else:
                print(split_value)
        except AttributeError or TypeError:
            print(str(value) + "!")

for column in data_one_hot:
    replace(column, column, 0)


#%% md
### 7.2 Timestamp transformation (Unix)
Converting Start_Time to seconds from Unix Epoch.

TODO Irene @Jasmin - why do we need to convert? We dont want our model to put a larger weightage on bigger time values
right? Also - unix epoch is 1.1.1970 UTC time. So we will need to think about the timezone for each state before
converting to seconds ༼☯﹏☯༽

But imo 4pm in Timezone1 should be considered the same as 4pm in another timezone. So I disregard timezones n give the 'wrong'
unix epoch time? If thats the case, I have implemented it below.
if I want to see what effect time of day (for eg) has on severity/num of accidents - is conversion to UTC epoch necessary

#%%
d = data_encoding['Start_Time']
# converting to unix epoch time and adding to df
data_encoding['N_Start_Time'] = d.view('int64')

# dropping original Start_Time column
data_encoding.drop('Start_Time', axis=1)
#%% md

### 7.3 Transformation

#%%

data_final = data_encoding.copy(deep=True)

# divide columns into dependant and independant
data_independant = data_final.drop(['Severity'], axis=1)
data_dependant = data_final[['Severity']]

# normalize the data between 0 and 1
data_independent = (data_independant - data_independant.min()) / (data_independant.max() - data_independant.min())



#%% md

---
## 8 Model
### 8.1 Partitioning the Data

#%%



#%% md

### 8.2 Sampling

#%%



#%% md

### 8.3 ""Model""

#%%

# How much does the inclusion of apples mobility value increase the accurancy of our prediction model?
# LSTM-GBRT https://downloads.hindawi.com/journals/jcse/2020/4206919.pdf
# hybrid K-means and random forest https://link.springer.com/content/pdf/10.1007/s42452-020-3125-1.pdf
# OCT https://towardsdatascience.com/using-machine-learning-to-predict-car-accidents-44664c79c942
# Regression-kriging https://carto.com/blog/predicting-traffic-accident-hotspots-with-spatial-data-science/


#%% md

### 8.4 Testing

#%% md

### 8.5 Prediction driving factors

# SHAP diagram






