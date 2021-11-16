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

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler

#%% md

### 1.2 Data

#%%

file_path = './US_Accidents_Dec20.csv'

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

# all parameters in original data

column_list = [
    'ID',
    'Source',
    'TMC',
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
    'Day': True,
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

#%% md

### 5.1 Drop columns
Dropping irrelevant columns.

Reasons:
End_Lat, End_Lng - Shows end position of car crash. Full of NaNs.

Country - Since it is all happening in the US, this is an insignificant column.

ID - ID of each crash. Unnecessary for modelling reasons.

Source - API source of where the data comes from. This has no relationship to accident type/severity.

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

Sunrise_Sunset, Civil_Twilight, Astronomical_Twilight - to avoid spurious correlatons. Nautical Twilight is the point at
which artificial light is recommended so we chose that as our indicator of Day/Night
#%%

columns_to_drop = [
    'End_Lat',
    'End_Lng',
    'Country',
    'ID',
    'Source',
    'Description',
    'Number',
    'Precipitation(in)',
    'Turning_Loop',
    'Timezone',
    'Airport_Code',
    'Weather_Timestamp',
    'Wind_Chill(F)',
    'End_Time',
    'Sunrise_Sunset',
    'Civil_Twilight',
    'Astronomical_Twilight'

]

data_ori.drop(columns=columns_to_drop, inplace=True)  # inplace -> no need to store result in new variable

#%% md

### 5.2 Drop missing values

checking for nan values in each column

#%%
# assuming, that no TMC value means it is not a severe accident (not severe enough to be mentioned)
data_ori['TMC'].replace(np.NAN, 0, inplace=True)
data_ori['TMC'].value_counts()
#%%

new_col_list = []  #39 cols
for col in column_list:
    if col not in columns_to_drop:
        new_col_list.append(col)

# 13 cols contain nan values
# City 137
# Zipcode 1292
# Temperature(F) 89900
# Humidity(%) 95467
# Pressure(in) 76384
# Visibility(mi) 98668
# Wind_Direction 83611
# Wind_Speed(mph) 479326
# Weather_Condition 98383
# Nautical_Twilight 141


for col in new_col_list:
    nan_sum = data_ori[col].isnull().sum()
    if nan_sum:
        print(col, nan_sum)

#%%
# deleting nan rows

# deleting 141 total rows - City, Nautical_Twilight
print(len(data_ori))  # 4232541
data_ori.dropna(subset=["City", 'Nautical_Twilight'], inplace=True)

# deleting remaining rows since it is only a small percentage of the entire dataset
data_ori.dropna(
    subset=['City', 'Zipcode', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Direction',
            'Wind_Speed(mph)', 'Weather_Condition'], inplace=True)
print(len(data_ori))  # 3713887

# about 12% data removed (all NaNs)
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
print(len(data_ori))
# Extreme Wind_Speed -> 13 rows dropped
data_ori.drop(data_ori[data_ori['Wind_Speed(mph)'] >= 471.8].index, inplace=True)
print(len(data_ori))
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
data_ori['Month'] = data_ori['Start_Time'].dt.month
data_ori['Week'] = data_ori['Start_Time'].dt.week
data_ori['Weekday'] = data_ori['Start_Time'].dt.dayofweek  # Monday = 0
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



#%% md

### 6.2 Univariate Graphical

#%%

# histogram of accidents of the biggest cities
data_prep.City.value_counts()[:20].plot(kind='bar', figsize=(12,6), color="#173F74")
plt.xticks(rotation=30)
plt.ylabel('Number of accidents')
plt.title("The 20 US-Cities with most accidents.")
# stacked histogram by year/severity

#%%
# Prepare data
x_var = 'Month'
groupby_var = 'Severity'
df_agg = data_prep.loc[:, [x_var, groupby_var]].groupby(groupby_var)
vals = [data_prep[x_var].values.tolist() for i, data_prep in df_agg]

# Draw
plt.figure(figsize=(16, 9), dpi=80)
colors = [plt.cm.Spectral(i / float(len(vals))) for i in range(len(vals))]
n, bins, patches = plt.hist(vals, data_prep[x_var].unique().__len__(), stacked=True, density=False,
                            color=colors[:len(vals)])

# Decoration
plt.legend({group: col for group, col in zip(np.unique(data_prep[groupby_var]).tolist(), colors[:len(vals)])})
plt.title(f"Stacked Histogram of ${x_var}$ colored by ${groupby_var}$", fontsize=22)
plt.xlabel(x_var)
plt.ylabel("Number of Accidents")
# plt.ylim(0, 40)
month_list = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
              'November', 'December', None]
plt.xticks(bins, month_list, rotation=90, horizontalalignment='left')
plt.show()

#%%

# Prepare data
x_var = 'State'
groupby_var = 'Severity'
df_agg = data_prep.loc[:, [x_var, groupby_var]].groupby(groupby_var)
vals = [data_prep[x_var].values.tolist() for i, data_prep in df_agg]

# Draw
plt.figure(figsize=(16, 9), dpi=80)
colors = [plt.cm.Spectral(i / float(len(vals) - 1)) for i in range(len(vals))]
n, bins, patches = plt.hist(vals, data_prep[x_var].unique().__len__(), stacked=True, density=False,
                            color=colors[:len(vals)])

# Decoration
plt.legend({group: col for group, col in zip(np.unique(data_prep[groupby_var]).tolist(), colors[:len(vals)])})
plt.title(f"Stacked Histogram of ${x_var}$ colored by ${groupby_var}$", fontsize=22)
plt.xlabel(x_var)
plt.ylabel("Number of Accidents")
plt.xticks(bins, rotation=90, horizontalalignment='left')
plt.show()

#%%

# Prepare data
x_var = 'Hour'
groupby_var = 'Severity'
df_agg = data_prep.loc[:, [x_var, groupby_var]].groupby(groupby_var)
vals = [data_prep[x_var].values.tolist() for i, data_prep in df_agg]

# Draw
plt.figure(figsize=(16, 9), dpi=80)
colors = [plt.cm.Spectral(i / float(len(vals))) for i in range(len(vals))]
n, bins, patches = plt.hist(vals, data_prep[x_var].unique().__len__(), stacked=True, density=False,
                            color=colors[:len(vals)])
print(data_prep[x_var].unique().__len__())
print(len(bins))
print(len(patches))
print(len(vals))

# Decoration
plt.legend({group: col for group, col in zip(np.unique(data_prep[groupby_var]).tolist(), colors[:len(vals)])})
plt.title(f"Stacked Histogram of ${x_var}$ colored by ${groupby_var}$", fontsize=22)
plt.xlabel(x_var)
plt.ylabel("Number of Accidents")
# plt.ylim(0, 40)
month_list = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
              'November', 'December', None]
plt.xticks(bins, rotation=90, horizontalalignment='left')
plt.show()

#%%

# histogram of accidents according to the weather condition (how to standardize?)
# preferably after feature engineering
data_prep.Weather_Condition.value_counts()[:15].plot(kind='bar', figsize=(12, 6), color="#173F74")
plt.xticks(rotation=30)
plt.ylabel('Number of accidents', color="#173F74")
plt.title("The 15 most common weather conditions.", color="#173F74")
plt.show()

#%%

# ggplot with the development of accidents over time
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(18, 18))
data_prep.groupby(['Year', 'Week', 'Severity']).count()['ID'].unstack().plot(ax=ax, cmap="cividis")  #
ax.set_xlabel('Week', color="#173F74")
ax.set_ylabel('Number of Accidents', color="#173F74")
ax.set_title("Development of accidents over time distinguished by the severity")
plt.show()

#%%

# histogram of accidents according to time of day
hours = [hour for hour, df in data_prep.groupby('Hour')]
plt.plot(hours, data_prep.groupby(['Hour'])['ID'].count(), color="#173F74")
plt.xticks(hours)
plt.xlabel('Hour', color="#173F74")
plt.ylabel('Number of accidents', color="#173F74")
plt.title("Histogram of accidents according to the time of day")
plt.show()

#%%

# histogram of accidents according to day of the week
days = [day for day, df in data_prep.groupby('Weekday')]
plt.plot(days, data_prep.groupby(['Weekday'])['ID'].count(), color="#173F74")
plt.xticks(days, ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], rotation=30)
plt.xlabel('Weekday')
plt.ylabel('Number of accidents')
plt.show()

#%%

# histogram of accidents filtered by state
data_prep.State.value_counts().plot(kind='bar', figsize=(12, 6), color="#173F74")
plt.xticks(rotation=30)
plt.ylabel('Number of accidents')
plt.title("The US States ordered by the number of accidents")

#%%

# pie diagram on severity
data_prep.Severity.value_counts().plot.pie(cmap="cividis")
plt.title("Share of the different severity levels")
plt.legend()


#%% md

### 6.3 Multivariate Non-Graphical

#%%

# correlation matrices and PCA??
data_prep.corr()

#%% md

### 6.4 Multivariate Graphical

#%%

# to be adjusted:
fig = plt.gcf()
fig.set_size_inches(16, 12)
fig = sns.heatmap(data_prep.corr(), annot=True, linewidths=1, linecolor='k', square=True, mask=False,
                  vmin=-1, vmax=1, cbar_kws={"orientation": "vertical"}, cbar=True, cmap="cividis")
sns.set(style='ticks')
plt.title("Correlogram of all features")

#%%

# to be adjusted:
# US map simple: scatterplot based on latitude and longitude data, with correct alpha, to show densitiy
plt.figure(figsize=(15, 9))
sns.scatterplot(x="Start_Lng", y="Start_Lat", data=data_prep, hue="Severity", legend="auto",
                s=15, palette="cividis", alpha=0.3)
plt.title("Location of all accidents in the USA in the time from 2016 to 2020, distinguished by severtiy")
plt.show()

#%%

state_list = data_prep["State"].unique()
sorted_state_list = sorted(state_list)
plt.figure(figsize=(16, 9))
g = sns.scatterplot(x="Start_Lng", y="Start_Lat", data=data_prep, hue="State", hue_order=sorted_state_list,
                    legend="auto", s=15, palette="cividis", alpha=0.3)
g.legend(loc='center right', bbox_to_anchor=(1.1, 0.5), ncol=2)
plt.title("Location of all accidents in the USA in the time from 2016 to 2020, distinguished by State")
plt.show()
#%%

# ideas: A MAP of the US, showing the accident intensity for each place by colour
# https://runestone.academy/runestone/books/published/httlads/WorldFacts/cs1_graphing_infant_mortality.html
#%%
#@luke !!! TO DO
#graph of number of accidents per state to show backlog

#%% md

### 6.5 Comparison of 2019 with 2020
- Dumbbell plot https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python

#### 6.5.1 Preparation
#%%
data_prep_wo_bias = data_prep.copy(deep=True)

#dropping states which cause backlog
#@luke - check if in fact these are the right states !!!!!
data_prep_wo_bias = data_prep_wo_bias[data_prep_wo_bias['State'] != 'CA']
data_prep_wo_bias = data_prep_wo_bias[data_prep_wo_bias['State'] != 'FL']

#%%
data_prep_wo_bias_2020 = data_prep_wo_bias[data_prep_wo_bias.Year == 2020]
data_prep_wo_bias_2019 = data_prep_wo_bias[data_prep_wo_bias.Year == 2019]

#%% md

#### 6.5.2 Comparison
- Dumbbell plot https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python

#%%



#%% md

---
## 7 Feature Engineering

#%%

# Preparation
data_encoding = data_prep_wo_bias.copy(deep=True)
data_encoding.reset_index(inplace=True, drop=True)
data_encoding.head()


#%% md

### 7.1 Type Conversion
#### 7.1.1 Ordinal Encoding

(Update)
- attempt freq encoding for Counties
    - attempt ordianl encoding for Streets, Cities
        - one hot encoding for States

duration

#%%
# ordinal encoding instead of frequency encoding -> time reasons
# for report: talk about adv & disadv
ordinal_encoder = OrdinalEncoder()

data_encoding[['Street']] = ordinal_encoder.fit_transform(data_encoding[['Street']])
print(ordinal_encoder.categories_)

data_encoding[['City']] = ordinal_encoder.fit_transform(data_encoding[['City']])
print(ordinal_encoder.categories_)

#%% md

#### 7.1.2 'Binary' Encoding
Ordinal encoding for columns with Day/Night values to bool - Nautical_Twilight
Ordinal encoding for Side (Left/Right) to bool

#%%
# L - 0, R - 1
data_encoding[['Side']] = ordinal_encoder.fit_transform(data_encoding[['Side']])
print(ordinal_encoder.categories_)

#%%
#Day - 0, Night 1
data_encoding[['Nautical_Twilight']] = ordinal_encoder.fit_transform(data_encoding[['Nautical_Twilight']])
print(ordinal_encoder.categories_)

#%%

bool_columns = [
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
]

for column in bool_columns:
    data_encoding[[column]] = ordinal_encoder.fit_transform(data_encoding[[column]])
    print(ordinal_encoder.categories_)

#%% md

#### 7.1.3 OneHot Encoding

For states

#%%

ohc = OneHotEncoder()
one_hot_encoded = ohc.fit_transform(data_encoding.State.values.reshape(-1, 1)).toarray()
#%%
# generate array with correct column names
categories = ohc.categories_
column_names = []

for category in categories[0]:
    column_name = category
    column_names.append(column_name)

# set correct column names
one_hot_data = pd.DataFrame(one_hot_encoded, columns=column_names)
#%%
# delete one column to avoid the dummy variable trap
one_hot_data.drop(one_hot_data.columns[-1], axis=1, inplace=True)

# combining ohc dataframe to previous df
data_encoding = pd.concat([data_encoding, one_hot_data], axis=1)

data_encoding.drop('State', axis=1, inplace=True)
data_encoding.head()

#%% md
For Wind Direction

#%%
one_hot_encoded = ohc.fit_transform(data_encoding.Wind_Direction.values.reshape(-1, 1)).toarray()

# generate array with correct column names
categories = ohc.categories_
column_names = []

for category in categories[0]:
    column_name = category
    column_names.append(column_name)

# set correct column names
one_hot_data = pd.DataFrame(one_hot_encoded, columns=column_names)
one_hot_data.head()

# delete one column to avoid the dummy variable trap
one_hot_data.drop(one_hot_data.columns[-1], axis=1, inplace=True)

# combining ohc dataframe to previous df
data_encoding = pd.concat([data_encoding, one_hot_data], axis=1)

data_encoding.drop('Wind_Direction', axis=1, inplace=True)
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
data_one_hot.drop(data_one_hot.columns[-1], axis=1, inplace=True)  # drop last n rows
data_one_hot.head()
#%%

split_words = ['/', 'and', 'with', ' ']


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
                    split_value = split_value.strip()
                    replace(value, split_value, split_index + 1)
            else:
                print(split_value)
        except AttributeError or TypeError:
            print(str(value) + "!")


for column in data_one_hot:
    replace(column, column, 0)


data_encoding.drop('Weather_Condition', axis=1, inplace=True)
#%% md

#### 7.1.5 Frequency encoding
For Street, City, County

#%%
county_dict = data_encoding['County'].value_counts().to_dict()

#%%
# Ordinal Freq Encoding
county_array = county_dict.keys() #keys from the dict are now arranged in descending order in the array. Most frequent -> least

county_encoder = OrdinalEncoder(categories=county_array)
data_encoding[['County']] = ordinal_encoder.fit_transform(data_encoding[['County']])

#%%
# Real Freq Encoding -> takes a lot of time because of looping?
# data_encoding['County'] = data_encoding['County'].replace(county_dict)

#%% md

### 7.2 Timestamp transformation (Unix)
Converting Start_Time to seconds from Unix Epoch.

#%%

d = data_encoding['Start_Time']
# converting to unix epoch time and adding to df
data_encoding['N_Start_Time'] = d.view('int64')

# dropping original Start_Time column
data_encoding.drop('Start_Time', axis=1, inplace=True)

#%% md

### 7.3 Transformation

#%%

data_final = data_encoding.copy(deep=True)

# divide columns into dependant and independent
data_independent = data_final.drop(['Severity'], axis=1)
data_dependant = data_final[['Severity']]

data_independent.head()
#%%
# to scale each column so that every feature/parameter has equal weight

# x = data_independent.values # returns a numpy array with all the values of the dataframe
min_max_scaler = MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# data_independent = pd.DataFrame(x_scaled)

# TODO:  restore column names. Irene: proposed change. But it might be taking longer than before
data_independent = pd.DataFrame(min_max_scaler.fit_transform(data_independent), index=data_independent.index, columns=data_independent.columns)
data_independent.head()
#%% md

---
## 8 Model
### 8.1 Partitioning the Data

#%%

# assign data
X = data_independent
Y = data_dependant

# generate test and trainings set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

#%% md

### 8.2 Sampling

Training data
#%%

# concatenate data for sampling
data_tbsampled = X_train.copy(deep=True)
data_tbsampled['severity'] = Y_train

data_tbsampled.reset_index(inplace=True, drop=True)
#%%

severity_values = Y_train['Severity'].value_counts()

#%%
# ensuring every severity level has equal proportions in the data
def balanced_subsample(y, size=None, random_state=None): # returns a List with randomly chosen row numbers
    subsample = []
    if size is None:
        n_smp = y.value_counts().min()
    else:
        n_smp = int(size / len(y.value_counts().index))

    if not random_state is None:
        np.random.seed(random_state)

    for label in y.value_counts().index:
        samples = y[y == label].index.values
        index_range = range(samples.shape[0])
        indexes = np.random.choice(index_range, size=n_smp, replace=False)
        subsample += samples[indexes].tolist()

    return subsample

rows = balanced_subsample(data_tbsampled['severity'], size = 4000, random_state=0)
#%%
data_downsampled = data_tbsampled.iloc[rows, :]

X_train = data_downsampled.drop('severity', axis=1)
Y_train = data_downsampled['severity']

#%% md
Test Data

#%%
#making sure train Y has the same number of rows

# concatenate data for sampling
data_test_tbsampled = X_test.copy(deep=True)
data_test_tbsampled['severity'] = Y_test

data_test_tbsampled.reset_index(inplace=True, drop=True)

#%%
from sklearn.utils import resample
data_test_sampled = resample(data_test_tbsampled, replace=False, n_samples=4000, random_state=0)

#%%

X_test = data_test_sampled.drop('severity', axis=1)
Y_test = data_test_sampled['severity']

#%% md

### 8.3 Fitting
#%% md

How much does the inclusion of apples mobility value increase the accurancy of our prediction model?
LSTM-GBRT https://downloads.hindawi.com/journals/jcse/2020/4206919.pdf
hybrid K-means and random forest https://link.springer.com/content/pdf/10.1007/s42452-020-3125-1.pdf
OCT https://towardsdatascience.com/using-machine-learning-to-predict-car-accidents-44664c79c942
Regression-kriging https://carto.com/blog/predicting-traffic-accident-hotspots-with-spatial-data-science/
#%%
report = pd.DataFrame(columns=['Model','Mean Acc. Training','Standard Deviation', 'Acc. Train', 'Acc. Test'])


#%% md

#### 8.3.1 KNN
#%%
from sklearn.neighbors import KNeighborsClassifier
knnmodel = KNeighborsClassifier()

from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_neighbors': [ 3, 4, 5]
}

CV_knnmodel = GridSearchCV(estimator=knnmodel, param_grid=param_grid, cv=10)
CV_knnmodel.fit(X_train, Y_train)
print(CV_knnmodel.best_params_)

# use the best parameters
knnmodel = knnmodel.set_params(**CV_knnmodel.best_params_)
knnmodel.fit(X_train, Y_train)

#%%
# predict training and test data
Y_train_pred = knnmodel.predict(X_train)
acctrain = accuracy_score(Y_train, Y_train_pred)

Y_test_pred = knnmodel.predict(X_test)
acctest = accuracy_score(Y_test, Y_test_pred)

#%%
# fill report
report.loc[len(report)] = ['k-NN (grid)',
                           CV_knnmodel.cv_results_['mean_test_score'][CV_knnmodel.best_index_],
                           CV_knnmodel.cv_results_['std_test_score'][CV_knnmodel.best_index_],
                           acctrain,
                           acctest]
print(report.loc[len(report)-1])

#%%
# visualize confusion matrix
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)

#%%
plot_confusion_matrix(knnmodel, X_test, Y_test, labels=[1, 2, 3, 4],
                      cmap=plt.cm.Blues, values_format='d')

#%% md

#### 8.3.2 Decision Trees
#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
dtree_model = RandomForestClassifier()

from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [50,100,150],
    'max_depth': [5,6,7,8] #why not more? it is suggested that the best number of splits lie between 5-8
}

CV_dtree_model = GridSearchCV(estimator=dtree_model, param_grid=param_grid, cv=10)
CV_dtree_model.fit(X_train, Y_train)
print(CV_dtree_model.best_params_)

# use the best parameters
dtree_model = dtree_model.set_params(**CV_dtree_model.best_params_)
dtree_model.fit(X_train, Y_train)
#%%
# predict training and test data
Y_train_pred = dtree_model.predict(X_train)
acctrain = accuracy_score(Y_train, Y_train_pred)

Y_test_pred = dtree_model.predict(X_test)
acctest = accuracy_score(Y_test, Y_test_pred)

#%%
# fill report
report.loc[len(report)] = ['Random Forest Classifier (grid)',
                           CV_dtree_model.cv_results_['mean_test_score'][CV_dtree_model.best_index_],
                           CV_dtree_model.cv_results_['std_test_score'][CV_dtree_model.best_index_],
                           acctrain,
                           acctest]
print(report.loc[len(report)-1])

#%%
# visualize confusion matrix
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)

#%%
plot_confusion_matrix(dtree_model, X_test, Y_test, labels=[1, 2, 3, 4], cmap=plt.cm.Blues, values_format='d')
#%% md

#### 8.3.3 Neural Networks
#%%
from sklearn.neural_network import MLPClassifier
nnetmodel = MLPClassifier(max_iter=400)

from sklearn.model_selection import GridSearchCV
param_grid = {
    'hidden_layer_sizes': [(3,), (5,), (9,)],
    'activation': ['logistic', 'tanh', 'relu']
}

CV_nnetmodel = GridSearchCV(estimator=nnetmodel, param_grid=param_grid, cv=10)
CV_nnetmodel.fit(X_train, Y_train)
print(CV_nnetmodel.best_params_)

# use the best parameters
nnetmodel = nnetmodel.set_params(**CV_nnetmodel.best_params_)
nnetmodel.fit(X_train, Y_train)

#%%
# predict training and test data
Y_train_pred = nnetmodel.predict(X_train)
acctrain = accuracy_score(Y_train, Y_train_pred)

Y_test_pred = nnetmodel.predict(X_test)
acctest = accuracy_score(Y_test, Y_test_pred)

#%%
# fill report
report.loc[len(report)] = ['Neural Networks (grid)',
                           CV_nnetmodel.cv_results_['mean_test_score'][CV_nnetmodel.best_index_],
                           CV_nnetmodel.cv_results_['std_test_score'][CV_nnetmodel.best_index_],
                           acctrain,
                           acctest]
print(report.loc[len(report)-1])

#%%
# visualize confusion matrix
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)

#%%
plot_confusion_matrix(nnetmodel, X_test, Y_test, labels=[1, 2, 3, 4],
                      cmap=plt.cm.Blues, values_format='d')

#%% md

### 8.5 Prediction driving factors

# SHAP diagram






