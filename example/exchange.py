#%% md

# Data Analysis Project

Traffic accidents in the US from Feb 2016 – Dec 2020 from https://smoosavi.org/datasets/us_accidents

Motivation:
1. Are there changes in accidents between the first half of 2019 and of 2020? Are the number of accidents affected by Covid-19?
2. What factors affect the severity of an accident?

---
## 1 Imports
### 1.1 Libraries

#%%

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample

import seaborn as sns
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

# All features in original data
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

# Defining all columns in original data with numeric values
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

# Defining all columns in original data with boolean values
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

# Defining wind values for value transformation (section 5.4.4)
wind_values = {
    'North': 'N',
    'South': 'S',
    'West': 'W',
    'East': 'E',
    'Calm': 'CALM',
    'Variable': 'VAR',
}

# Defining weather values for frequency encoding (section 7.1.4)
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
list(data_ori)

#%%

# Head
data_ori.head()

#%%

# Descriptions
data_ori.describe()

#%% md

---
## 5 Data Cleaning

### 5.1 Drop columns
Dropping irrelevant columns.

Reasons:
- End_Lat, End_Lng - Shows end position of car crash. Full of NaNs.

- Country - Since it is all happening in the US, this is an insignificant column.

- ID - ID of each crash. Unnecessary for modelling reasons.

- Source - API source of where the data comes from. This has no relationship to accident type/severity.

- Description - contains unstructured text data (with typos) which contains information such as address/ zipcode which
are already present in the data set. Other information in this column such as exact names, details of those involved
etc are unimportant for our current project.

- Number, Precipitation - too many NaN values, others mostly 0. Weather data already included in another column.

- Turning_Loop - all values are 'False'. Will not make any change to model.

- Timezone - our analysis will be based on local time. Timezone does not have any effect on accidents.

- Airport_Code - Location of accident already included in data set. Airport code unimportant.

- Weather_Timestamp - shows us exact time of weather measurement which all match day of accident. Unimportant for now.

- Wind_Chill(F) - We already have weather data. Wind chill is calculated using temperature and wind speed which we
already have in dataset. Affect of wind on skin is unimportant for accident rates.

- End_Time - End time in this dataset is just Start_time + 6 hours. Doesn't have any significant meaning.

- Sunrise_Sunset, Civil_Twilight, Astronomical_Twilight - to avoid spurious correlatons. Nautical Twilight is the point at
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

Checking for nan values in each column

#%%
# Assuming, that no TMC value means it is not a severe accident (not severe enough to be mentioned)
# Therefore, replacing the NaN values in TMC with 0
data_ori['TMC'].replace(np.NAN, 0, inplace=True)
data_ori['TMC'].value_counts()

#%%

# Checking for number of nan values in columns
new_col_list = []  # 39 cols
for col in column_list:
    if col not in columns_to_drop:
        new_col_list.append(col)

for col in new_col_list:
    nan_sum = data_ori[col].isnull().sum()
    if nan_sum:
        print(col, nan_sum)

#%% md
13 cols contain nan values
- City: 137
- Zipcode 1292
- Temperature(F) 89900
- Humidity(%) 95467
- Pressure(in) 76384
- Visibility(mi) 98668
- Wind_Direction 83611
- Wind_Speed(mph) 479326
- Weather_Condition 98383
- Nautical_Twilight 141

#%%
# Deleting nan rows

# Deleting 141 total rows - City, Nautical_Twilight
print(len(data_ori))  # 4232541

data_ori.dropna(subset=["City", 'Nautical_Twilight'], inplace=True)

# Deleting remaining rows since it is only a small percentage of the entire dataset
data_ori.dropna(
    subset=['City', 'Zipcode', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Direction',
            'Wind_Speed(mph)', 'Weather_Condition'], inplace=True)

print(len(data_ori))  # 3713887
#%% md
-> about 12% data removed (all NaNs)

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

# Taking first 5 digits of zip code -> save it in Zipcode again
data_ori['Zipcode'] = data_ori['Zipcode'].str[:5]

#%% md

#### 5.4.2 Unit conversion to SI units

Convert from US to SI units and create a new column for each
#%%

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

# Dropping previous columns with american units
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

# Converting from string type to datetime
data_ori['Start_Time'] = pd.to_datetime(data_ori['Start_Time'])

# Creating columns for Section 6 analysis.
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

---
## 6 Exploratory Data Analysis

#%%
# Copy original dataset and work with the new data_prep
data_prep = data_ori.copy(deep=True)

#%% md
### 6.1 Univariate Non-Graphical

#%%

# Display all value counts
for column in data_prep:  # list of columns
    print(data_prep[column].value_counts().sort_index(), "\n")

#%%

# Display data types
data_prep.dtypes

#%%

# Describe numerical columns
data_prep.describe()

#%% md

### 6.2 Univariate Graphical

Histogram of accidents of the biggest cities
#%%
data_prep.City.value_counts()[:20].plot(kind='bar', figsize=(12, 6), color="#173F74")
plt.xticks(rotation=45, fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Number of accidents', fontsize=15)
plt.title("The 20 US-Cities with most accidents.", fontsize=21)

#%% md
Histogram of number of accidents over month grouped by severity

#%%

# Prepare data
x_var = 'Month'
groupby_var = 'Severity'
df_agg = data_prep.loc[:, [x_var, groupby_var]].groupby(groupby_var)
vals = [data_prep[x_var].values.tolist() for i, data_prep in df_agg]

# Draw
plt.figure(figsize=(16, 9), dpi=80)
colors = [plt.cm.cividis(i / float(len(vals))) for i in range(len(vals))]
n, bins, patches = plt.hist(vals, data_prep[x_var].unique().__len__(), stacked=True, density=False,
                            color=colors[:len(vals)])

# Decoration
plt.legend({group: col for group, col in zip(np.unique(data_prep[groupby_var]).tolist(), colors[:len(vals)])},
           fontsize=18)
plt.title(f"Stacked Histogram of ${x_var}$ colored by ${groupby_var}$", fontsize=22)
plt.xlabel(x_var, fontsize=18)
plt.ylabel("Number of Accidents", fontsize=18)
# plt.ylim(0, 40)
month_list = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
              'November', 'December', None]
plt.xticks(bins, month_list, rotation=90, horizontalalignment='left', fontsize=18)
plt.yticks(fontsize=18)
plt.show()

#%% md
Histogram of number of accidents over states grouped by severity

#%%

# Prepare data
x_var = 'State'
groupby_var = 'Severity'
# data_prep.sort_values(by=[x_var])
df_agg = data_prep.loc[:, [x_var, groupby_var]].groupby(groupby_var)
vals = [data_prep[x_var].values.tolist() for i, data_prep in df_agg]

# Draw
plt.figure(figsize=(16, 9), dpi=80)
colors = [plt.cm.cividis(i / float(len(vals) - 1)) for i in range(len(vals))]
n, bins, patches = plt.hist(vals, data_prep[x_var].unique().__len__(), stacked=True, density=False,
                            color=colors[:len(vals)])

# Decoration
plt.legend({group: col for group, col in zip(np.unique(data_prep[groupby_var]).tolist(), colors[:len(vals)])},
           fontsize=18)
plt.title(f"Stacked Histogram of ${x_var}$ colored by ${groupby_var}$", fontsize=22)
plt.xlabel(x_var, fontsize=18)
plt.ylabel("Number of Accidents", fontsize=18)
plt.xticks(bins, rotation=90, horizontalalignment='left', fontsize=18)
plt.yticks(fontsize=18)
plt.show()

#%% md
Histogram of number of accidents over hours grouped by severity

#%%

# Prepare data
x_var = 'Hour'
groupby_var = 'Severity'
df_agg = data_prep.loc[:, [x_var, groupby_var]].groupby(groupby_var)
vals = [data_prep[x_var].values.tolist() for i, data_prep in df_agg]

# Draw
plt.figure(figsize=(16, 9), dpi=80)
colors = [plt.cm.cividis(i / float(len(vals))) for i in range(len(vals))]
n, bins, patches = plt.hist(vals, data_prep[x_var].unique().__len__(), stacked=True, density=False,
                            color=colors[:len(vals)])

# Decoration
plt.legend({group: col for group, col in zip(np.unique(data_prep[groupby_var]).tolist(), colors[:len(vals)])},
           fontsize=18)
plt.title(f"Stacked Histogram of ${x_var}$ colored by ${groupby_var}$", fontsize=22)
plt.xlabel(x_var, fontsize=18)
plt.ylabel("Number of Accidents", fontsize=18)
hour_list = ['0 am', '1 am', '2 am', '3 am', '4 am', '5 am', '6 am', '7 am', '8 am', '9 am', '10 am', '11 am',
             '12 pm', '1 pm', '2 pm', '3 pm', '4 pm', '5 pm', '6 pm', '7 pm', '8 pm', '9 pm', '10 pm', '11 pm',
             '11:59:59 pm']
plt.xticks(bins, hour_list, rotation=90, horizontalalignment='left', fontsize=18)
plt.yticks(fontsize=18)
plt.show()

#%% md
Histogram of number of accidents over weekday grouped by severity

#%%
# Prepare data
x_var = 'Weekday'
groupby_var = 'Severity'
df_agg = data_prep.loc[:, [x_var, groupby_var]].groupby(groupby_var)
vals = [data_prep[x_var].values.tolist() for i, data_prep in df_agg]

# Draw
plt.figure(figsize=(16, 9), dpi=80)
colors = [plt.cm.cividis(i / float(len(vals))) for i in range(len(vals))]
n, bins, patches = plt.hist(vals, data_prep[x_var].unique().__len__(), stacked=True, density=False,
                            color=colors[:len(vals)])

# Decoration
plt.legend({group: col for group, col in zip(np.unique(data_prep[groupby_var]).tolist(), colors[:len(vals)])},
           fontsize=18)
plt.title(f"Stacked Histogram of ${x_var}$ colored by ${groupby_var}$", fontsize=22)
plt.xlabel(x_var, fontsize=18)
plt.ylabel("Number of Accidents", fontsize=18)
weekday_list = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", None]
plt.xticks(bins, weekday_list, rotation=90, horizontalalignment='left', fontsize=18)
plt.yticks(fontsize=18)
plt.show()

#%% md
Histogram of accidents according to the weather condition

#%%
data_prep.Weather_Condition.value_counts()[:15].plot(kind='bar', figsize=(12, 6), color="#173F74")
plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Number of accidents', color="#173F74", fontsize=14)
plt.title("The 15 most common weather conditions.", color="#173F74", fontsize=21)
plt.show()

#%% md

ggplot with the development of accidents over time grouped by severity

#%%
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(16, 9))
data_prep.groupby(['Year', 'Week', 'Severity']).count()["City"].unstack().plot(ax=ax, cmap="cividis")
ax.set_xlabel('Time (Year, Week)', color="#173F74", fontsize=18)
ax.set_ylabel('Number of Accidents', color="#173F74", fontsize=18)

ax.set_title("Development of accidents per week distinguished by the severity", fontsize=22)
ax.legend(fontsize=14)

# labels = [item.get_text() for item in ax.get_xticklabels()]
# labels[1] = 'Testing'
# ax.set_xticklabels(labels)
# start, end = (2016, 1), (2020, 52)
# ax.xaxis.set_ticks(np.arange(start, end, 26))
# ax.set_xticklabels(["2016 H1", "2016 H2","2017 H1", "2017 H2","2018 H1", "2018 H2","2019 H1", "2019 H2","2020 H1", "2020 H2", "2021"])
plt.show()

#%% md
Graph of number of accidents per state to show backlog

#%%
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(16, 9))
data_prep.groupby(['Year', 'Month', 'State']).count()['City'].unstack().plot(ax=ax, cmap="jet")  #
ax.set_xlabel('Time (Year, Month)', color="#173F74", fontsize=18)
ax.set_ylabel('Number of Accidents', color="#173F74", fontsize=18)
ax.set_title("Development of accidents per week distinguished by the state", fontsize=22)
ax.legend(loc='center right', bbox_to_anchor=(1.1, 0.5), ncol=2)
plt.show()

#%% md
Plot of accidents according to time of day

#%%
hours = [hour for hour, df in data_prep.groupby('Hour')]
plt.plot(hours, data_prep.groupby(['Hour'])['City'].count(), color="#173F74")
plt.xticks(hours)
plt.xlabel('Hour', color="#173F74")
plt.ylabel('Number of accidents', color="#173F74")
plt.title("Histogram of accidents according to the time of day")
plt.show()

#%% md
Plot of accidents according to day of the week

#%%
days = [day for day, df in data_prep.groupby('Weekday')]
plt.plot(days, data_prep.groupby(['Weekday'])['City'].count(), color="#173F74")
plt.xticks(days, ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], rotation=30)
plt.xlabel('Weekday')
plt.ylabel('Number of accidents')
plt.show()

#%% md
Histogram of accidents ordered by state

#%%
plt.title("The US States ordered by the number of accidents")
plt.xticks(rotation=30)
plt.ylabel('Number of accidents')
data_prep.State.value_counts().plot(kind='bar', figsize=(12, 6), color="#173F74")
plt.show()

#%% md
Pie diagram of severity

#%%
fig = plt.figure(figsize=[10, 10])
ax = fig.add_subplot(111)
cmap = plt.cm.cividis
colors = cmap(np.linspace(0., 1., 4))
sizes = data_prep['Severity'].value_counts().sort_index() / data_prep['Severity'].value_counts().sum() * 100
ax.pie(sizes, labels=sizes.index,
       autopct='%1.1f%%', shadow=False, startangle=90, colors=colors, textprops={'fontsize': 14})
# data_prep.Severity.value_counts().plot.pie(cmap="cividis")
ax.set_title("Share of the different severity levels", fontsize=20)
ax.legend(fontsize=14)
plt.show()

#%% md

### 6.3 Multivariate Non-Graphical

#%%

# correlation matrix
data_prep.corr()

#%% md

### 6.4 Multivariate Graphical

#%% md
Correlogram

#%%
fig = plt.gcf()
fig.set_size_inches(16, 9)
fig = sns.heatmap(data_prep.corr(), annot=False, linewidths=1, linecolor='k', square=True, mask=False,
                  vmin=-1, vmax=1, cbar_kws={"orientation": "vertical"}, cbar=True, cmap="cividis")
sns.set(style='ticks')
plt.title("Correlogram of all features", fontsize=20)

#%% md

US map simple: scatterplot based on latitude and longitude data grouped by severity

#%%
plt.figure(figsize=(15, 9))
sns.scatterplot(x="Start_Lng", y="Start_Lat", data=data_prep, hue="Severity", legend="auto",
                s=2, palette="cividis", alpha=0.5)
plt.title("Location of all accidents in the USA in the time from 2016 to 2020, distinguished by severity")
plt.show()

#%% md
US map complex: scatterplot based on latitude and longitude data grouped by state

#%%
state_list = data_prep["State"].unique()
sorted_state_list = sorted(state_list)
plt.figure(figsize=(16, 9))
g = sns.scatterplot(x="Start_Lng", y="Start_Lat", data=data_prep, hue="State", hue_order=sorted_state_list,
                    legend="auto", s=3, palette="cividis", alpha=0.3)
g.legend(loc='center right', bbox_to_anchor=(1.1, 0.5), ncol=2)
plt.title("Location of all accidents in the USA in the time from 2016 to 2020, distinguished by State")
plt.show()



#%%

#import plotly.express as px
#fig = px.scatter(data_prep, x="Start_Lng", y="Start_Lat", color="Year", facet_col="Severity", facet_row="Nautical_Twilight")
#fig.show()

#%% md

### 6.5 Comparison of 2019 with 2020

#### 6.5.1 Preparation Part 1

#%%

data_prep_wo_bias = data_prep.copy(deep=True)

# dropping states which cause huge bias

data_prep_wo_bias = data_prep_wo_bias[data_prep_wo_bias['State'] != 'CA']
data_prep_wo_bias = data_prep_wo_bias[data_prep_wo_bias['State'] != 'FL']


#%% md

#### 6.5.2
#%% md
Graph of number of accidents per state to show backlog

#%%
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(18, 8))
data_prep_wo_bias.groupby(['Year', 'Week', 'State']).count()['City'].unstack().plot(ax=ax, cmap="jet")  #
ax.set_xlabel('Month', color="#173F74", fontsize=14)
ax.set_ylabel('Number of Accidents', color="#173F74", fontsize=14)
ax.set_title("Development of accidents per week distinguished by the severity", fontsize=14)
ax.legend(loc='center right', bbox_to_anchor=(1.1, 0.5), ncol=2)
plt.show()

#%% md
Graph of number of accidents of severity 1 to show that it mainly depends on a short time period

#%%
data_prep_sev_1 = data_prep[data_prep.Severity == 1]

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(18, 8))
data_prep_sev_1.groupby(['Year', 'Month', 'State']).count()['City'].unstack().plot(ax=ax, cmap="cividis")

ax.set_xlabel('Week', color="#173F74", fontsize=14)
ax.set_ylabel('Number of Accidents', color="#173F74", fontsize=14)
ax.set_title("Development of severity level 1 accidents per week distinguished by the state", fontsize=20,
             color="#173F74")
ax.legend(loc='center right', bbox_to_anchor=(1.12, 0.5), ncol=2)
plt.show()

#%% md
Graph of number of accidents of severity 1 to show that it mainly depends on a short time period

#%%
data_prep_sev_2 = data_prep[data_prep.Severity == 2]

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(18, 8))

data_prep_sev_2.groupby(['Year', 'Month', 'State']).count()['City'].unstack().plot(ax=ax, cmap="cividis")

ax.set_xlabel('Week', color="#173F74", fontsize=14)
ax.set_ylabel('Number of Accidents', color="#173F74", fontsize=14)
ax.set_title("Development of severity level 2 accidents per week distinguished by the state", fontsize=20,
             color="#173F74")
ax.legend(loc='center right', bbox_to_anchor=(1.12, 0.5), ncol=2)
plt.show()

#%% md

#### 6.5.3 Preparation Part 2

Splitting into first half of 2019 and first half of 2020

#%%

# For 2020
data_prep_wo_bias_2020 = data_prep_wo_bias[data_prep_wo_bias.Year == 2020]
data_prep_wo_bias_2020_h1 = data_prep_wo_bias_2020[data_prep_wo_bias_2020.Week <= 26]

# For 2019
data_prep_wo_bias_2019 = data_prep_wo_bias[data_prep_wo_bias.Year == 2019]
data_prep_wo_bias_2019_h1 = data_prep_wo_bias_2019[data_prep_wo_bias_2019.Week <= 26]
# Needed to exclude last days of 2019 who are counted towards the first week of the new year:
data_prep_wo_bias_2019_h1 = data_prep_wo_bias_2019_h1[data_prep_wo_bias_2019_h1.Month <= 6]

#%% md

#### 6.5.4 Rerun the existing graphs with the reduced data set

#%% md
Stacked histogram of Hour colored by Severity in 2019 H1

#%%
# Prepare data
x_var = 'Hour'
groupby_var = 'Severity'
df_agg = data_prep_wo_bias_2019_h1.loc[:, [x_var, groupby_var]].groupby(groupby_var)
vals = [data_prep_wo_bias_2019_h1[x_var].values.tolist() for i, data_prep_wo_bias_2019_h1 in df_agg]

# Draw
plt.figure(figsize=(16, 9), dpi=80)
colors = [plt.cm.cividis(i / float(len(vals))) for i in range(len(vals))]
n, bins, patches = plt.hist(vals, data_prep_wo_bias_2019_h1[x_var].unique().__len__(), stacked=True, density=False,
                            color=colors[:len(vals)])

# Decoration
plt.legend(
    {group: col for group, col in zip(np.unique(data_prep_wo_bias_2019_h1[groupby_var]).tolist(), colors[:len(vals)])},
    fontsize=18)
plt.title(f"Stacked Histogram of ${x_var}$ colored by ${groupby_var}$ in 2019 H1", fontsize=22)
plt.xlabel(x_var, fontsize=18)
plt.ylabel("Number of Accidents", fontsize=18)
hour_list = ['0 am', '1 am', '2 am', '3 am', '4 am', '5 am', '6 am', '7 am', '8 am', '9 am', '10 am', '11 am',
             '12 pm', '1 pm', '2 pm', '3 pm', '4 pm', '5 pm', '6 pm', '7 pm', '8 pm', '9 pm', '10 pm', '11 pm',
             '11:59:59 pm']
plt.xticks(bins, hour_list, rotation=90, horizontalalignment='left', fontsize=18)
plt.yticks(fontsize=18)
plt.show()

#%% md
Stacked histogram of Hour colored by Severity in 2020 H1

#%%
# Prepare data
x_var = 'Hour'
groupby_var = 'Severity'
df_agg = data_prep_wo_bias_2020_h1.loc[:, [x_var, groupby_var]].groupby(groupby_var)
vals = [data_prep_wo_bias_2020_h1[x_var].values.tolist() for i, data_prep_wo_bias_2020_h1 in df_agg]

# Draw
plt.figure(figsize=(16, 9), dpi=80)
colors = [plt.cm.cividis(i / float(len(vals))) for i in range(len(vals))]
n, bins, patches = plt.hist(vals, data_prep_wo_bias_2020_h1[x_var].unique().__len__(), stacked=True, density=False,
                            color=colors[:len(vals)])

# Decoration
plt.legend(
    {group: col for group, col in zip(np.unique(data_prep_wo_bias_2020_h1[groupby_var]).tolist(), colors[:len(vals)])},
    fontsize=18)
plt.title(f"Stacked Histogram of ${x_var}$ colored by ${groupby_var}$ in 2020 H1", fontsize=22)
plt.xlabel(x_var, fontsize=18)
plt.ylabel("Number of Accidents", fontsize=18)
hour_list = ['0 am', '1 am', '2 am', '3 am', '4 am', '5 am', '6 am', '7 am', '8 am', '9 am', '10 am', '11 am',
             '12 pm', '1 pm', '2 pm', '3 pm', '4 pm', '5 pm', '6 pm', '7 pm', '8 pm', '9 pm', '10 pm', '11 pm',
             '11:59:59 pm']
plt.xticks(bins, hour_list, rotation=90, horizontalalignment='left', fontsize=18)
plt.yticks(fontsize=18)
plt.show()

#%% md
Stacked histogram of Weekday colored by Severity in 2019 H1

#%%
# Prepare data
x_var = 'Weekday'
groupby_var = 'Severity'
df_agg = data_prep_wo_bias_2019_h1.loc[:, [x_var, groupby_var]].groupby(groupby_var)
vals = [data_prep_wo_bias_2019_h1[x_var].values.tolist() for i, data_prep_wo_bias_2019_h1 in df_agg]

# Draw
plt.figure(figsize=(16, 9), dpi=80)
colors = [plt.cm.cividis(i / float(len(vals))) for i in range(len(vals))]
n, bins, patches = plt.hist(vals, data_prep_wo_bias_2019_h1[x_var].unique().__len__(), stacked=True, density=False,
                            color=colors[:len(vals)])

# Decoration
plt.legend(
    {group: col for group, col in zip(np.unique(data_prep_wo_bias_2019_h1[groupby_var]).tolist(), colors[:len(vals)])},
    fontsize=18)
plt.title(f"Stacked Histogram of ${x_var}$ colored by ${groupby_var}$ in 2019 H1", fontsize=22)
plt.xlabel(x_var, fontsize=18)
plt.ylabel("Number of Accidents", fontsize=18)
weekday_list = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", None]
plt.xticks(bins, weekday_list, rotation=90, horizontalalignment='left', fontsize=18)
plt.yticks(fontsize=18)
plt.show()

#%% md
Stacked histogram of Weekday colored by Severity in 2020 H1

#%%
# Prepare data
x_var = 'Weekday'
groupby_var = 'Severity'
df_agg = data_prep_wo_bias_2020_h1.loc[:, [x_var, groupby_var]].groupby(groupby_var)
vals = [data_prep_wo_bias_2020_h1[x_var].values.tolist() for i, data_prep_wo_bias_2020_h1 in df_agg]

# Draw
plt.figure(figsize=(16, 9), dpi=80)
colors = [plt.cm.cividis(i / float(len(vals))) for i in range(len(vals))]
n, bins, patches = plt.hist(vals, data_prep_wo_bias_2020_h1[x_var].unique().__len__(), stacked=True, density=False,
                            color=colors[:len(vals)])

# Decoration
plt.legend(
    {group: col for group, col in zip(np.unique(data_prep_wo_bias_2020_h1[groupby_var]).tolist(), colors[:len(vals)])},
    fontsize=18)
plt.title(f"Stacked Histogram of ${x_var}$ colored by ${groupby_var}$ in 2020 H1", fontsize=22)
plt.xlabel(x_var, fontsize=18)
plt.ylabel("Number of Accidents", fontsize=18)
weekday_list = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", None]
plt.xticks(bins, weekday_list, rotation=90, horizontalalignment='left', fontsize=18)
plt.yticks(fontsize=18)
plt.show()

#%% md
US map: scatterplot based on latitude and longitude data for 2019 H1

#%%
plt.figure(figsize=(16, 9))
sns.scatterplot(x="Start_Lng", y="Start_Lat", data=data_prep_wo_bias_2019_h1, hue="Severity", legend="auto",
                s=2, palette="cividis", alpha=0.7)
plt.title("Location of all accidents in the USA in the first half of 2019, distinguished by severity")
plt.show()

#%% md
US map: scatterplot based on latitude and longitude data for 2020 H1

#%%
plt.figure(figsize=(16, 9))
sns.scatterplot(x="Start_Lng", y="Start_Lat", data=data_prep_wo_bias_2020_h1, hue="Severity", legend="auto",
                s=2, palette="cividis", alpha=0.7)
plt.title("Location of all accidents in the USA in the first half of 2020, distinguished by severity")
plt.show()

#%% md
Stacked histogram of Week colored by Severity in 2019 H1

#%%
# Prepare data
x_var = 'Week'
groupby_var = 'Severity'
df_agg = data_prep_wo_bias_2020_h1.loc[:, [x_var, groupby_var]].groupby(groupby_var)
vals = [data_prep_wo_bias_2020_h1[x_var].values.tolist() for i, data_prep_wo_bias_2020_h1 in df_agg]

# Draw
plt.figure(figsize=(16, 9), dpi=80)
colors = [plt.cm.cividis(i / float(len(vals))) for i in range(len(vals))]
n, bins, patches = plt.hist(vals, data_prep_wo_bias_2020_h1[x_var].unique().__len__(), stacked=True, density=False,
                            color=colors[:len(vals)])

# Decoration
plt.legend(
    {group: col for group, col in zip(np.unique(data_prep_wo_bias_2020_h1[groupby_var]).tolist(), colors[:len(vals)])},
    fontsize=18)
plt.title(f"Stacked Histogram of ${x_var}$ colored by ${groupby_var}$ in 2020", fontsize=22)
plt.xlabel(x_var, fontsize=18)
plt.ylabel("Number of Accidents", fontsize=18)
# plt.ylim(0, 40)
#week_list = list(range(bins))
plt.xticks(rotation=0, horizontalalignment='right', fontsize=18)
plt.yticks(fontsize=18)
plt.show()

#%% md
Stacked histogram of Week colored by Severity in 2020 H1

#%%
# Prepare data
x_var = 'Week'
groupby_var = 'Severity'
df_agg = data_prep_wo_bias_2019_h1.loc[:, [x_var, groupby_var]].groupby(groupby_var)
vals = [data_prep_wo_bias_2019_h1[x_var].values.tolist() for i, data_prep_wo_bias_2019_h1 in df_agg]

# Draw
plt.figure(figsize=(16, 9), dpi=80)
colors = [plt.cm.cividis(i / float(len(vals))) for i in range(len(vals))]
n, bins, patches = plt.hist(vals, data_prep_wo_bias_2019_h1[x_var].unique().__len__(), stacked=True, density=False,
                            color=colors[:len(vals)])

# Decoration
plt.legend(
    {group: col for group, col in zip(np.unique(data_prep_wo_bias_2019_h1[groupby_var]).tolist(), colors[:len(vals)])},
    fontsize=18)
plt.title(f"Stacked Histogram of ${x_var}$ colored by ${groupby_var}$ in 2019", fontsize=22)
plt.xlabel(x_var, fontsize=18)
plt.ylabel("Number of Accidents", fontsize=18)
# plt.ylim(0, 40)
#month_list = ['January', 'February', 'March', 'April', 'May', 'June', None]
#plt.xticks(bins, month_list, rotation=90, horizontalalignment='left', fontsize=18)
plt.yticks(fontsize=18)
plt.show()

#%% md
Graph of number of accidents per state to show backlog

#%%
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(18, 8))
data_prep_wo_bias_2019_h1.groupby(['Year', 'Week', 'Severity']).count()['City'].unstack().plot(ax=ax, cmap="cividis")
data_prep_wo_bias_2020_h1.groupby(['Year', 'Week', 'Severity']).count()['City'].unstack().plot(ax=ax, cmap="cividis",
                                                                                               linestyle="dashed")
week_list = list(range(26))
week_list_ = list(range(1, 27, 1))
ax.set_xticks(week_list)
ax.set_xticklabels(week_list_)
ax.set_xlabel('Week', color="#173F74", fontsize=18)
ax.set_ylabel('Number of Accidents', color="#173F74", fontsize=18)
ax.set_title("Development of accidents per week distinguished by the severity", fontsize=22)
ax.legend(loc='center right', bbox_to_anchor=(1.12, 0.5), ncol=2, title="2019         2020")
plt.show()

#%% md

---
## 7 Feature Engineering

#%%

# Preparation
data_encoding = data_prep_wo_bias.copy(deep=True)

# Only data 01.01.2017 - 30.06.2020
data_encoding = data_encoding[data_encoding['Year'] >= 2017]
data_encoding = data_encoding[(data_encoding['Year'] != 2020) | (data_encoding['Month'] < 7)]
# Drop severity 1
data_encoding = data_encoding[data_encoding['Severity'] != 1]

# Reset index
data_encoding.reset_index(inplace=True, drop=True)
data_encoding.head()
#%%

data_encoding['Severity'].value_counts()
#%% md

### 7.1 Type Conversion

- attempt freq encoding for Counties
    - attempt ordinal encoding for Streets, Cities
    - one hot encoding for States
    - binary:

#### 7.1.1 Ordinal Encoding

#%%
# Ordinal encoding instead of frequency encoding -> time reasons
ordinal_encoder = OrdinalEncoder()

data_encoding[['Street']] = ordinal_encoder.fit_transform(data_encoding[['Street']])
print(ordinal_encoder.categories_)

data_encoding[['City']] = ordinal_encoder.fit_transform(data_encoding[['City']])
print(ordinal_encoder.categories_)

#%% md

#### 7.1.2 'Binary' Encoding
Ordinal encoding for Nautical_Twilight with Day/Night values to bool
Ordinal encoding for Side (Left/Right) to bool
... bool_columns

#%%
# L - 0, R - 1
data_encoding[['Side']] = ordinal_encoder.fit_transform(data_encoding[['Side']])
print(ordinal_encoder.categories_)

#%%
# Day - 0, Night 1
data_encoding[['Nautical_Twilight']] = ordinal_encoder.fit_transform(data_encoding[['Nautical_Twilight']])
print(ordinal_encoder.categories_)

#%%

for column in bool_columns:
    data_encoding[[column]] = ordinal_encoder.fit_transform(data_encoding[[column]])
    print(ordinal_encoder.categories_)

#%% md

#### 7.1.3 OneHot Encoding

#%%
# Initialize encoder
ohc = OneHotEncoder()
#%% md
For States

#%%
one_hot_encoded = ohc.fit_transform(data_encoding.State.values.reshape(-1, 1)).toarray()

# Generate array with correct column names
categories = ohc.categories_
column_names = []

for category in categories[0]:
    column_name = category
    column_names.append(column_name)

# Set correct column names
one_hot_data = pd.DataFrame(one_hot_encoded, columns=column_names)

#%%
# Delete one column to avoid the dummy variable trap
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

#%%
# delete one column to avoid the dummy variable trap
one_hot_data.drop(one_hot_data.columns[-1], axis=1, inplace=True)

# combining ohc dataframe to previous df
data_encoding = pd.concat([data_encoding, one_hot_data], axis=1)

data_encoding.drop('Wind_Direction', axis=1, inplace=True)
data_encoding.head()

#%% md

#### 7.1.4 Manual Encoding

#%%
# Initialize OneHotEncoder
one_hot_encoder = OneHotEncoder()

data_one_hot = one_hot_encoder.fit_transform(data_encoding[['Weather_Condition']])
data_one_hot_array = data_one_hot.toarray()

# Generate array with correct column names
categories = one_hot_encoder.categories_
column_names = []

for category in categories[0]:
    column_name = category
    column_names.append(column_name)

# Set correct column names
data_one_hot = pd.DataFrame(data_one_hot_array, columns=column_names)

# Delete one column to avoid the dummy variable trap
data_one_hot.drop(data_one_hot.columns[-1], axis=1, inplace=True)  # drop last n rows
data_one_hot.head()
#%%
# Concatenate OneHot columns according to weather_value dict
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
For County

#%%
# Frequency dict
county_dict = data_encoding['County'].value_counts().to_dict()

#%%
# Ordinal Freq Encoding
county_array = county_dict.keys()  # Keys from the dict are now arranged in descending order in the array. Most frequent -> least

# Ordinal Encoding according to frequency hierarchy
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
# Converting to unix epoch time and adding to df
data_encoding['N_Start_Time'] = d.view('int64')

# Dropping original Start_Time column
data_encoding.drop('Start_Time', axis=1, inplace=True)

#%% md

### 7.3 Normalization

#%%

data_final = data_encoding.copy(deep=True)

# divide columns into dependant and independent
data_independent = data_final.drop(['Severity'], axis=1)
data_dependant = data_final[['Severity']]

data_independent.head()

#%%
# Scale each column so that every feature/parameter has equal weight

x = data_independent.values  # returns a numpy array with all the values of the dataframe
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

# TODO:  restore column names. Irene: proposed change. But it might be taking longer than before
data_independent = pd.DataFrame(x_scaled, index=data_independent.index, columns=data_independent.columns)
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
severity_values
#%%
# ensuring every severity level has equal proportions in the data
def balanced_subsample(y, size=None, random_state=None):  # returns a List with randomly chosen row numbers
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


rows = balanced_subsample(data_tbsampled['severity'], size=20000, random_state=0)

#%%
data_downsampled = data_tbsampled.iloc[rows, :]

X_train = data_downsampled.drop('severity', axis=1)
Y_train = data_downsampled['severity']

#%% md
Test Data

#%%
# Making sure train Y has the same number of rows

# concatenate data for sampling
data_test_tbsampled = X_test.copy(deep=True)
data_test_tbsampled['severity'] = Y_test

data_test_tbsampled.reset_index(inplace=True, drop=True)

#%%
data_test_sampled = resample(data_test_tbsampled, replace=False, n_samples=2000, random_state=0)

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
report = pd.DataFrame(columns=['Model', 'Mean Acc. Training', 'Standard Deviation', 'Acc. Test'])

#%% md

#### 8.3.1 KNN
#%%

knnmodel = KNeighborsClassifier(n_jobs=-1)

param_grid = {
    'n_neighbors': [3, 4, 5]
}

CV_knnmodel = GridSearchCV(estimator=knnmodel, param_grid=param_grid, cv=10)
CV_knnmodel.fit(X_train, Y_train)
print(CV_knnmodel.best_params_)

# use the best parameters
knnmodel = knnmodel.set_params(**CV_knnmodel.best_params_)
knnmodel.fit(X_train, Y_train)

#%%
Y_test_pred = knnmodel.predict(X_test)
acctest = accuracy_score(Y_test, Y_test_pred)

#%%
# fill report
report.loc[len(report)] = ['k-NN (grid)',
                           CV_knnmodel.cv_results_['mean_test_score'][CV_knnmodel.best_index_],
                           CV_knnmodel.cv_results_['std_test_score'][CV_knnmodel.best_index_],
                           acctest]
print(report.loc[len(report) - 1])

#%%
# visualize confusion matrix

cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)

#%%
plot_confusion_matrix(knnmodel, X_test, Y_test, labels=[2, 3, 4],
                      cmap=plt.cm.Blues, values_format='d')

#%% md

#### 8.3.2 Decision Trees
#%%
dtree_model = RandomForestClassifier(n_jobs=-1)

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 6, 7, 8]  # why not more? it is suggested that the best number of splits lie between 5-8
}

CV_dtree_model = GridSearchCV(estimator=dtree_model, param_grid=param_grid, cv=10)
CV_dtree_model.fit(X_train, Y_train)
print(CV_dtree_model.best_params_)

# use the best parameters
dtree_model = dtree_model.set_params(**CV_dtree_model.best_params_)
dtree_model.fit(X_train, Y_train)
#%%
# predict test data
Y_test_pred = dtree_model.predict(X_test)
acctest = accuracy_score(Y_test, Y_test_pred)

#%%
# fill report
report.loc[len(report)] = ['Random Forest Classifier (grid)',
                           CV_dtree_model.cv_results_['mean_test_score'][CV_dtree_model.best_index_],
                           CV_dtree_model.cv_results_['std_test_score'][CV_dtree_model.best_index_],
                           acctest]
print(report.loc[len(report) - 1])

#%%
# visualize confusion matrix

cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)

#%%
plot_confusion_matrix(dtree_model, X_test, Y_test, labels=[2, 3, 4], cmap=plt.cm.Blues, values_format='d')
#%% md

#### 8.3.3 Neural Networks
#%%
nnetmodel = MLPClassifier(max_iter=400)

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
Y_test_pred = nnetmodel.predict(X_test)
acctest = accuracy_score(Y_test, Y_test_pred)

#%%
# fill report
report.loc[len(report)] = ['Neural Networks (grid)',
                           CV_nnetmodel.cv_results_['mean_test_score'][CV_nnetmodel.best_index_],
                           CV_nnetmodel.cv_results_['std_test_score'][CV_nnetmodel.best_index_],
                           acctest]
print(report.loc[len(report) - 1])

#%%
# visualize confusion matrix
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)

#%%
plot_confusion_matrix(nnetmodel, X_test, Y_test, labels=[2, 3, 4],
                      cmap=plt.cm.Blues, values_format='d')

#%% md

### 8.5 Prediction driving factors

# SHAP diagram

