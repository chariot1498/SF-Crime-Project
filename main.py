import numpy as np 
import pandas as pd
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,log_loss
le = LabelEncoder()




def convert_address(addr):
    if '/' in addr and 'of' not in addr:
        return 1.
    else:
        return 0.


def Hour_to_number(hour):
    if(hour >= 2 and hour < 8): return 1;
    if(hour >= 8 and hour < 12): return 2;
    if(hour >= 12 and hour < 14): return 3;
    if(hour >= 14 and hour < 18): return 4;
    if(hour >= 18 and hour < 22): return 5;
    if(hour < 2 or hour >= 22): return 6;

def feature_engineering(data):
	data['Address_Type'] = list(map(convert_address, data.Address))
	data['Resolution_Type'] = le.fit_transform(data['Resolution'])
	data['PdDistrict_Type'] = le.fit_transform(data['PdDistrict'])
	data['Lat'] = data['Y'] - 37.7749
	data['Long'] = data['X'] + 122.4194
	data['Day'] = data['Dates'].dt.day
	data['Month'] = data['Dates'].dt.month
	data['Year'] = data['Dates'].dt.year-2003
	data['Hour'] = data['Dates'].dt.hour
	data['Minute'] = data['Dates'].dt.minute
	data['WeekOfYear'] = data['Dates'].dt.weekofyear
	data['HourZn'] = preprocessing.scale(list(map(Hour_to_number, data['Dates'].dt.hour)))
	data = data.drop(['PdDistrict','Resolution','Address','X','Y','Id','Dates'],axis = 1)
	data = pd.get_dummies(data, columns=['DayOfWeek','PdDistrict_Type','Resolution_Type', 'Year', 'Month', 'Day', 'Hour', 'Minute'])
	labels = data['Category'].replace({'LARCENY/THEFT': 16, 'SECONDARY CODES': 27, 'OTHER OFFENSES': 21,
	   'WEAPON LAWS': 38, 'WARRANTS': 37, 'VEHICLE THEFT': 36, 'ASSAULT': 1,
	   'NON-CRIMINAL': 20, 'PROSTITUTION': 23, 'BURGLARY': 4, 'DRUG/NARCOTIC': 7,
	   'VANDALISM': 35, 'MISSING PERSON': 19, 'ROBBERY': 25, 'SEX OFFENSES FORCIBLE': 28,
	   'FRAUD': 13, 'SUSPICIOUS OCC': 32, 'LOITERING': 18, 'RUNAWAY': 26, 'DRUNKENNESS': 8,
	   'STOLEN PROPERTY': 30, 'TRESPASS': 34, 'EMBEZZLEMENT': 9, 'LIQUOR LAWS': 17,
	   'FORGERY/COUNTERFEITING': 12, 'KIDNAPPING': 15,
	   'DRIVING UNDER THE INFLUENCE': 6, 'DISORDERLY CONDUCT': 5,
	   'RECOVERED VEHICLE': 24, 'BRIBERY': 3, 'SUICIDE': 31, 'ARSON': 0,
	   'FAMILY OFFENSES': 11, 'BAD CHECKS': 2, 'EXTORTION': 10, 'GAMBLING': 14,
	   'SEX OFFENSES NON FORCIBLE': 29, 'PORNOGRAPHY/OBSCENE MAT': 22, 'TREA': 33})
	data = data.drop(['Category'],axis=1)    
	return data,labels

def feature_engineering_test(data):
	data['Address_Type'] = list(map(convert_address, data.Address))
	data['Resolution_Type'] = le.fit_transform(data['Resolution'])
	data['PdDistrict_Type'] = le.fit_transform(data['PdDistrict'])
	data['Lat'] = data['Y'] - 37.7749
	data['Long'] = data['X'] + 122.4194
	data['Day'] = data['Dates'].dt.day
	data['Month'] = data['Dates'].dt.month
	data['Year'] = data['Dates'].dt.year-2003
	data['Hour'] = data['Dates'].dt.hour
	data['Minute'] = data['Dates'].dt.minute
	data['WeekOfYear'] = data['Dates'].dt.weekofyear
	data['HourZn'] = preprocessing.scale(list(map(Hour_to_number, data['Dates'].dt.hour)))
	data = data.drop(['PdDistrict','Resolution','Address','X','Y','Dates'],axis = 1)
	data = pd.get_dummies(data, columns=['DayOfWeek','PdDistrict_Type','Resolution_Type', 'Year', 'Month', 'Day', 'Hour', 'Minute'])
	return data

def get_inputs(train,test):
	train = pd.read_csv(train,parse_dates=['Dates'], index_col=False)
	test = pd.read_csv(test,parse_dates=['Dates'], index_col=False)
	train.drop(['Descript'],axis=1,inplace=True)
	return train,test


train = sys.argv[1]
test  = sys.argv[2]
#print(1)
train_df,test_df = get_inputs(train,test)
X,Y = feature_engineering(train_df)
#print(2)
X_test = feature_engineering_test(test_df)
#print(3)
clf = RandomForestClassifier(max_depth=28, n_estimators=100)
clf.fit(X,Y)
#print(4)
pred = clf.predict_proba(X_test.drop(['Id'],axis=1,inplace=False))
#print(5)
df2 = pd.DataFrame(pred)
df2.columns=['ARSON' ,'ASSAULT' ,'BAD CHECKS', 'BRIBERY' ,'BURGLARY' ,'DISORDERLY CONDUCT',
'DRIVING UNDER THE INFLUENCE', 'DRUG/NARCOTIC' ,'DRUNKENNESS' ,'EMBEZZLEMENT',
'EXTORTION', 'FAMILY OFFENSES', 'FORGERY/COUNTERFEITING', 'FRAUD', 'GAMBLING',
'KIDNAPPING', 'LARCENY/THEFT', 'LIQUOR LAWS', 'LOITERING', 'MISSING PERSON',
'NON-CRIMINAL', 'OTHER OFFENSES', 'PORNOGRAPHY/OBSCENE MAT', 'PROSTITUTION',
'RECOVERED VEHICLE', 'ROBBERY', 'RUNAWAY', 'SECONDARY CODES',
'SEX OFFENSES FORCIBLE', 'SEX OFFENSES NON FORCIBLE' ,'STOLEN PROPERTY',
'SUICIDE', 'SUSPICIOUS OCC', 'TREA','TRESPASS', 'VANDALISM' ,'VEHICLE THEFT',
'WARRANTS', 'WEAPON LAWS']
df2.insert(0,'Id',X_test['Id'])
df2.to_csv('result.csv',index_label=False,index=False)



