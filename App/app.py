import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

#-------------
#layout design 
#-------------

st.title('Country GDP Estimation Tool')
st.write('''
         This app will estimate the GDP per capita for a country, given some 
         attributes for that specific country as input.
         
         Please fill in the attributes below, then hit the GDP Estimate button
         to get the estimate. 
         ''')

st.header('Input Attributes')
att_popl = st.number_input('Population (Example: 7000000)', min_value=1e4, max_value=2e9, value=2e7)
att_area = st.slider('Area (sq. Km)', min_value= 2.0, max_value= 17e6, value=6e5, step=1e4)
att_dens = st.slider('Population Density (per sq. mile)', min_value= 0, max_value= 12000, value=400, step=10)
att_cost = st.slider('Coastline/Area Ratio', min_value= 0, max_value= 800, value=30, step=10)
att_migr = st.slider('Annual Net Migration (migrant(s)/1,000 population)', min_value= -20, max_value= 25, value=0, step=2) 
att_mort = st.slider('Infant mortality (per 1000 births)', min_value= 0, max_value=195, value=40, step=10)
att_litr = st.slider('Population literacy Percentage', min_value= 0, max_value= 100, value=80, step=5)
att_phon = st.slider('Phones per 1000', min_value= 0, max_value= 1000, value=250, step=25)
att_arab = st.slider('Arable Land (%)', min_value= 0, max_value= 100, value=25, step=2)
att_crop = st.slider('Crops Land (%)', min_value= 0, max_value= 100, value=5, step=2)
att_othr = st.slider('Other Land (%)', min_value= 0, max_value= 100, value=70, step=2)
st.text('(Arable, Crops, and Other land should add up to 100%)')
att_clim = st.selectbox('Climate', options=(1, 1.5, 2, 2.5, 3))
st.write('''
         * 1: Mostly hot (like: Egypt and Australia)
         * 1.5: Mostly hot and Tropical (like: China and Cameroon)
         * 2: Mostly tropical (like: The Bahamas and Thailand)
         * 2.5: Mostly cold and Tropical (like: India)
         * 3: Mostly cold (like: Argentina and Belgium)
         '''
         )
att_brth = st.slider('Annual Birth Rate (births/1,000)', min_value= 7, max_value= 50, value=20, step=2)
att_deth = st.slider('Annual Death Rate (deaths/1,000)', min_value= 2, max_value= 30, value=10, step=2)
att_agrc = st.slider('Agricultural Economy', min_value= 0.0, max_value= 1.0, value=0.15, step=0.05)
att_inds = st.slider('Industrial Economy', min_value= 0.0, max_value= 1.0, value=0.25, step=0.05)
att_serv = st.slider('Services Economy', min_value= 0.0, max_value= 1.0, value=0.60, step=0.05)
st.text('(Agricultural, Industrial, and Services Economy should add up to 1)')
att_regn = st.selectbox('Region', options=(1,2,3,4,5,6,7,8,9,10,11))
st.write('''
         * 1: ASIA (EX. NEAR EAST)
         * 2: BALTICS
         * 3: C.W. OF IND. STATES
         * 4: EASTERN EUROPE
         * 5: LATIN AMER. & CARIB
         * 6: NEAR EAST
         * 7: NORTHERN AFRICA
         * 8: NORTHERN AMERICA
         * 9: OCEANIA
         * 10: SUB-SAHARAN AFRICA 
         * 11: WESTERN EUROPE
         '''
         )

if att_regn == 1:
    att_regn_1 = 1
    att_regn_2 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = 0
elif att_regn == 2: 
    att_regn_2 = 1
    att_regn_1 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = 0
elif att_regn == 3: 
    att_regn_3 = 1
    att_regn_1 = att_regn_2 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = 0
elif att_regn == 4: 
    att_regn_4 = 1
    att_regn_1 = att_regn_3 = att_regn_2 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = 0
elif att_regn == 5: 
    att_regn_5 = 1
    att_regn_1 = att_regn_3 = att_regn_4 = att_regn_2 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = 0
elif att_regn == 6: 
    att_regn_6 = 1
    att_regn_1 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_2 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = 0
elif att_regn == 7: 
    att_regn_7 = 1
    att_regn_1 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_2 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = 0
elif att_regn == 8: 
    att_regn_8 = 1
    att_regn_1 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_2 = att_regn_9 = att_regn_10 = att_regn_11 = 0
elif att_regn == 9: 
    att_regn_9 = 1
    att_regn_1 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_2 = att_regn_10 = att_regn_11 = 0
elif att_regn == 10: 
    att_regn_10 = 1
    att_regn_1 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_2 = att_regn_11 = 0
else: 
    att_regn_11 = 1
    att_regn_1 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_2 = 0

user_input = np.array([att_popl, att_area, att_dens, att_cost, att_migr, 
                       att_mort, att_litr, att_phon, att_arab, att_crop, 
                       att_othr, att_clim, att_brth, att_deth, att_agrc, 
                       att_inds, att_serv, att_regn_1, att_regn_2, att_regn_3,
                       att_regn_4, att_regn_5, att_regn_6, att_regn_7, 
                       att_regn_8, att_regn_9, att_regn_10, att_regn_11]).reshape(1,-1)

#------
# Model
#------

#import dataset
def get_dataset():
    data = pd.read_csv('countries-of-the-world.csv')
    return data

if st.button('Estimate GDP'):
    data = get_dataset()
    
    #fix column names
    data.columns = (["country","region","population","area","density",
                     "coastline_area_ratio","net_migration","infant_mortality",
                     "gdp_per_capita","literacy","phones","arable","crops","other",
                     "climate","birthrate","deathrate","agriculture","industry",
                      "service"])
    
    #Fix data types
    data.country = data.country.astype('category')
    data.region = data.region.astype('category')
    data.density = data.density.astype(str)
    data.density = data.density.str.replace(",",".").astype(float)
    data.coastline_area_ratio = data.coastline_area_ratio.astype(str)
    data.coastline_area_ratio = data.coastline_area_ratio.str.replace(",",".").astype(float)
    data.net_migration = data.net_migration.astype(str)
    data.net_migration = data.net_migration.str.replace(",",".").astype(float)
    data.infant_mortality = data.infant_mortality.astype(str)
    data.infant_mortality = data.infant_mortality.str.replace(",",".").astype(float)
    data.literacy = data.literacy.astype(str)
    data.literacy = data.literacy.str.replace(",",".").astype(float)
    data.phones = data.phones.astype(str)
    data.phones = data.phones.str.replace(",",".").astype(float)
    data.arable = data.arable.astype(str)
    data.arable = data.arable.str.replace(",",".").astype(float)
    data.crops = data.crops.astype(str)
    data.crops = data.crops.str.replace(",",".").astype(float)
    data.other = data.other.astype(str)
    data.other = data.other.str.replace(",",".").astype(float)
    data.climate = data.climate.astype(str)
    data.climate = data.climate.str.replace(",",".").astype(float)
    data.birthrate = data.birthrate.astype(str)
    data.birthrate = data.birthrate.str.replace(",",".").astype(float)
    data.deathrate = data.deathrate.astype(str)
    data.deathrate = data.deathrate.str.replace(",",".").astype(float)
    data.agriculture = data.agriculture.astype(str)
    data.agriculture = data.agriculture.str.replace(",",".").astype(float)
    data.industry = data.industry.astype(str)
    data.industry = data.industry.str.replace(",",".").astype(float)
    data.service = data.service.astype(str)
    data.service = data.service.str.replace(",",".").astype(float)
    
    #fix missing data
    data['net_migration'].fillna(0, inplace=True)
    data['infant_mortality'].fillna(0, inplace=True)
    data['gdp_per_capita'].fillna(2500, inplace=True)
    data['literacy'].fillna(data.groupby('region')['literacy'].transform('mean'), inplace= True)
    data['phones'].fillna(data.groupby('region')['phones'].transform('mean'), inplace= True)
    data['arable'].fillna(0, inplace=True)
    data['crops'].fillna(0, inplace=True)
    data['other'].fillna(0, inplace=True)
    data['climate'].fillna(0, inplace=True)
    data['birthrate'].fillna(data.groupby('region')['birthrate'].transform('mean'), inplace= True)
    data['deathrate'].fillna(data.groupby('region')['deathrate'].transform('mean'), inplace= True)
    data['agriculture'].fillna(0.17, inplace=True)
    data['service'].fillna(0.8, inplace=True)
    data['industry'].fillna((1 - data['agriculture'] - data['service']), inplace= True)
    
    #Region Transform
    data_final = pd.concat([data,pd.get_dummies(data['region'], prefix='region')], axis=1).drop(['region'],axis=1)
    
    #Data Split
    y = data_final['gdp_per_capita']
    X = data_final.drop(['gdp_per_capita','country'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    
    #model training
    gbm_opt = GradientBoostingRegressor(learning_rate=0.01, n_estimators=500,
                                        max_depth=5, min_samples_split=10, 
                                        min_samples_leaf=1, subsample=0.7,
                                        max_features=7, random_state=101)
    gbm_opt.fit(X_train,y_train)
    
    #making a prediction
    gbm_predictions = gbm_opt.predict(user_input) #user_input is taken from input attrebutes 
    st.write('The estimated GDP per capita is: ', gbm_predictions)




