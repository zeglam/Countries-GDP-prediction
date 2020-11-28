# Countries-GDP-prediction [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/zeglam/countries-gdp-prediction/app.py)

#### Data Source
We have 'Countries of The World' data set 
(from kaggle: Fernando Lasso: https://www.kaggle.com/fernandol/countries-of-the-world).

#### Data Description
This dataset have each country as a data point (227 countries in total), and for each, we have 20 columns, each column represents a different aspect or measure of the specific country. 

#### Project Goal
The goal of the project is to understand this dataset, get some insights from it, and finally to train a model that can predict GDP per capita for each country. 

![](/regional-average-gdp-per-capita.png)

#### Conclusion 
4 different learning regressors (Linear Regression, SVM, Random Forest, and Gradient Boosting) were tested, and we have achieved the best prediction performance using Random Forest, followed by Gradient Boosting, then Linear Regression, while SVM achieved the worst performance of the four.

The best prediction performance was achieved with a Random Forest regressor, using all features in the dataset, and resulted in the following metrics:

* Mean Absolute Error (MAE): 2142.13
* Root mean squared error (RMSE): 3097.19
* R-squared Score (R2_Score): 0.8839

(gdp_per_capita values in the dataset ranges from 500 to 55100 USD).

![](/Prediction_performance.png)

#### Deployment 
I have created a web app using streamlit library to deploy the final model. In the app, the user can input the different attributes of a certain country, and click a button in order to get the estimated GDP per capita for that country. 

please feel free to  try my [app here](https://share.streamlit.io/zeglam/countries-gdp-prediction/app.py), and send me an email if you have any questions or feedback. 
