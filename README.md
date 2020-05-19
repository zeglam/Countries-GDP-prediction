# Countries-GDP-prediction

#### Data Source
We have 'Countries of The World' data set 
(from kaggle: Fernando Lasso: https://www.kaggle.com/fernandol/countries-of-the-world).

#### Data Description
This dataset have each country as a data point (227 countries in total), and for each, we have 20 columns, each column represents a different aspect or measure of the specific country. 

#### Project Goal
The goal of the project is to understand this dataset, get some insights from it, and finally to train a model that can predict GDP per capita for each country. 

![](/regional-average-gdp-per-capita.png)

#### Conclusion 
4 different learning regressors (Linear Regression, SVM, Random Forest, and Gradiant Boosting) were tested, and we have acheived the best prediction performance using Random Forest, followed by Gradiant Boosting, then Linear Regression, while SVM acheived the worst performance of the four.

The best prediction performance was acheived with a Random Forest regressor, using all features in the dataset, and resulted in the following metrics:

* Mean Absolute Error (MAE): 2142.13
* Root mean squared error (RMSE): 3097.19
* R-squared Score (R2_Score): 0.8839

(gdp_per_capita values in the dataset ranges from 500 to 55100 USD).

![](/Prediction_performance.png)
