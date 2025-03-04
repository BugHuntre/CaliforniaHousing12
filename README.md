California Housing Price Analysis Documentation
Overview
This Jupyter Notebook contains an in-depth analysis of the California housing dataset. The objective is to eLibraries Used
The following libraries are imported for data handling, analysis, and modeling:
- pandas for data manipulation
- numpy for numerical computations
- matplotlib and seaborn for data visualization
- scikit-learn for machine learning models and preprocessing
Data Loading
The dataset is loaded using pandas:
import pandas as pd
import numpy as np
df = pd.read_csv("housing.csv")
df.head()
Exploratory Data Analysis (EDA)
Key steps performed in EDA include:
1. Checking for missing values using df.isnull().sum()
2. Statistical summary using df.describe()
3. Data visualization using histograms, scatter plots, and correlation heatmaps:
 - Distribution of numerical features using seaborn.histplot
 - Scatter plots to examine relationships between median_house_value and other features
 - Correlation heatmap to identify strongly correlated features
4. Identifying categorical vs. numerical attributes
Data Preprocessing
1. Handling Missing Values:
 - The total_bedrooms column has missing values, which are filled using mode:
 df["total_bedrooms"].fillna(df["total_bedrooms"].mode()[0], inplace=True)
 - Other missing values, if present, are analyzed and addressed appropriately.
2. Encoding Categorical Variables:
 - ocean_proximity is a categorical feature and is transformed using OneHotEncoder.
3. Feature Scaling:
 - Numerical attributes are standardized using StandardScaler to normalize the data.
4. Splitting Dataset:
 - The dataset is split into training and testing sets using train_test_split to ensure proper model evaluationModel Building
Several machine learning models are trained and evaluated:
1. Linear Regression:
 - A simple baseline model to understand linear relationships.
 from sklearn.linear_model import LinearRegression
 model = LinearRegression()
 model.fit(X_train, y_train)
2. Decision Tree Regressor:
 - Captures non-linear relationships but may overfit.
3. Random Forest Regressor:
 - An ensemble method that improves accuracy by reducing overfitting.
4. Gradient Boosting Regressor:
 - A boosting technique that refines weak models iteratively.
Model Evaluation
Each model is evaluated using:
- Mean Absolute Error (MAE): Measures the average absolute difference between actual and predicted val- Mean Squared Error (MSE): Penalizes large errors more than MAE.
- R-squared Score (R²): Represents the proportion of variance explained by the model.
Example evaluation:
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae}, MSE: {mse}, R²: {r2}")
Conclusion
- The performance of different models is compared based on evaluation metrics.
- The best-performing model is selected for final predictions.
- Insights are drawn regarding key features influencing house prices.
---
This documentation provides a structured and detailed overview of the California housing price analysis pro
