# 🌦️ Weather Prediction Using Decision Tree
## 📌 Overview
This project focuses on predicting weather conditions using Decision Tree, a supervised machine learning algorithm. It leverages historical weather data to provide interpretable and accurate forecasts for parameters such as temperature, humidity, precipitation, and wind speed. The model is built using Python in Google Colab, utilizing scikit-learn for implementation.

## 🎯 Objectives
* Predict short-term weather conditions with reliable accuracy.

* Select and process relevant features like temperature, wind, and humidity.

* Implement and evaluate a Decision Tree model using standard ML metrics.

* Visualize the model for interpretability.

* Compare performance with traditional forecasting methods.

* Explore improvements via hyperparameter tuning and pruning.

## 📂 Dataset
The dataset contains the following columns:

* date – Date of observation (YYYY-MM-DD)

* precipitation – Rain, snow, etc.

* temp_max – Maximum daily temperature

* temp_min – Minimum daily temperature

* wind – Wind speed

* weather – Weather condition (target variable)

📎 Dataset Link

## 🧠 Algorithm – Decision Tree
* Model Type: Supervised classification

* Splitting Criteria: Entropy (Information Gain)

* **Key Concepts:**

     * Root, internal, and leaf nodes

     * Recursive splitting of data

     * Stopping criteria: max depth, minimum samples, purity

## 📌 Improvements:

* Hyperparameter tuning (max_depth, min_samples_split, etc.)

* Pruning

* Ensemble methods like Random Forest (future scope)

## 🔧 Methodology
* **1.Data Collection & Preprocessing**

    * Handle missing values (using forward fill)

    * Convert categorical data with one-hot encoding

* **2.Model Training**

    * Split data: 80% training, 20% testing

    * Train DecisionTreeClassifier using scikit-learn

* **3.Evaluation**

    * Metrics: Accuracy, Precision, Recall, F1 Score

    * Confusion matrix and classification reports

* **4.Visualization**

    * Decision tree plotted using plot_tree()

## 📊 Results
* Training Accuracy: ~99.7%

* Testing Accuracy: ~89%

* Confusion Matrix and Classification Report show strong performance for common conditions like rain and sun, with room for improvement in rarer categories like fog or drizzle.

## 🧪 Evaluation Metrics
* Accuracy: Overall correctness of the model.

* Precision & Recall: Evaluated per weather category.

* F1 Score: Balances precision and recall.

* Cross-validation: Ensures robust model evaluation.

## ✅ Conclusion
The decision tree model provides a transparent, interpretable, and efficient solution for weather prediction. While it performs well on moderate datasets, integrating ensemble techniques like Random Forest can enhance accuracy and generalizability. This model is ideal for real-time forecasting in resource-constrained environments such as mobile or embedded systems.

## 🔮 Future Scope
Incorporate ensemble methods (e.g., Random Forest, Gradient Boosting)

* Expand feature set (e.g., humidity, cloud cover)

* Deploy the model as an API/web app

* Use real-time streaming weather data for predictions

# Website: https://weatherpredictionwebsite.streamlit.app/
