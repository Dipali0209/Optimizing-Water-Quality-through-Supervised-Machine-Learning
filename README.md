# Optimizing Water Quality through Supervised Machine Learning

### 📝 **Project Summary**
Ensuring water quality is critical for protecting **public health** and **environmental sustainability**. This project evaluates the effectiveness of **Supervised Machine Learning** models in predicting the **Water Quality Index (WQI)** and classifying **Water Quality** into five categories. The study employs advanced regression and classification models to offer a reliable and efficient approach for **environmental monitoring**.

### 🔍 **Problem Statement**
Traditional water quality assessment methods are often resource-intensive and slow, lacking the precision needed for real-time monitoring. This project seeks to improve the **accuracy** and **efficiency** of water quality predictions by using **Machine Learning** models to automate both **WQI prediction** and **Water Quality Classification (WQC)**.

### 📊 **Dataset**
The dataset used for this study is sourced from [Kaggle](https://www.kaggle.com/), containing 3,276 samples with 10 major water quality parameters:
- **pH**
- **Hardness**
- **Solids**
- **Chloramines**
- **Sulfate**
- **Conductivity**
- **Organic Carbon**
- **Trihalomethanes**
- **Turbidity**

### 🚀 **Project Approach**

#### 1. Data Preprocessing
- **Data Imputation**: Handled missing values using **kNN imputation** from Scikit-learn, which substitutes missing data based on similar nearby observations.
- **Normalization & Standardization**: Applied **Standard Scaler** to standardize features by subtracting the mean and scaling to unit variance, ensuring models perform optimally.

#### 2. Models Used
- **Regression Models** (for WQI Prediction):
  - Random Forest Regressor (RFR)
  - Support Vector Regressor (SVR)
  - Decision Tree Regressor (DTR)
  - Gradient Boosting Regressor (GBR)
  - Multi-Layer Perceptron (MLP)
  
- **Classification Models** (for WQC):
  - Decision Tree Classifier (DTC)
  - Random Forest Classifier (RFC)
  - Gradient Boosting Classifier (GBC)
  - Support Vector Machine (SVM)
  - Multi-Layer Perceptron (MLP)

### 🔑 **Key Findings**

#### WQI Prediction (Regression Models)
- The **Random Forest Regressor** achieved the highest accuracy with an **R² score of 0.996**, outperforming all other models.
- **Decision Tree Regressor** and **GBR** also showed strong predictive power with **R² scores above 0.98**.
  
#### WQC (Classification Models)
- The **Decision Tree Classifier** led with an accuracy of **96.34%**, closely followed by **GBR (96.04%)** and **Random Forest (94.82%)**.
- The models demonstrated high **Precision**, **Recall**, and **F1 Scores**, making them effective for classifying water quality across multiple categories (Excellent, Good, Poor, Very Poor, Unsafe).

### 📊 **Evaluation Metrics**

#### Regression Model Performance (WQI Prediction)

| Model            | MAE         | MSE        | RMSE       | R²       |
|------------------|-------------|------------|------------|----------|
| Random Forest    | 0.723       | 1.611      | 1.269      | 0.996    |
| SVR              | 10.893      | 215.225    | 14.670     | 0.549    |
| Decision Tree    | 1.154       | 3.406      | 1.845      | 0.993    |
| MLP              | 6.811       | 91.919     | 9.587      | 0.808    |
| GBR              | 1.818       | 6.531      | 2.556      | 0.986    |

#### Classification Model Performance (WQC)

| Classifier       | Accuracy    | Precision  | Recall     | F1 Score  |
|------------------|-------------|------------|------------|-----------|
| Decision Tree    | 96.34%      | 0.963      | 0.963      | 0.963     |
| Random Forest    | 94.82%      | 0.949      | 0.948      | 0.948     |
| Gradient Boosting| 96.04%      | 0.960      | 0.960      | 0.960     |
| MLP              | 89.63%      | 0.898      | 0.896      | 0.894     |
| SVM              | 71.34%      | 0.716      | 0.713      | 0.711     |

### 📈 **Results and Visualizations**

- **Figure 1**: Comparative performance of regression models for WQI prediction.
<p align="center">
  <img src="path_to_image/figure1a.png" alt="Figure 1a" width="400" />
  <img src="path_to_image/figure1b.png" alt="Figure 1b" width="400" />
</p>

- **Figure 2**: Receiver Operating Characteristic (ROC) Curve and Precision-Recall Curve for **Decision Tree Classifier**.

<p align="center">
  <img src="path_to_image/figure2a.png" alt="Figure 2a" width="400" />
  <img src="path_to_image/figure2b.png" alt="Figure 2b" width="400" />
</p>


### 🔧 **Tools and Technologies**
- **Languages**: Python
- **Libraries**: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
- **Imputation**: kNN Imputer
- **Scaling**: StandardScaler

### 🌍 **Impact**
This project highlights the potential of **Machine Learning** in automating water quality monitoring and analysis. The study provides a robust solution for **real-time environmental monitoring**, allowing authorities to quickly identify water quality issues and make data-driven decisions to protect public health.

### 📈 **Future Work**
- Incorporating additional datasets for real-time water quality monitoring.
- Exploring **deep learning** methods for further improvement in prediction accuracy.
- Developing a web-based tool for live water quality tracking.

### 🤝 **Connect with Me**
Feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/your-linkedin-url/) to discuss this project and more!

---

