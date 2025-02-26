# Diabetes Risk Prediction

## Overview
This project focuses on developing a machine learning model to predict diabetes risk in individuals based on various health indicators and symptoms. Diabetes is a chronic metabolic disorder characterized by elevated blood glucose levels, affecting millions worldwide. Early detection and risk assessment are crucial for timely intervention and prevention of complications.

## Problem Statement
Diabetes mellitus is a growing global health concern with significant morbidity and mortality. Many cases remain undiagnosed until complications develop, highlighting the need for effective screening tools. This project addresses this challenge by:

1. Analyzing a comprehensive dataset of patient symptoms and characteristics
2. Developing predictive models to identify individuals at high risk of diabetes
3. Determining the most significant predictors of diabetes risk
4. Creating an accurate classification system that could potentially be used in clinical settings

## Dataset Description
The dataset contains records of patients with various attributes related to diabetes risk factors. Each record includes:

### Demographics
- **Age**: Range from 20 to 65 years
- **Gender**: Male (1) or Female (2)

### Symptoms and Risk Factors
- **Polyuria** (1: Yes, 2: No): Excessive urination, often a key early symptom
- **Polydipsia** (1: Yes, 2: No): Excessive thirst, frequently accompanying polyuria
- **Sudden Weight Loss** (1: Yes, 2: No): Unexplained weight reduction
- **Weakness** (1: Yes, 2: No): General physical weakness or fatigue
- **Polyphagia** (1: Yes, 2: No): Excessive hunger despite adequate food intake
- **Genital Thrush** (1: Yes, 2: No): Fungal infection in genital area
- **Visual Blurring** (1: Yes, 2: No): Impaired vision or focus difficulties
- **Itching** (1: Yes, 2: No): Persistent skin itching
- **Irritability** (1: Yes, 2: No): Mood changes and increased irritability
- **Delayed Healing** (1: Yes, 2: No): Slow wound healing
- **Partial Paresis** (1: Yes, 2: No): Partial loss of voluntary movement
- **Muscle Stiffness** (1: Yes, 2: No): Rigidity or reduced flexibility in muscles
- **Alopecia** (1: Yes, 2: No): Hair loss
- **Obesity** (1: Yes, 2: No): Excessive body weight

### Target Variable
- **Class** (Positive, Negative): Diabetes diagnosis status

## Methodology

### Data Preprocessing
- **Data Cleaning**: Handling missing values, removing duplicates, and correcting inconsistencies
- **Data Transformation**: Converting categorical variables to numerical format
- **Feature Scaling**: Normalizing or standardizing numerical features
- **Data Splitting**: Dividing the dataset into training (70%), validation (15%), and test (15%) sets

### Exploratory Data Analysis (EDA)
- **Univariate Analysis**: Distribution of individual features
- **Bivariate Analysis**: Relationship between features and target variable
- **Correlation Analysis**: Identifying relationships between different features
- **Statistical Tests**: Chi-square tests for categorical variables, t-tests for numerical variables
- **Visualization**: Histograms, box plots, correlation matrices, and pair plots

### Feature Engineering and Selection
- **Feature Importance**: Using statistical methods and model-based approaches
- **Dimensionality Reduction**: PCA or t-SNE if necessary
- **Feature Creation**: Developing composite features from existing attributes
- **Feature Selection**: Recursive feature elimination, L1 regularization, or tree-based methods

### Model Development
The project implements and compares several classification algorithms:

1. **Logistic Regression**: Baseline model with regularization
2. **Decision Trees**: With pruning to prevent overfitting
3. **Random Forest**: Ensemble method with bagging
4. **Gradient Boosting**: XGBoost and LightGBM implementations
5. **Support Vector Machines**: With various kernel functions
6. **Neural Networks**: Multi-layer perceptron with appropriate architecture
7. **K-Nearest Neighbors**: Distance-based classification

### Hyperparameter Tuning
- **Grid Search**: Exhaustive search over specified parameter values
- **Random Search**: Sampling from parameter distributions
- **Bayesian Optimization**: For more efficient parameter search
- **Cross-Validation**: k-fold cross-validation to ensure model robustness

### Model Evaluation
- **Accuracy**: Overall correctness of predictions
- **Precision and Recall**: Measuring false positives and false negatives
- **F1 Score**: Harmonic mean of precision and recall
- **ROC Curve and AUC**: Evaluating model discrimination ability
- **Confusion Matrix**: Detailed breakdown of prediction results
- **Classification Report**: Comprehensive performance metrics

## Results and Findings

### Key Insights from EDA
- Polyuria and polydipsia show strong correlation with diabetes diagnosis
- Age distribution reveals higher risk in middle-aged and older individuals
- Gender differences in symptom presentation and diabetes risk
- Clustering of symptoms in positive cases suggests syndrome patterns

### Model Performance
- **Best Performing Model**: Gradient Boosting (XGBoost)
  - Accuracy: 95.2%
  - Precision: 94.8%
  - Recall: 96.1%
  - F1 Score: 95.4%
  - AUC: 0.978

- **Feature Importance**:
  1. Polyuria
  2. Polydipsia
  3. Age
  4. Sudden weight loss
  5. Weakness

### Clinical Implications
- The model identifies high-risk individuals with high accuracy
- Early symptoms like polyuria and polydipsia are crucial warning signs
- Combination of multiple symptoms significantly increases risk
- Age remains an important non-modifiable risk factor

## Implementation Details

### Technologies Used
- **Python 3.8+**: Primary programming language
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, TensorFlow, XGBoost
- **Data Visualization**: Matplotlib, Seaborn, Plotly
- **Statistical Analysis**: SciPy, StatsModels
- **Development Environment**: Jupyter Notebooks, VS Code
- **Version Control**: Git, GitHub
- **Documentation**: Markdown, Sphinx

## Installation and Usage

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Instructions
# Clone the repository
git clone https://github.com/SaurabhJalendra/Diabetes-Risk-Prediction.git

# Navigate to project directory
cd Diabetes-Risk-Prediction

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebook server
jupyter notebook

## Future Work
- **Model Improvement**: Incorporate ensemble techniques and deep learning approaches
- **External Validation**: Test the model on diverse populations and clinical settings
- **Feature Expansion**: Include laboratory values and additional risk factors
- **Longitudinal Analysis**: Track risk progression over time
- **Deployment**: Develop a web application or API for clinical use
- **Explainability**: Enhance model interpretability for healthcare professionals
- **Personalization**: Develop individualized risk profiles and recommendations

## Challenges and Limitations
- **Data Imbalance**: Addressing class imbalance in the dataset
- **Feature Correlation**: Managing multicollinearity between symptoms
- **Generalizability**: Ensuring model works across different populations
- **Binary Classification**: Expanding to multi-class risk stratification
- **Self-reported Symptoms**: Potential subjectivity in symptom reporting
- **Missing Biochemical Markers**: Lack of laboratory values like blood glucose

## Contributing
Contributions to this project are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows the project's coding standards and includes appropriate tests.

## Acknowledgments
- Dataset provided as part of academic coursework
- Inspiration from WHO and ADA diabetes risk assessment tools
- Thanks to all contributors and reviewers
- Special appreciation to the open-source machine learning community

## Contact
For questions or feedback, please open an issue on this repository or contact me on saurabhjalendra@gmail.com.

---

© 2024 Diabetes Risk Prediction Project
