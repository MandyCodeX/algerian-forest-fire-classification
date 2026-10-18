# 🔥 Algerian Forest Fires: Prediction & Classification Analysis

[![GitHub license](https://img.shields.io/github/license/MandyCodeX/algerian-forest-fire-classification)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/MandyCodeX/algerian-forest-fire-classification/pulls)

A comprehensive data science and machine learning repository analyzing Algerian forest fire occurrences and predicting the **Fire Weather Index (FWI)**. This project includes complete end-to-end exploratory data analysis (EDA), feature engineering, multicollinearity elimination, and regression modeling with 5-fold cross-validation.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset Summary](#-dataset-summary)
- [Dataset Attributes](#-dataset-attributes)
- [Exploratory Data Analysis (EDA) Insights](#-exploratory-data-analysis-eda-insights)
- [Feature Engineering & Multicollinearity Handling](#-feature-engineering--multicollinearity-handling)
- [Model Training & Real Results](#-model-training--real-results)
- [Repository Structure](#-repository-structure)
- [Installation & Usage](#-installation--usage)
- [Technologies Used](#-technologies-used)
- [License](#-license)
- [Author](#-author)

---

## 📖 Project Overview

Wildfires cause severe ecological, economic, and environmental destruction. Predicting wildfire hazard severity enables authorities to deploy preventive measures and allocate firefighting resources effectively.

This project analyzes meteorological observations and indices from the **Canadian Forest Fire Weather Index (FWI) System** across two distinct regions in Algeria:
1. **Bejaia Region** (Northeast Algeria)
2. **Sidi-Bel Abbes Region** (Northwest Algeria)

The project addresses two core analytical facets:
- **Classification Perspective:** Analyzing patterns differentiating fire outbreaks (`fire`) versus non-fire periods (`not fire`).
- **Regression Modeling:** Predicting the continuous **Fire Weather Index (FWI)** rating with high precision using regularized linear regression techniques (Ridge, Lasso, ElasticNet) with cross-validation.

---

## 📊 Dataset Summary

The dataset spans daily observations from **June 2012 to September 2012**.

| Metric | Value |
| :--- | :--- |
| **Total Instances (Raw)** | 246 rows (includes region division headers) |
| **Cleaned Instances** | 243 rows (122 from Bejaia, 121 from Sidi-Bel Abbes) |
| **Total Attributes** | 15 (11 features used for modeling) |
| **Fire Classes** | **Fire**: 137 (56.38%) \| **Not Fire**: 106 (43.62%) |
| **Time Period** | June 1, 2012 – September 30, 2012 |

---

## 📋 Dataset Attributes

### Weather Observations
- **`day`**, **`month`**, **`year`**: Calendar date of observation.
- **`Temperature`**: Noon temperature in Celsius degrees (Range: $22^\circ\text{C}$ to $42^\circ\text{C}$).
- **`RH`**: Relative Humidity percentage (Range: $21\%$ to $90\%$).
- **`Ws`**: Wind speed in km/h (Range: $6$ to $29\text{ km/h}$).
- **`Rain`**: Daily total rainfall in mm (Range: $0.0$ to $16.8\text{ mm}$).

### FWI System Components
- **`FFMC` (Fine Fuel Moisture Code)**: Moisture rating of surface litter; indicates ease of ignition (Range: $28.6$ to $96.0$).
- **`DMC` (Duff Moisture Code)**: Moisture rating of decomposing organic layers (Range: $0.7$ to $65.9$).
- **`DC` (Drought Code)**: Moisture rating of deep, compact organic layers (Range: $6.9$ to $220.4$).
- **`ISI` (Initial Spread Index)**: Expected rate of fire spread combining FFMC and wind (Range: $0.0$ to $56.1$).
- **`BUI` (Buildup Index)**: Total fuel available for combustion, combining DMC and DC (Range: $1.1$ to $68.0$).
- **`FWI` (Fire Weather Index)**: Numerical rating of fire intensity (Target variable for regression, Range: $0.0$ to $31.1$).

### Metadata & Labels
- **`Classes`**: Fire status (`1` = fire, `0` = not fire).
- **`Region`**: Geographic region indicator (`0` = Bejaia, `1` = Sidi-Bel Abbes).

---

## 🔍 Exploratory Data Analysis (EDA) Insights

The exploratory phase in [`EDA_and_FE.ipynb`](EDA_and_FE.ipynb) revealed key real-world patterns:

1. **Seasonal Concentration:**
   - Forest fires peaked during **July and August** across both Bejaia and Sidi-Bel Abbes.
   - Fire frequency dropped drastically in **September** due to increased relative humidity and precipitation.
2. **Weather Drivers:**
   - Wildfires showed strong positive correlation with higher temperatures ($> 30^\circ\text{C}$) and lower relative humidity ($< 60\%$).
   - Fires rarely occurred on days with measurable rainfall ($> 1\text{ mm}$).
3. **Regional Dynamics:**
   - Both regions exhibited similar temporal behavior, with Sidi-Bel Abbes experiencing higher maximum temperatures and slightly drier conditions.

---

## ⚙️ Feature Engineering & Multicollinearity Handling

During correlation analysis, strong multicollinearity was identified among several FWI indices:
- $\text{corr}(\text{DMC}, \text{DC}) = 0.869$ ($> 0.85$)
- $\text{corr}(\text{DMC}, \text{BUI}) = 0.983$ ($> 0.85$)
- $\text{corr}(\text{DC}, \text{BUI}) = 0.942$ ($> 0.85$)

### Preprocessing Actions:
- **Redundant Feature Removal:** Using a correlation threshold of $0.85$, **`DC`** and **`BUI`** were dropped to prevent variance inflation and coefficient instability.
- **Categorical Encoding:** `Classes` converted to binary labels (`1` for fire, `0` for not fire).
- **Date Columns:** `day`, `month`, and `year` dropped for model generalization.
- **Train-Test Split:** 75% training (182 samples) and 25% test (61 samples) with `random_state=42`.
- **Feature Scaling:** `StandardScaler` applied to prevent scale dominance among variables (e.g., Temperature vs. FFMC).

---

## 🏆 Model Training & Real Results

The target variable for regression is **`FWI` (Fire Weather Index)**. Models were evaluated using Mean Absolute Error (**MAE**) and the Coefficient of Determination (**$R^2$ Score**).

### Real Evaluation Metrics (From [`Model_Training.ipynb`](Model_Training.ipynb))

| Model | Hyperparameters / CV | MAE | $R^2$ Score | Performance Remarks |
| :--- | :--- | :---: | :---: | :--- |
| **Linear Regression** | Default (OLS) | **0.5468** | **0.9848** (98.48%) | Strongest baseline fit |
| **Ridge Regression** | $\alpha = 1.0$ | 0.5642 | 0.9843 (98.43%) | L2 penalty controls weight explosion |
| **RidgeCV** | 5-Fold CV ($\text{best } \alpha = 1.0$) | 0.5642 | 0.9843 (98.43%) | Consistent and robust across folds |
| **Lasso Regression** | $\alpha = 1.0$ | 1.1332 | 0.9492 (94.92%) | High penalty over-shrinks coefficients |
| **LassoCV** | 5-Fold CV ($\text{best } \alpha \approx 0.0658$) | **0.6359** | **0.9814** (98.14%) | Cross-validation optimizes sparsity |
| **ElasticNet** | $\alpha = 1.0, \text{l1\_ratio} = 0.5$ | 1.8822 | 0.8753 (87.53%) | Default alpha excessively dampens weights |
| **ElasticNetCV** | 5-Fold CV ($\text{best } \alpha \approx 0.0431$) | **0.6576** | **0.9814** (98.14%) | Recovers high accuracy with tuned $\alpha$ |

### 📈 Key Model Findings:
- **Ridge Regression ($\alpha=1.0$)** and **Linear Regression** achieved the best predictive power with an $R^2$ score over **98.4%** and MAE around **0.55**.
- Hyperparameter tuning via 5-Fold CV significantly improved **Lasso** ($R^2$ rose from $94.92\%$ to $98.14\%$) and **ElasticNet** ($R^2$ rose from $87.53\%$ to $98.14\%$).

---

## 📁 Repository Structure

```text
algerian-forest-fire-classification/
├── Algerian_forest_fires_cleaned_dataset.csv     # Cleaned dataset (243 rows, 15 features)
├── Algerian_forest_fires_dataset_UPDATE.csv      # Raw original dataset from UCI
├── EDA_and_FE.ipynb                              # Exploratory Data Analysis & Feature Engineering
├── Model_Training.ipynb                          # Model training, cross-validation & evaluation
├── LICENSE                                       # MIT License
├── README.md                                     # Project documentation
└── .gitignore                                    # Git ignore rules for notebooks & cache
```

---

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/MandyCodeX/algerian-forest-fire-classification.git
cd algerian-forest-fire-classification
```

### 2. Set Up a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 4. Launch Jupyter Notebooks
```bash
jupyter notebook
```
- Open `EDA_and_FE.ipynb` to view data preprocessing, visualizations, and correlations.
- Open `Model_Training.ipynb` to inspect model training, cross-validation, and metrics.

---

## 🛠️ Technologies Used

- **Language:** Python 3
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn` (LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV, StandardScaler)
- **Environment:** Jupyter Notebook

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Mandeep Kumar**  
- GitHub: [@MandyCodeX](https://github.com/MandyCodeX)
- Email: immandy108@gmail.com
