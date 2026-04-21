# House Price Prediction - ML Analysis Platform

A comprehensive **Machine Learning analysis tool** that loads any CSV dataset from the internet, performs exploratory data analysis, data cleaning, and builds a predictive Linear Regression model with professional visualizations.

## 🎯 Project Overview

This project demonstrates a **complete data science pipeline** from data loading through model evaluation. It's designed to work with **any numerical dataset** from the internet—not just house prices. Simply provide a dataset URL and specify your target column, and the system handles everything automatically.

### ✨ Key Features

✅ **Load Any Dataset** - Works with any CSV file from the internet (Kaggle, GitHub, UCI ML Repository, etc.)  
✅ **Automatic EDA** - Exploratory Data Analysis with shape, statistics, and missing value detection  
✅ **Intelligent Data Cleaning** - Automatically handles missing values and selects numeric features  
✅ **Multiple Features** - Uses all relevant features (not just single variable prediction)  
✅ **Train/Test Split** - 80/20 split ensures model generalization testing  
✅ **Comprehensive Evaluation** - R² Score, MAE, and RMSE metrics  
✅ **Professional Dashboard** - 4-panel visualization with dark theme analytics  
✅ **Feature Analysis** - Shows which features have the most impact on predictions

## 📊 What the Program Does

### Step-by-Step Workflow:

1. **Load Dataset** - Downloads CSV from any internet URL
2. **Explore Data** - Shows dataset shape, columns, data types, and missing values
3. **Clean Data** - Fills missing values and keeps only numeric columns
4. **Prepare Features** - Auto-selects features or uses your custom list
5. **Split Data** - Divides into 80% training, 20% testing
6. **Train Model** - Builds Linear Regression on training data
7. **Evaluate Model** - Calculates R², MAE, and RMSE on test data
8. **Make Predictions** - Predicts price for a typical house
9. **Generate Dashboard** - Creates `dashboard.png` with 4 analysis charts

## 🚀 Installation

### Requirements

- Python 3.7+
- pip (Python package manager)

### Setup

```bash
# Clone or download this project
cd House-Price-Prediction

# Install required packages
pip install pandas numpy scikit-learn matplotlib
```

### Verify Installation

```bash
python main.py
```

## 📖 Usage

### Basic Usage (Default Dataset)

```bash
python main.py
```

The default URL points to a house price dataset. The script will:

- Load the data
- Perform full analysis
- Generate `dashboard.png`
- Print detailed results to console

### Use Your Own Dataset

Edit **lines 33-35** in `main.py`:

```python
# Change this URL to your dataset
DATASET_URL = "https://your-dataset-url/data.csv"

# Change target column name (what you want to predict)
TARGET_COLUMN = "price"  # or "salary", "temperature", etc.

# Optional: Specify which columns to use as features
FEATURE_COLUMNS = None   # Auto-select, or ["col1", "col2", "col3"]
```

### Example Dataset URLs

**Titanic Dataset** (Kaggle):

```python
DATASET_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
TARGET_COLUMN = "Survived"
FEATURE_COLUMNS = ["Age", "Fare", "Pclass"]
```

**California Housing** (UCI ML):

```python
# First download from: https://archive.ics.uci.edu/ml/datasets/Housing
# Upload to GitHub and get raw URL
TARGET_COLUMN = "MedHouseVal"
```

**Auto Insurance** (Kaggle):

```python
DATASET_URL = "https://raw.githubusercontent.com/your-username/datasets/main/insurance.csv"
TARGET_COLUMN = "charges"
```

## 📈 Output

### Console Output

```
============================================================
   HOUSE PRICE PREDICTION — UPGRADED VERSION
============================================================

📥 Loading dataset from:
   https://raw.githubusercontent.com/dhruv-bamal/House-Price-Prediction/main/data.csv

✅ Dataset loaded successfully!
...
   R² Score:  0.8543  ✅ Excellent
   MAE:       5234.45   → avg prediction error
   RMSE:      6789.12   → penalized prediction error
...
```

### Visual Output: `dashboard.png`

The script generates a **4-panel professional dashboard** showing:

| Panel                     | Description                                                  |
| ------------------------- | ------------------------------------------------------------ |
| **① Actual vs Predicted** | Scatter plot showing model accuracy (higher R² = better fit) |
| **② Feature Impact**      | Bar chart showing which features influence price most        |
| **③ Price Distribution**  | Histogram showing price range and central tendency           |
| **④ Residuals**           | Error distribution (should be centered at 0)                 |

## 🛠 Technologies Used

| Technology       | Purpose                                       |
| ---------------- | --------------------------------------------- |
| **Python 3**     | Programming language                          |
| **Pandas**       | Data loading, cleaning, and manipulation      |
| **NumPy**        | Numerical computations                        |
| **Scikit-Learn** | Machine Learning (Linear Regression, metrics) |
| **Matplotlib**   | Data visualization and dashboard generation   |

## 📊 Understanding the Metrics

### R² Score (Coefficient of Determination)

- **Range**: 0.0 to 1.0
- **What it means**: Percentage of price variation explained by the model
- **Good value**: > 0.75
- **Interpretation**: R² = 0.85 means the model explains 85% of price variation

### MAE (Mean Absolute Error)

- **What it means**: Average prediction error in actual units
- **Example**: MAE = ₹5,234 means predictions are off by ₹5,234 on average
- **Good value**: Lower is better

### RMSE (Root Mean Square Error)

- **What it means**: Like MAE but penalizes larger errors more heavily
- **Good value**: Lower is better, similar to MAE
- **When to use**: When large errors are worse than small errors

## 🔄 How to Get CSV URLs from Popular Sources

### GitHub (Recommended for beginners)

1. Find CSV file in GitHub repo
2. Click the file
3. Click **"Raw"** button
4. Copy URL from address bar

### Kaggle

1. Download dataset (requires account)
2. Upload to GitHub (create a repo)
3. Use GitHub raw URL

### Google Dataset Search

1. Visit https://datasetsearch.research.google.com
2. Search for your dataset
3. Look for direct CSV download links

## 💡 Tips & Tricks

**Tip 1**: For GitHub URLs, always use the **Raw** button to get the direct CSV link

```
✅ Correct:  https://raw.githubusercontent.com/...
❌ Wrong:    https://github.com/...
```

**Tip 2**: Test your URL in browser first

```
If you can see CSV text in your browser, the URL is correct
```

**Tip 3**: Handle categorical columns

```python
# The script automatically keeps only numeric columns
# So non-numeric columns are ignored
```

**Tip 4**: Auto feature selection

```python
# Leave FEATURE_COLUMNS = None to use all numeric columns except target
# Or specify exactly which columns: FEATURE_COLUMNS = ["col1", "col2"]
```

## 🎓 Learning Outcomes

After using this project, you'll understand:

✅ How to load data from the internet into Python  
✅ How to perform Exploratory Data Analysis (EDA)  
✅ How to clean real-world messy data  
✅ How to train a Machine Learning model  
✅ How to evaluate model performance with multiple metrics  
✅ How to visualize data and results professionally  
✅ How Linear Regression works and when to use it

## 🚀 Future Enhancements

- [ ] Support for categorical features (one-hot encoding)
- [ ] Multiple model algorithms (Decision Trees, Random Forest, etc.)
- [ ] Cross-validation for better model evaluation
- [ ] Hyperparameter tuning
- [ ] Model persistence (save/load trained models)
- [ ] Web UI for easier dataset configuration
- [ ] Support for time-series data
- [ ] Outlier detection and handling

## 📝 Project Structure

```
House-Price-Prediction/
├── main.py           # Main analysis script
├── README.md         # This file
├── report.txt        # Academic project report
├── data.csv          # Dataset
└── dashboard.png     # (Generated) Analysis dashboard visualization
```

## 🐛 Troubleshooting

### Error: "Could not load dataset from URL"

- ✅ Check if URL ends with `.csv`
- ✅ Verify URL works in your browser
- ✅ For GitHub, use the **Raw** button URL
- ✅ Check internet connection

### Error: "Target column 'price' not found!"

- ✅ Check the exact column name (case-sensitive)
- ✅ Print all available columns first:

```python
FEATURE_COLUMNS = None  # Auto mode
# Then check console output for "Available columns:"
```

### Model shows poor R² score

- ✅ The features may not be related to the target
- ✅ Try with a different dataset
- ✅ Check for data quality issues

## 👥 Authors

**Submitted By:**

- Dhruv Bamal (RA2311003030447)
- Prashant (RA2311003030441)

**Institution:** SRM Institute of Science and Technology

## 📚 References

- [Scikit-Learn Linear Regression Documentation](https://scikit-learn.org/stable/modules/linear_model.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Visualization Guide](https://matplotlib.org/stable/tutorials/index.html)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)

## 📄 License

This is an academic project. Feel free to use and modify for educational purposes.

---

**Happy Data Science Learning!** 🎉

For questions or suggestions, refer to the project report or modify the script configuration at the top of `main.py`.
