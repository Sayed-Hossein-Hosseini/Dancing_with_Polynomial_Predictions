# Dancing with Polynomial Predictions 🕺💃

## Overview 🎯

This project focuses on predicting the output of a three-variable polynomial function, `F(x, y, z)`. The primary goal is to model this function using two distinct regression approaches and compare their accuracy:

1.  **Analytical Method (Normal Equation):** 💯 This method aims to find the optimal weights (polynomial coefficients) directly using matrix operations. Theoretically, if the model perfectly matches the data and there's no noise, this method can achieve very high accuracy. The *final prediction* using pre-defined weights derived from this approach is implemented in the Python script (`.py`).
2.  **Gradient Descent Method:** 📉 This is an iterative approach that gradually adjusts the weights to minimize the Mean Squared Error (MSE) between the predicted and actual values. This method is implemented in the Jupyter Notebook (`.ipynb`) and includes data preprocessing steps (like outlier removal and normalization) and feature engineering.

The target polynomial function is defined as follows (with specific weights hardcoded in the `.py` file):
`F(x, y, z) = w[0]*x + w[1]*x^2 + ... + w[18]*x*y*z + b`

## Methods Used 🛠️

### 1. Analytical Method

*   **File:** `Dancing_with_Polynomial_Predictions.py`
*   **Description:** This Python script implements the `F(x, y, z)` function using pre-defined (hardcoded) weights. These weights *represent* the result obtained from analytically solving the Normal Equation (`beta = (X^T * X)^(-1) * X^T * Y`) on *ideal* training data. (The code for actually calculating these weights from the Excel file is commented out in the `.py` file).
*   **Accuracy:** ✅ Due to using the exact weights and direct implementation of the target function, the output of this script (for given inputs) perfectly matches the true function value, effectively simulating 100% accuracy.

### 2. Gradient Descent Method

*   **File:** `Dancing_with_Polynomial_Predictions.ipynb`
*   **Description:** This Jupyter Notebook implements the following steps:
    *   📥 Loading data from the Excel file (`Polynomial_Functions.xlsx`).
    *   **Preprocessing:** 🧹
        *   Removing outliers using the IQR method.
        *   Normalizing data (Features and Target) using Z-score standardization.
    *   **Feature Engineering:** ✨ Creating polynomial features (like x², x³, xy, x²y, etc.) from the normalized inputs x, y, z.
    *   **Model Training:** ⚙️ Implementing linear regression using the Gradient Descent algorithm to learn the weights (coefficients) on the engineered features.
    *   **Evaluation:** 📈 Calculating the MSE on training and test data and plotting the learning curve.
    *   **Prediction:** 🔮 Using the trained model to predict the output for new x, y, z inputs.
*   **Accuracy:** 🤔 As seen in the notebook, this method achieves lower accuracy compared to the ideal analytical method. This is due to the iterative and approximate nature of Gradient Descent, its dependence on hyperparameters (learning rate, epochs), and the effects of preprocessing steps. The final MSE is non-zero.

## Files in the Project 📁

*   `Dancing_with_Polynomial_Predictions.ipynb`: Jupyter Notebook containing the Gradient Descent implementation, preprocessing, feature engineering, and evaluation.
*   `Dancing_with_Polynomial_Predictions.py`: Python script containing the `F(x, y, z)` function implemented with pre-defined analytical weights for precise prediction. (Includes commented-out code for analytical training).
*   `Polynomial_Functions.xlsx`: Excel file containing the training data with columns 'x', 'y', 'z', and 'F(x, y, z)'.
*   `README.md`: This project description file.
*   `(regression_weights.npy)`: (If the commented-out section in `.py` is run) A file storing the weights calculated by the analytical method.

## How to Run ▶️

### Jupyter Notebook (Gradient Descent)

1.  Open your Jupyter environment (Jupyter Lab/Notebook) or Google Colab.
2.  Open the `Dancing_with_Polynomial_Predictions.ipynb` file.
3.  Ensure the `Polynomial_Functions.xlsx` file is in the same directory as the notebook, or update the path in the code accordingly.
4.  Run the cells in the notebook sequentially.

### Python Script (Analytical Prediction)

1.  Open a terminal or command prompt.
2.  Navigate to the directory containing the `Dancing_with_Polynomial_Predictions.py` file.
3.  Run the following command:
    ```bash
    python Dancing_with_Polynomial_Predictions.py
    ```
4.  The script will prompt you to enter values for x, y, and z. Input them and press Enter after each.
5.  The script will print the output of `F(x, y, z)` rounded to the specified precision.
    *(Note: In its current state, this script doesn't require the Excel file or the `.npy` weights file as it uses hardcoded weights).*

## Data Description 📊

The training data is located in `Polynomial_Functions.xlsx` and includes the following columns:

*   `x`: First input variable
*   `y`: Second input variable
*   `z`: Third input variable
*   `F(x, y, z)`: The target function output value

## Results and Comparison 🏆 vs. 🤔

*   **Analytical Method (in `.py`):** 💯 This method (using the exact function weights) predicts the output with extremely high accuracy (effectively 100% match with the target function) because it directly implements the known function.
*   **Gradient Descent Method (in `.ipynb`):** 📉 This method converges to an approximate model of the function. Its accuracy is lower than the analytical approach, influenced by factors like the algorithm's approximate nature, hyperparameter choices (learning rate, epochs), and the impact of outlier removal and normalization. The learning curve in the notebook shows the error decreasing but not reaching zero.

This comparison highlights that when the exact structure of the target function is known and data is noise-free, the analytical method can yield the best possible result. However, for real-world problems involving noise and uncertainty, iterative methods like Gradient Descent (with appropriate preprocessing) are more practical, even if they don't achieve absolute precision.

## Dependencies 📦

You need the following libraries to run the full project:

*   `numpy`
*   `pandas`
*   `matplotlib`
*   `seaborn`
*   `scikit-learn` (`sklearn`)
*   `gdown` (Used in the Colab notebook)
*   `math` (Used in the `.py` script)
*   `openpyxl` (Required by pandas to read `.xlsx` files)

You can install them using `pip`:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn gdown openpyxl
```

## Auther 🕺
***Sayyed Hossein Hosseini DolatAbadi***
