import math

"""
import numpy as np
import pandas as pd
import os
import sys


# --- Configuration ---
excel_file_name = 'Polynomial_Functions.xlsx' # Excel file with training data
feature_columns = ['x', 'y', 'z']             # Input feature column names in Excel
target_column = 'F(x, y, z)'                  # Target column name in Excel
weights_filename = 'regression_weights.npy'   # File to store the calculated weights
num_features = 20                             # 1 bias + 19 weights (w1 to w19)


def create_feature_vector(x, y, z):
    # Creates a feature vector (1D numpy array) based on the polynomial function F(x,y,z).
    # Includes 20 features: 1 (bias), x, x^2, x^3, y, y^2, y^3, z, z^2, z^3,
    #                  xy, x^2*y, x*y^2, x*z, x^2*z, x*z^2, y*z, y^2*z, y*z^2, x*y*z

    return [
        1.0,  # Bias term (for b) - ensure it's float for consistency
        float(x), float(x ** 2), float(x ** 3),
        float(y), float(y ** 2), float(y ** 3),
        float(z), float(z ** 2), float(z ** 3),
        float(x * y), float(x ** 2 * y), float(x * y ** 2),
        float(x * z), float(x ** 2 * z), float(x * z ** 2),
        float(y * z), float(y ** 2 * z), float(y * z ** 2),
        float(x * y * z)
    ]

# --- Training Phase (only runs if weights file doesn't exist) ---
beta = None
if not os.path.exists(weights_filename):
    # print(f"Weights file '{weights_filename}' not found. Training model from '{excel_file_name}'...", file=sys.stderr)
    # --- 1. Read training data from Excel file ---
    try:
        train_data = pd.read_excel(excel_file_name)

        # Extract input features (X)
        X_train_values = train_data[feature_columns].values

        # Extract target values (Y) and reshape to column vector
        Y_train = train_data[target_column].values.reshape(-1, 1)
        # print(f"Successfully read {len(Y_train)} samples from Excel.", file=sys.stderr)

    except FileNotFoundError:
        print(f"Error: File '{excel_file_name}' not found. Cannot train.", file=sys.stderr)
        exit(1)
    except KeyError:
        print(f"Error: Expected columns ({', '.join(feature_columns + [target_column])}) not found in '{excel_file_name}'.", file=sys.stderr)
        print(f"Available columns: {list(train_data.columns)}", file=sys.stderr)
        exit(1)
    except ImportError:
         print("Error: 'openpyxl' package is required to read .xlsx files.", file=sys.stderr)
         print("Please install it using 'pip install openpyxl'", file=sys.stderr)
         exit(1)
    except Exception as e:
        print(f"An error occurred while reading the Excel file: {e}", file=sys.stderr)
        exit(1)

    # --- 2. Build the Design Matrix (X_design) ---
    num_samples = X_train_values.shape[0]
    if num_samples < num_features:
        print(f"Error: Not enough training samples ({num_samples}) for the number of features ({num_features}).", file=sys.stderr)
        exit(1)

    # print("Building the design matrix...", file=sys.stderr)
    X_design = np.zeros((num_samples, num_features))
    for i in range(num_samples):
        xi, yi, zi = X_train_values[i]
        X_design[i, :] = create_feature_vector(xi, yi, zi)

    # --- 4. Calculate Coefficients (beta) using the Normal Equation ---
    # This is the core calculation part that solves for the optimal weights
    # analytically, equivalent to finding where the cost function gradient is zero.
    # beta = (X^T * X)^(-1) * X^T * Y
    # >>> Start of Core Weight Calculation <<<
    # print("Calculating regression coefficients using Normal Equation...", file=sys.stderr)
    try:
        XT = X_design.T
        XTX = XT @ X_design
        XTX_inv = np.linalg.inv(XTX) # Attempt to calculate the inverse
        XTY = XT @ Y_train
        beta = XTX_inv @ XTY       # Calculate weights

    except np.linalg.LinAlgError:
        # If matrix is singular, use pseudo-inverse
        print("Warning: Matrix X^T*X is singular. Using pseudo-inverse (pinv).", file=sys.stderr)
        try:
            # Need XT and XTY again if the first attempt failed mid-way
            XT = X_design.T
            XTX = XT @ X_design
            XTY = XT @ Y_train
            XTX_pinv = np.linalg.pinv(XTX) # Calculate pseudo-inverse
            beta = XTX_pinv @ XTY          # Calculate weights using pinv
        except np.linalg.LinAlgError:
             print("Error: Failed to compute both inverse and pseudo-inverse.", file=sys.stderr)
             exit(1)
    # >>> End of Core Weight Calculation <<<

    # --- Save the calculated weights ---
    try:
        np.save(weights_filename, beta)
        # print(f"Calculated weights saved to '{weights_filename}'.", file=sys.stderr)
    except Exception as e:
        print(f"Error: Could not save weights to '{weights_filename}'. Error: {e}", file=sys.stderr)

else:
    # --- Load existing weights ---
    print(f"Loading weights from existing file: '{weights_filename}'", file=sys.stderr)
    try:
        beta = np.load(weights_filename)
        if beta.shape[0] != num_features:
             print(f"Error: Loaded weights from '{weights_filename}' have incorrect shape ({beta.shape}). Expected ({num_features}, 1) or ({num_features},).", file=sys.stderr)
             exit(1)
    except Exception as e:
        print(f"Error: Could not load weights from '{weights_filename}'. Error: {e}", file=sys.stderr)
        exit(1)

# --- Check if beta was successfully loaded or calculated ---
if beta is None:
    print("Error: Weights (beta) could not be determined.", file=sys.stderr)
    exit(1)

"""

def F(x, y, z):
    b = 0.0
    w = [
        0.50,
        1.00,
        0.25,
        0.60,
        0.20,
        0.40,
        0.25,
        0.11,
        0.23,
        0.31,
        0.76,
        0.14,
        0.15,
        0.62,
        0.89,
        0.30,
        0.10,
        0.60,
        0.21
    ]
    return  (w[0]*x + w[1]*x**2 + w[2]*x**3 +
            w[3]*y + w[4]*y**2 + w[5]*y**3 +
            w[6]*z + w[7]*z**2 + w[8]*z**3 +
            w[9]*x*y + w[10]*x**2*y + w[11]*x*y**2 +
            w[12]*x*z + w[13]*x**2*z + w[14]*x*z**2 +
            w[15]*y*z + w[16]*y**2*z + w[17]*y*z**2 +
            w[18]*x*y*z + b)

def ceil_to_precision(value, precision):
    factor = 10 ** precision
    return math.ceil(value * factor) / factor

x = float(input())
y = float(input())
z = float(input())

print(ceil_to_precision(F(x, y, z),2))
# print(beta)