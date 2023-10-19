import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_file, output_file):
    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return

    df.dropna(inplace=True)

    target_column = "Concrete compressive strength(MPa, megapascals) "
    y = df[target_column]

    df.drop(columns=[target_column], inplace=True)

    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(df)

    df_normalized = pd.DataFrame(X_normalized, columns=df.columns)

    df_normalized[target_column] = y

    try:
        df_normalized.to_excel(output_file, index=False, engine='openpyxl')
        print(f"Preprocessed data saved to {output_file}")
    except Exception as e:
        print(f"Error saving data to {output_file}: {e}")

    visualize_preprocessing(df, df_normalized, feature_to_plot="Cement (component 1)(kg in a m^3 mixture)")
    visualize_preprocessing(df, df_normalized, feature_to_plot="Age (day)")

def visualize_preprocessing(df_before, df_after, feature_to_plot):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df_before[feature_to_plot], kde=True)
    plt.title(f"Distribution of {feature_to_plot} (Before Preprocessing)")

    plt.subplot(1, 2, 2)
    sns.histplot(df_after[feature_to_plot], kde=True)
    plt.title(f"Distribution of {feature_to_plot} (After Normalization)")

    plt.tight_layout()
    plt.show()

def hypothesis(X, theta):
    return np.dot(X, theta)

def compute_mse(y, y_pred):
    mse = np.mean((y - y_pred) ** 2)
    return mse

def gradient_descent_with_early_stopping(X_train, y_train, X_val, y_val, initial_theta, learning_rates, n_iterations, tolerance):
    best_theta = None
    best_learning_rate = None
    best_iteration = None
    lowest_val_error = float("inf")

    for learning_rate in learning_rates:
        theta = initial_theta.copy()
        min_val_error = float("inf")
        error_going_up = 0
        for iteration in range(n_iterations):
            gradients = 2/len(X_train) * X_train.T.dot(X_train.dot(theta) - y_train)
            theta = theta - learning_rate * gradients
            y_val_predict = X_val.dot(theta)
            val_error = mean_squared_error(y_val, y_val_predict)

            if val_error < min_val_error:
                min_val_error = val_error
                best_theta = theta
                best_iteration = iteration
                best_learning_rate = learning_rate
                error_going_up = 0
            else:
                error_going_up += 1
                if error_going_up == tolerance:
                    break 
            
            if min_val_error < lowest_val_error:
                lowest_val_error = min_val_error

    return {"theta": best_theta, "learning_rate": best_learning_rate, "iterations": best_iteration + 1}

def calculate_r_squared(y_actual, y_predicted):
    mean_y_actual = np.mean(y_actual)

    total_sum_of_squares = np.sum((y_actual - mean_y_actual) ** 2)

    residual_sum_of_squares = compute_mse(y_actual, y_predicted) * len(y_actual)

    r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)

    return r_squared

def run_linear_regression_optimized(input_features, target, learning_rates, test_size=0.1262, is_univariate=True):
    X = input_features
    y = np.ravel(target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    X_train_b = np.c_[np.ones((len(X_train), 1)), X_train]
    X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]

    if is_univariate:
        initial_theta = np.zeros(2)
    else:
        number_of_features = input_features.shape[1]
        initial_theta = np.zeros(number_of_features + 1)

    n_iterations = 1000
    tolerance = 50
    optimal_params = gradient_descent_with_early_stopping(X_train_b, y_train, X_test_b, y_test, initial_theta, learning_rates, n_iterations, tolerance)

    predictions_train = hypothesis(X_train_b, optimal_params['theta'])
    mse_train = compute_mse(y_train, predictions_train)
    r_squared_train = calculate_r_squared(y_train, predictions_train)

    predictions_test = hypothesis(X_test_b, optimal_params['theta'])
    mse_test = compute_mse(y_test, predictions_test)
    r_squared_test = calculate_r_squared(y_test, predictions_test)

    return {
        "theta": optimal_params['theta'], 
        "mse_train": mse_train, 
        "r_squared_train": r_squared_train,
        "mse_test": mse_test, 
        "r_squared_test": r_squared_test, 
        "learning_rate": optimal_params['learning_rate'], 
        "iterations": optimal_params['iterations']
    }

def run_linear_regression_unoptimized(input_features, target, learning_rates, test_size=0.1262, is_univariate=True):
    X = input_features
    y = np.ravel(target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    X_train_b = np.c_[np.ones((len(X_train), 1)), X_train]
    X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]

    if is_univariate:
        initial_theta = np.zeros(2)
    else:
        number_of_features = input_features.shape[1]
        initial_theta = np.zeros(number_of_features + 1)

    n_iterations = 1000
    tolerance = 50

    optimal_params = gradient_descent_with_early_stopping(X_train_b, y_train, X_test_b, y_test, initial_theta, [.000003], n_iterations, tolerance)

    predictions_train = hypothesis(X_train_b, optimal_params['theta'])
    mse_train = compute_mse(y_train, predictions_train)
    r_squared_train = calculate_r_squared(y_train, predictions_train)

    predictions_test = hypothesis(X_test_b, optimal_params['theta'])
    mse_test = compute_mse(y_test, predictions_test)
    r_squared_test = calculate_r_squared(y_test, predictions_test)

    return {
        "theta": optimal_params['theta'], 
        "mse_train": mse_train, 
        "r_squared_train": r_squared_train,
        "mse_test": mse_test, 
        "r_squared_test": r_squared_test, 
        "learning_rate": optimal_params['learning_rate'], 
        "iterations": optimal_params['iterations']
    }

def run_linear_regression_experiment_optimized(input_file, output_file, test_size=0.1262):
    preprocess_data(input_file, output_file)
    data = pd.read_excel(output_file)

    features = [
        "Cement (component 1)(kg in a m^3 mixture)", 
        "Blast Furnace Slag (component 2)(kg in a m^3 mixture)",
        "Fly Ash (component 3)(kg in a m^3 mixture)",
        "Water  (component 4)(kg in a m^3 mixture)",
        "Superplasticizer (component 5)(kg in a m^3 mixture)",
        "Coarse Aggregate  (component 6)(kg in a m^3 mixture)",
        "Fine Aggregate (component 7)(kg in a m^3 mixture)",
        "Age (day)"
    ]
    
    y_column = "Concrete compressive strength(MPa, megapascals) "
    y = data[y_column].values

    learning_rates = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3] 

    for feature in features:
        X_uni = data[feature].values.reshape(-1, 1)
        uni_results = run_linear_regression_optimized(X_uni, y, learning_rates, test_size, is_univariate=True)

        print(f"\nUnivariate Training results For {feature}:")
        print(f"MSE: {uni_results['mse_train']}")
        print(f"R-squared: {uni_results['r_squared_train']}")
        print(f"Univariate Testing Mean Squared Error For {feature}: {uni_results['mse_test']}")
        print(f"Univariate Testing R-squared For {feature}: {uni_results['r_squared_test']}")

        plt.figure(figsize=(10, 6))
        plt.scatter(X_uni, y, color='blue', s=10)
        y_predict = hypothesis(np.c_[np.ones((len(X_uni), 1)), X_uni], uni_results['theta'])
        plt.plot(X_uni, y_predict, color='red', linewidth=2)
        plt.xlabel(feature)
        plt.ylabel(y_column)
        plt.title(f"Fit for feature: {feature}")
        plt.show()

    X_multi = data[features].values
    multi_results = run_linear_regression_optimized(X_multi, y, learning_rates, test_size, is_univariate=False)

    print("\nMultivariate Training results:")
    print(f"MSE: {multi_results['mse_train']}")
    print(f"R-squared: {multi_results['r_squared_train']}")
    print("Multivariate Testing Mean Squared Error: ", multi_results['mse_test'])
    print("Multivariate Testing R-squared: ", multi_results['r_squared_test'])

    
def run_linear_regression_experiment_unoptimized(input_file, test_size=0.1262):
    data = pd.read_excel(input_file)

    features = [
        "Cement (component 1)(kg in a m^3 mixture)", 
        "Blast Furnace Slag (component 2)(kg in a m^3 mixture)",
        "Fly Ash (component 3)(kg in a m^3 mixture)",
        "Water  (component 4)(kg in a m^3 mixture)",
        "Superplasticizer (component 5)(kg in a m^3 mixture)",
        "Coarse Aggregate  (component 6)(kg in a m^3 mixture)",
        "Fine Aggregate (component 7)(kg in a m^3 mixture)",
        "Age (day)"
    ]
    
    y_column = "Concrete compressive strength(MPa, megapascals) "
    y = data[y_column].values

    learning_rates = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    for feature in features:
        X_uni = data[feature].values.reshape(-1, 1)  
        uni_results = run_linear_regression_unoptimized(X_uni, y, .000003, test_size, is_univariate=True)

        print(f" Univariate Training results For {feature} MSE: {uni_results['mse_train']}")
        print(f"Univariate Training results For {feature} R-squared: {uni_results['r_squared_train']}")
        print(f"Univariate Testing Mean Squared Error For {feature}: {uni_results['mse_test']}")
        print(f"Univariate Testing R-squared For {feature}: {uni_results['r_squared_test']}")

        plt.figure(figsize=(10, 6))
        plt.scatter(X_uni, y, color='blue', s=10) 
        y_predict = hypothesis(np.c_[np.ones((len(X_uni), 1)), X_uni], uni_results['theta'])
        plt.plot(X_uni, y_predict, color='red', linewidth=2)
        plt.xlabel(feature)
        plt.ylabel(y_column)
        plt.title(f"Fit for feature: {feature}")
        plt.show()

    X_multi = data[features].values
    multi_results = run_linear_regression_unoptimized(X_multi, y, .000003, test_size, is_univariate=False)

    print(f" Multivariate Training MSE: {multi_results['mse_train']}")
    print(f"Multivariate Training R-squared: {multi_results['r_squared_train']}")
    print("Multivariate Testing MSE: ", multi_results['mse_test'])
    print("Multivariate Testing R-squared: ", multi_results['r_squared_test'])

if __name__ == "__main__":
    input_file = "Concrete_Data.xls"
    output_file = "output.xlsx"
    run_linear_regression_experiment_optimized(input_file, output_file=output_file)
    run_linear_regression_experiment_unoptimized(input_file)