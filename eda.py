"""This is a preliminary data_frame exploration file for all data_frame I will be working on.
It can perform the following features:

1. Data Loading: Functions to load data_frame from various sources such as
 CSV files, Excel sheets, databases, etc.

2. Data Inspection: Functions to quickly inspect the data_frame, including viewing the first few rows, 
checking data_frame types, and getting basic statistics like mean, median, and standard deviation.

3. Data Cleaning:  Functions to handle missing values, duplicate records, and outliers.
 This might include imputation techniques, removal of duplicates, and visualization of outliers.
 
4. Data Visualization: Functions for creating visualizations such as 
histograms, box plots, scatter plots, correlation matrices, etc., to gain insights into the data_frame's distribution and relationships.

5. Feature Engineering: Basic feature engineering techniques like
encoding categorical variables, scaling numerical features, and creating new features based on domain knowledge.

6. Correlation Analysis: Functions to compute and visualize correlations between features,
helping to identify important relationships in the data_frame.

7. Dimensionality Reduction: Techniques like principal component analysis (PCA) or 
t-distributed stochastic neighbor embedding (t-SNE) to visualize high-dimensional data_frame in lower dimensions.

8. Data Preprocessing: Functions to preprocess the data_frame for machine learning algorithms,
such as splitting into training and testing sets, scaling features, and handling class imbalances.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random # type: ignore
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.utils import resample


class DataLoader:
    def __init__(self, source_type, **kwargs):
        self.source_type  = source_type
        self.kwargs = kwargs

    def load_data(self):
        try:
            if self.source_type == "csv":
                data_frame = pd.read_csv(**self.kwargs)
            elif self.source_type == "excel":
                data_frame = pd.read_excel(**self.kwargs)
            elif self.source_type == "json":
                data_frame = pd.read_json(**self.kwargs)
            elif self.source_type == "sql":
                connection_string = self.kwargs.get("connection_string")
                query = self.kwargs.get("query")
                data_frame = pd.read_sql(query, connection_string)
            else:
                raise ValueError("Unsupported data_frame source type")
            
            return data_frame
        except Exception as e:
            print(f"An error occurred while loading the data_frame: {e}")
            return None

class DataInspector:
    def __init__(self, data_frame):
        """
        Initialize the DataInspector with a pandas DataFrame.

        Args:
            data_frame (pd.DataFrame): The input DataFrame to inspect.
        """
        self.data_frame = data_frame

    def view_data(self):
        """
        Displays the first few rows of the DataFrame.
        """
        print(self.data_frame.head())

    def data_info(self):
        """
        Provides information about the DataFrame, including data types and non-null counts.
        """
        print(self.data_frame.info())

    def missing_data(self):
        """
        Prints the count of missing values (NaN) for each column.
        """
        print(self.data_frame.isnull().sum())

    def data_stats(self):
        """
        Computes and displays summary statistics (mean, min, max, quartiles) for numerical columns.
        """
        print(self.data_frame.describe().T)

    def column_names(self):
        """
        Prints the names of all columns in the DataFrame.
        """
        print(self.data_frame.columns.tolist())

    def unique_values(self):
        """
        Prints unique values for each column.
        """
        for column in self.data_frame.columns:
            unique_vals = self.data_frame[column].unique()
            print(f"{column}: {unique_vals}")

    def sample_data(self, n_rows=5):
        """
        Print a random sample of the data
        Args:
            n_rows (int): Number of rows to return from the data.
        """
        print(self.data_frame.sample(n_rows))

class DataCleaner:

    def __init__(self, data_frame):
        self.df = data_frame

    def drop_columns(self, columns_to_drop):
        """
        Drop unnecessary columns from DataFrame.
        Args:
            columns_to_drop (list): List of column names to drop.
        """
        self.df.drop(columns=columns_to_drop, inplace=True)

    def clean_column(self, column_name, cleaning_function):
        """
        Clean a specific column using a custom cleaning function.
        Args:
            column_name(str): Name of the column to clean
            cleaning_function(callable): Custom function to clean the column.
        """
        self.df[column_name] = self.df[column_name].apply(cleaning_function)

    def remove_duplicates(self):
        """
        Remove duplicate rows from the DataFrame.
        """
        self.df.drop_duplicates(inplace=True)

    def fill_missing_values(self, method="mean"):
        """
        Fill missing values in the DataFrame based on a specified method.
        Args:
            method (str): Method for filling missing values (default: "mean").
            Options: "mean", "median", "mode", or a specific value.

        """
        if method == "mean":
            self.df.fillna(self.df.mean(), inplace=True)
        elif method == "median":
            self.df.fillna(self.df.median(), inplace=True)
        elif method == "mode":
            self.df.fillna(self.df.mode().iloc[0], inplace=True)
        else:
            # Fill with a specific value (e.g., 0)
            self.df.fillna(method, inplace=True)

    def correct_column_dtype(self, column_name, new_dtype):
        """
        Corrects the data type of a specific column.
        Args:
            column_name (str): Name of the column to correct.
            new_dtype (str or dtype): Desired data type for the column.

        """
        self.df[column_name] = self.df[column_name].astype(new_dtype)


    def visualize_outliers(self, num_iterations=5):
        """
        Randomly select columns and create scatter plot to visualize outliers.
        Args:
            num_iterations (int): Number of iterations (default: 5)
        """
        num_colums = len(self.df.columns)
        for _  in range(num_iterations):
            x_column, y_column = random.sample(self.df.columns.tolist(), 2)

            plt.figure(figsize=(8,6))
            plt.scatter(self.df[x_column], self.df[y_column], alpha=0.5, color="b", label="Data points")
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.title(f"Scatter Plot: {x_column} vs. {y_column}")
            plt.grid(True)
            plt.legend()
            plt.show()

class DataVisualization:    
    def __init__(self, data):
        """
        Initialize the DataVisualization class with a dataset.

        Args:
            data (pd.DataFrame): The input dataset.
        """
        self.data = data
    
    def plot_histogram(self, num_columns=6, bins=20):
        """
        Plot histograms for random columns in a 3x3 grid.

        Args:
            num_columns (int, optional): Number of columns to plot (default is 6).
            bins (int, optional): Number of bins for the histograms.
        """
        # Randomly select columns
        selected_columns = random.sample(list(self.data.columns), num_columns)

        # Create a 2x3 grid of subplots
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 10))
        fig.suptitle("Histograms of Random Columns", fontsize=16)

        for i, column in enumerate(selected_columns):
            row, col = divmod(i, 3)
            ax = axes[row, col]
            sns.histplot(data=self.data, x=column, bins=bins, kde=True, ax=ax)
            ax.set_title(f"{column} Histogram")
            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")

        plt.tight_layout()
        plt.show()

    def plot_boxplot(self, x_column, y_column, hue=None):
        """
        Plot a boxplot to compare two columns.

        Args:
            x_column (str): Name of the x-axis column.
            y_column (str): Name of the y-axis column.
            hue (str, optional): Name of the column to use for color grouping (e.g., for categorical data).
        """
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=x_column, y=y_column, data=self.data, hue=hue)
        plt.title(f"Boxplot: {x_column} vs. {y_column}")
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.show()

    def plot_countplot(self, x_column, hue=None):
        """
        Plot a countplot to visualize counts of other columns based on a specific column.

        Args:
            x_column (str): Name of the x-axis column.
            hue (str, optional): Name of the column to use for color grouping (e.g., for categorical data).
        """
        plt.figure(figsize=(8, 6))
        sns.countplot(x=x_column, data=self.data, hue=hue)
        plt.title(f"Countplot: Counts of {x_column}")
        plt.xlabel(x_column)
        plt.ylabel("Count")
        plt.show()

    def correlation_matrix(self):
        """
        Shows the correlation between columns.    
        
        """
        corr =  self.data.corr()
        sns.heatmap(corr, linewidths=.5, annot=True)
        plt.show()

from sklearn.preprocessing import PowerTransformer

class FeatureEngineering:
    def __init__(self, numerical_features=None, categorical_features=None, label_feature=None):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.label_feature = label_feature

    def create_pipeline(self, scaling_method='standard', encoding_method='onehot', power_transform=False):
        """
        Create a data processing pipeline.

        Args:
            scaling_method (str, optional): Scaling method ("standard", "minmax", or "robust"). Default is "standard".
            encoding_method (str, optional): Encoding method for categorical features ("onehot" or "label"). Default is "onehot".
            power_transform (bool, optional): Whether to apply PowerTransformer for numerical features. Default is False.

        Returns:
            sklearn.pipeline.Pipeline: Data processing pipeline.
        """
        transformers = []

        # Define preprocessing for numeric columns
        if self.numerical_features:
            numerical_transformer = self._get_numerical_transformer(scaling_method, power_transform)
            transformers.append(('num', numerical_transformer, self.numerical_features))

        # Define preprocessing for categorical columns
        if self.categorical_features:
            categorical_transformer = self._get_categorical_transformer(encoding_method)
            transformers.append(('cat', categorical_transformer, self.categorical_features))

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(transformers=transformers)
        
        return preprocessor

    def _get_numerical_transformer(self, scaling_method, power_transform):
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Invalid scaling method. Use 'standard', 'minmax', or 'robust'.")

        if power_transform:
            return Pipeline(steps=[('scaler', scaler), ('power_transform', PowerTransformer())])
        else:
            return Pipeline(steps=[('scaler', scaler)])

    def _get_categorical_transformer(self, encoding_method):
        if encoding_method == 'onehot':
            return Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
        elif encoding_method == 'label':
            return Pipeline(steps=[('label', LabelEncoder())])
        else:
            raise ValueError("Invalid encoding method. Use 'onehot' or 'label'.")

    
class CorrelationAnalysis:
    def __init__(self, data):
        """
        Initialize the CorrelationAnalysis class with a dataset.

        Args:
            data (pd.DataFrame): The input dataset.
        """
        self.data = data

    def compute_correlation_matrix(self, method='pearson'):
        """
        Compute the correlation matrix for numerical features in the dataset.

        Args:
            method (str, optional): Correlation method ('pearson', 'kendall', or 'spearman'). Default is 'pearson'.

        Returns:
            pd.DataFrame: Correlation matrix.
        """
        if method not in ['pearson', 'kendall', 'spearman']:
            raise ValueError("Invalid correlation method. Use 'pearson', 'kendall', or 'spearman'.")

        corr_matrix = self.data.select_dtypes(include=np.number).corr(method=method)
        return corr_matrix

    def visualize_correlation_matrix(self, method='pearson', cmap='coolwarm'):
        """
        Visualize the correlation matrix for numerical features in the dataset.

        Args:
            method (str, optional): Correlation method ('pearson', 'kendall', or 'spearman'). Default is 'pearson'.
            cmap (str, optional): Color map for the heatmap. Default is 'coolwarm'.
        """
        corr_matrix = self.compute_correlation_matrix(method=method)

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap=cmap, fmt=".2f", annot_kws={"size": 10})
        plt.title(f'Correlation Matrix ({method.capitalize()} correlation)')
        plt.show()

class DimensionalityReduction:
    def __init__(self, data):
        """
        Initialize the DimensionalityReduction class with a dataset.

        Args:
            data (pd.DataFrame): The input dataset.
        """
        self.data = data

    def apply_pca(self, n_components=2):
        """
        Apply Principal Component Analysis (PCA) for dimensionality reduction.

        Args:
            n_components (int, optional): Number of components. Default is 2.

        Returns:
            np.array: Transformed data after PCA.
        """
        pca = PCA(n_components=n_components)
        transformed_data = pca.fit_transform(self.data)
        return transformed_data

    def apply_tsne(self, n_components=2, perplexity=30, learning_rate=200):
        """
        Apply t-Distributed Stochastic Neighbor Embedding (t-SNE) for dimensionality reduction.

        Args:
            n_components (int, optional): Number of components. Default is 2.
            perplexity (float, optional): Perplexity parameter for t-SNE. Default is 30.
            learning_rate (float, optional): Learning rate for t-SNE. Default is 200.

        Returns:
            np.array: Transformed data after t-SNE.
        """
        tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate)
        transformed_data = tsne.fit_transform(self.data)
        return transformed_data

    def plot_dimensionality_reduction(self, transformed_data, labels=None, method='PCA'):
        """
        Plot the transformed data after dimensionality reduction.

        Args:
            transformed_data (np.array): Transformed data after dimensionality reduction.
            labels (array-like, optional): Labels for data points. Default is None.
            method (str, optional): Dimensionality reduction method ('PCA' or 't-SNE'). Default is 'PCA'.
        """
        if transformed_data.shape[1] != 2:
            raise ValueError("Plotting is supported for 2-dimensional data only.")

        plt.figure(figsize=(8, 6))
        if labels is None:
            plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c='b', alpha=0.5)
        else:
            unique_labels = set(labels)
            for label in unique_labels:
                plt.scatter(transformed_data[labels == label, 0], transformed_data[labels == label, 1], label=label, alpha=0.5)
            plt.legend()
        plt.title(f'Dimensionality Reduction using {method}')
        plt.xlabel(f'Component 1 ({method})')
        plt.ylabel(f'Component 2 ({method})')
        plt.grid(True)
        plt.show()

class DataPreprocessing(DataInspector, DataVisualization):
    def __init__(self, data):
        """
        Initialize the DataPreprocessing class with a dataset.

        Args:
            data (pd.DataFrame): The input dataset.
        """
        self.data = data
    
    def numerical_features(self, target_column):
        """
        Gets the numerical feature column names from a dataset.
        Args:
            target_column (str): Column name to be excluded from features in the data.
        """
        features = self.data.drop(columns=[target_column])
        numerical_features = [col for col in features if features[col].dtype in ['int64','float64']]
        return numerical_features
    
    def categorical_features(self, target_column):
        """
        Gets the categorical feature column names from a dataset.
        Args:
            target_column(str): Column name to be excluded from features in data.
        """
        features = self.data.drop(columns=[target_column])
        categorical_features = [col for col in features if features[col].dtype in ['object']]
        return categorical_features

    def split_train_and_test_data(self, X, y, test_size=0.3, random_state=None):
        """
        Split the dataset into training and testing sets.

        Args:
            X 
            test_size (float, optional): Proportion of the dataset to include in the test split. Default is 0.2.
            random_state (int or None, optional): Random seed for reproducibility. Default is None.

        Returns:
            tuple: Tuple containing X_train, X_test, y_train, y_test.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    def split_data(self, target_column):
        """
        Split the dataset into training and testing sets.

        Args:
            target_column (str): Name of the target column.
        Returns:
            tuple: features: X, labels: y
        """
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        return X, y
    

    def handle_class_imbalance(self, X_train,y_train, method='upsample', random_state=None):
        """
        Handle class imbalance in the target variable.

        Args:
            y_train (pd.Series): Target variable of the training data.
            method (str, optional): Resampling method ('upsample' or 'downsample'). Default is 'upsample'.
            random_state (int or None, optional): Random seed for reproducibility. Default is None.

        Returns:
            pd.DataFrame: Resampled dataset.
        """
        # Combine features and target variable
        train_data = pd.concat([X_train, y_train], axis=1)
        
        # Separate majority and minority classes
        majority_class = train_data[y_train == y_train.value_counts().idxmax()]
        minority_class = train_data[y_train != y_train.value_counts().idxmax()]
        
        if method == 'upsample':
            minority_class_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=random_state)
            resampled_data = pd.concat([majority_class, minority_class_upsampled])
        elif method == 'downsample':
            majority_class_downsampled = resample(majority_class, replace=False, n_samples=len(minority_class), random_state=random_state)
            resampled_data = pd.concat([majority_class_downsampled, minority_class])
        else:
            raise ValueError("Invalid resampling method. Use 'upsample' or 'downsample'.")

        return resampled_data
    
    def handle_outliers(self, method='isolation_forest', contamination=0.05, random_state=None):
        """
        Handle outliers in the dataset.

        Args:
            method (str, optional): Outlier detection method ('isolation_forest' or 'z_score'). Default is 'isolation_forest'.
            contamination (float, optional): Proportion of outliers to expect in the data. Default is 0.05.
            random_state (int or None, optional): Random seed for reproducibility. Default is None.

        Returns:
            pd.DataFrame: Dataset with outliers handled.
        """
        if method == 'isolation_forest':
            outlier_detector = IsolationForest(contamination=contamination, random_state=random_state)
            outlier_labels = outlier_detector.fit_predict(self.data)
            clean_data = self.data[outlier_labels == 1]
        elif method == 'z_score':
            z_scores = self.data.apply(lambda x: (x - x.mean()) / x.std())
            clean_data = self.data[(z_scores.abs() < 3).all(axis=1)]
        else:
            raise ValueError("Invalid outlier detection method. Use 'isolation_forest' or 'z_score'.")

        return clean_data
    
    def replace_values_with_indices(self):
        """
        Replace unique values in specified columns with their corresponding indices.
        Args:
            columns_to_modify (list or str): List of column names or a single column name to modify.
        """
        columns_to_modify = [col for col in self.data if self.data[col].dtypes in ['object']]

        for col in columns_to_modify:
            unique_values = self.data[col].unique()
            value_to_index = {value: index for index, value in enumerate(unique_values)}
            self.data[col] = self.data[col].replace(value_to_index)

        return self.data




