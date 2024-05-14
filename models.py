import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score 


class ClassificationModels:
    def __init__(self, model_type, **kwargs):
        self.model_type = model_type
        self.kwargs = kwargs
        self.model = self._get_model()

    def _get_model(self):
        if self.model_type == "logistic_regression":
            return LogisticRegression(**self.kwargs)
        elif self.model_type == "decision_tree":
            return DecisionTreeClassifier(**self.kwargs)
        elif self.model_type == "random_forest":
            rf_param_space = {
            'n_estimators': Integer(100, 300),
            'max_depth': Integer(5, 10),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1,4),
            'max_features': Categorical(['sqrt', 'log2'])
            }
            return BayesSearchCV(RandomForestClassifier(), rf_param_space,cv=5, n_jobs=1, n_iter=50,random_state=42)
        elif self.model_type == "svm":
            return SVC(**self.kwargs)
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(**self.kwargs)
        elif self.model_type == "xgb_classifier":
            xgb_param_space = {
            'n_estimators': Integer(100, 300),
            'max_depth': Integer(3, 7),
            'learning_rate': Real(0.001, 0.1, 'log-uniform'),
            'subsample':Real(0.8, 1.0, 'uniform'),
            'colsample_bytree': Real(0.8, 1.0, 'uniform')
            }
            return BayesSearchCV(XGBClassifier(), xgb_param_space, cv=5, n_jobs=1, n_iter=50, random_state=42)

        else:
            raise ValueError("Unsupported model type. Model type can only be logistic_regression, decision_tree\b"
                             "random_forest", "svm", "gradient_boosting", "xgb_classifer")

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_scores = self.model.predict_proba(X_test)
        cm = confusion_matrix(y_test, y_pred)
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"Confusion Matrix:\n {confusion_matrix(y_test, y_pred)}")
        print(f"Classification Report:\n {classification_report(y_test, y_pred)}")
        auc_score = roc_auc_score(y_test, y_scores[:, 1])
        print('AUC: + ',str(auc_score))

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])

        fig = plt.figure(figsize=(8,6))
        plt.plot([0,1], [0,1], 'k--')
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC curve for {self.model_type}")
        plt.show()       
       

class RegressionModels:
    def __init__(self, model_name='linear_regression', degree=1, **kwargs):
        self.model_name = model_name
        self.degree = degree
        self.kwargs = kwargs
        self.models = {
            "linear_regression": LinearRegression(**kwargs),
            "polynomial_regression": make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(**kwargs)),
            "decision_tree_regression": DecisionTreeRegressor(**kwargs),
            "random_forest_regression": RandomForestRegressor(**kwargs),
            "ridge_regression": Ridge(**kwargs),
            "lasso_regression": Lasso(**kwargs),
            "support_vector_regression": SVR(**kwargs)
        }
        

    def train(self, X_train, y_train):
        model = self.models.get(self.model_name)
        if model:
            model.fit(X_train, y_train)
        else:
            raise ValueError(f"Invalid model name: {self.model_name}")

    def predict(self, X_test):
        model = self.models.get(self.model_name)
        if model:
            return model.predict(X_test)
        else:
            raise ValueError(f"Invalid model name: {self.model_name}")

    def set_model(self, model_name):
        if model_name in self.models:
            self.model_name = model_name
        else:
            raise ValueError(f"Invalid model name: {model_name}")

    def get_model(self):
        return self.models.get(self.model_name)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        print(f"R2 score :{r2_score(y_test,y_pred)}")
        print(f"MSE score :{mean_squared_error(y_test,y_pred)}")
        plt.figure(figsize=(8,6))
        plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
        plt.title(f"Actual Output vs. Predicted Output {(self.model_name).upper()}")
        plt.xlabel("Actual Output")
        plt.ylabel("Predicted Output")
        plt.show()