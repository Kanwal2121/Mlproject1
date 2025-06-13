import os
import sys
from  src.exception import CustomException
from src.logger import logging
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from dataclasses import dataclass
from src.utils import model_evaluation,save_object
from sklearn.metrics import r2_score


@dataclass
class ModelTrainerConfig:
    trained_model_filepath=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting of train test split initiated")
            X_train_Data,X_test_Data,y_train_Data,y_test_Data=(train_array[:,:-1],test_array[:,:-1],train_array[:,-1],test_array[:,-1])
            models={
                        'LinearRegression':LinearRegression(),
                        'Ridge':Ridge(),
                        'Lasso':Lasso(),
                        'SVR':SVR(),
                        'DecisionTreeRegressor':DecisionTreeRegressor(),
                        'RandomForestRegressor':RandomForestRegressor(),
                        'KNeighborsRegressor':KNeighborsRegressor(),
                        'GradientBoostingRegressor':GradientBoostingRegressor()
                    }
            regressor_hyperparams ={
                        "LinearRegression": {
                            "fit_intercept": [True, False],
                            "positive": [True, False]
                        },
                        "Ridge": {
                            "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
                            "solver": ["auto", "svd", "cholesky", "lsqr", "sag"]
                        },
                        "Lasso": {
                            "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
                            "max_iter": [1000, 5000, 10000]
                        },
                        "SVR": {
                            "kernel": ["linear", "poly", "rbf", "sigmoid"],
                            "C": [0.1, 1.0, 10.0],
                            "gamma": ["scale", "auto"]
                        },
                        "DecisionTreeRegressor": {
                            "max_depth": [None, 5, 10, 20],
                            "min_samples_split": [2, 5, 10],
                            "min_samples_leaf": [1, 2, 4]
                        },
                        "RandomForestRegressor": {
                            "n_estimators": [100, 200, 500],
                            "max_depth": [None, 10, 20],
                            "min_samples_split": [2, 5, 10],
                            "min_samples_leaf": [1, 2, 4]
                        },
                        "KNeighborsRegressor": {
                            "n_neighbors": [3, 5, 7],
                            "weights": ["uniform", "distance"],
                            "p": [1, 2]  
                        },
                        "GradientBoostingRegressor": {
                            "n_estimators": [100, 200],
                            "learning_rate": [0.01, 0.05, 0.1],
                            "max_depth": [3, 5, 7],
                            "subsample": [0.8, 1.0]
                        }















            }
            model_report:dict=model_evaluation(X_train=X_train_Data,X_test=X_test_Data,y_train=y_train_Data,y_test=y_test_Data,models=models,params=regressor_hyperparams)
            best_model_score=max(sorted(model_report.values()))
            
            for key,value in model_report.items():
                if value==best_model_score:
                    best_model=key
                    break
            best_model_name=models[best_model]

            if best_model_score<0.6:
                raise CustomException("No Model Found")
            save_object(file_path=self.model_trainer_config.trained_model_filepath,obj=best_model_name)
            predicted=best_model_name.predict(X_test_Data)
            r2Score = r2_score(y_test_Data,predicted)
            return r2Score
        except Exception as e:
            raise CustomException(e,sys)



