import pandas as pd
import sys
import os
import numpy as np
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open (file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)
        


def model_evaluation(X_train,X_test,y_train,y_test,models,params):
    try:
        report={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            para=params[list(params.keys())[i]]
            gs=GridSearchCV(model,para,cv=3)

            gs.fit(X_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)



            y_predict=model.predict(X_test)

            Score=r2_score(y_test,y_predict)
            report[list(models.keys())[i]]=Score
        return report



    except Exception as e:
       raise CustomException(e,sys) 
    
def load_object(file_path):
    with open(file_path,'rb') as f:
        return dill.load(f)
