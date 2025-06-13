import os
import sys
from  dataclasses  import dataclass
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object



@dataclass
class DataTransformationConfig:
    preprocessor_obj_filepath=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformer_object(self):
        logging.info("Creating Pipeline")
        Numerical_features=['reading_score', 'writing_score']
        Categorical_features=['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

        N_pipeline=Pipeline(
            steps=[
                ("SimpleImputer",SimpleImputer(strategy='median')),
                ("StandardScaler",StandardScaler())
            ]
         )
        C_pipeline=Pipeline(
            steps=[
                ('Simple_Imputer',SimpleImputer(strategy='most_frequent')),
                ("OnehotEncoding",OneHotEncoder()),
                ("StandardScaler",StandardScaler(with_mean=False))
                

            ]
        )
        logging.info("PipeLine Created")

        logging.info("Applying Column Transformer")
        preprocessor=ColumnTransformer(
            [
            ("Transformation_on_Numerical_Features",N_pipeline,Numerical_features),
            ("Transformation_on_Categorical_Features",C_pipeline,Categorical_features)
            ]



        )
        return preprocessor
    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info("Reading of train test data started")
            training_dataset=pd.read_csv(train_path)
            test_dataset=pd.read_csv(test_path)
            logging.info("Reading of Training and Test Data Completed")

            logging.info("Preprocessing Object Started")
            preprocessing_object=self.get_data_transformer_object()
            logging.info("Preprocessing Object Completed")
            
            
            logging.info("Data into Dependent and Independent Started ")
            target_column_name="math_score"
            X_train=training_dataset.drop(target_column_name,axis=1)
            y_train=training_dataset[target_column_name]

            X_test=test_dataset.drop(target_column_name,axis=1)
            y_test=test_dataset[target_column_name]

            logging.info("Data into Dependent and Independent Completed")

            logging.info("Transformation Started")

            X_train_array=preprocessing_object.fit_transform(X_train)
            X_test_array=preprocessing_object.transform(X_test)

            

            train_array=np.c_[
                X_train_array,np.array(y_train)
            ]
            test_array=np.c_[
                X_test_array,np.array(y_test)
            ]

            logging.info("Transformation Completed")

            logging.info("Save object called ")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_filepath,
                obj=preprocessing_object

            )
            logging.info("Save object finshed ")


            return(
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_filepath
            )


        except Exception as  e:
            raise CustomException(e,sys) 
    





