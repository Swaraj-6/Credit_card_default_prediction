import os
import sys

from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_model
from src.utils import save_object

from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()


    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Splitting independent and dependent features from train and test array")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            models = {
                LogisticRegression(): 'Logistic Regression',
                SVC(kernel='rbf', C=10, degree=5, gamma='auto'): "Support vector",
                KNeighborsClassifier(n_neighbors=20): 'Knn classifier',
                DecisionTreeClassifier(min_samples_split=15, min_samples_leaf=2, max_features='sqrt', max_depth=8,
                                       criterion='entropy'): 'Decision Tree',
                RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=4, min_samples_leaf=5,
                                       max_features='sqrt'): 'Random Forest',
                GaussianNB(): 'Naiye Bayes'
            }

            results = {}

            for model in models.keys():
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                scores = evaluate_model(y_test, y_pred)

                results[scores] = model

            best_model_score = max(results.keys())
            best_model = results[best_model_score]
            best_model_name = models[results[best_model_score]]


            print("\n==========================================\n")
            logging.info(f"Best model is {best_model_name} with accuracy of {best_model_score*100}")
            print(f"Best model is {best_model_name} with accuracy of {best_model_score * 100}")


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return (
                best_model_name,
                best_model_score
            )


        except Exception as e:
            logging.info("An exception has occurred in initiate_model_training")
            raise CustomException(e,sys)










