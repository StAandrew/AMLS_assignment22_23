import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import os
import time

# defining parameter range
C = [0.01, 0.1, 1, 10]
gamma = [10, 1, 0.1, 0.01]
kernel = ["linear"]

class A1():
    def __init__(self, train_data, train_labels, test_data, test_labels, logger):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.logger = logger
    
    def train_grid_fit(self, model):
        param_grid = {
            "C": C,
            "gamma": gamma,
            "kernel": kernel,
        }

        grid = GridSearchCV(model, param_grid, refit=True, verbose=3, cv=5)
        grid.fit(self.train_data, self.train_labels)
        self.grid = grid
        self.results = grid.cv_results_
    
    def evaluate_best_model(self):        
        self.best_model = self.grid.best_estimator_.score(self.test_data, self.test_labels)

        label_test_predict = self.grid.predict(self.test_data)
        self.conf_matrix = confusion_matrix(self.test_labels, label_test_predict)

    def save_model(self, model_path):
        joblib.dump(self.grid, model_path, compress=1)

    def load_model(self, model_path):
        try:
            self.grid = joblib.load(model_path)
            self.results = self.grid.cv_results_
            return True
        except:
            return False

    def output_info(self):
        self.logger.info("Best parameters after tuning:")
        self.logger.info(self.grid.best_params_)
        self.logger.info("Verify score with best model:")
        self.logger.info(self.grid.best_estimator_)
        self.logger.info("Best model:")
        self.logger.info(self.best_model)
        self.logger.info("CV Results:")
        self.logger.info(self.grid.cv_results_)
        for mean_score, params in zip(self.results["mean_test_score"], self.results["params"]):
            self.logger.info(f"mean_score: {mean_score:.3f}, params: {params}")

    def plot(self):
        train_sizes, train_scores, val_scores = learning_curve(
            self.grid.best_estimator_, self.train_data, self.train_labels, cv=5
        )

        plt.plot(train_sizes, train_scores.mean(axis=1), label="Training Score")
        plt.plot(train_sizes, val_scores.mean(axis=1), label="Validation Score")
        plt.xlabel("Number of Training Examples")
        plt.ylabel("Score")
        plt.legend()
        plt.title("SVM learning curve", fontsize=32)
        plt.savefig("A1/SVC_learningcurve.png")
