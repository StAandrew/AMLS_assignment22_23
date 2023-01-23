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
from helper_utils import helper_plot_learning, helper_plot_learning, helper_plot_grid_c, helper_plot_grid_gamma, helper_plot_confusion_matrix

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

        self.label_test_predict = self.grid.predict(self.test_data)
        self.conf_matrix = confusion_matrix(self.test_labels, self.label_test_predict)

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
        self.logger.info("Results with best parameters:")
        self.logger.info(self.best_model)
        # self.logger.info("CV Results:")
        # self.logger.info(self.grid.cv_results_)
        self.logger.info("Detailed results:")
        for mean_score, params in zip(self.results["mean_test_score"], self.results["params"]):
            self.logger.info(f"mean_score: {mean_score:.3f}, params: {params}")
        self.logger.info("\nClassification Report:\n")
        self.logger.info(f"\n{classification_report(self.test_labels, self.label_test_predict)}")
        

    def plot_learning(self, figure_save_path, learning_curve_path):
        try:
            arrays = np.load(learning_curve_path)
            train_sizes = arrays['arr_0']
            train_scores = arrays['arr_1']
            val_scores = arrays['arr_2']
        except:
            self.logger.info("Learning curve data file not found, creating a new one...")
            train_sizes, train_scores, val_scores = learning_curve(
                self.grid.best_estimator_, self.train_data, self.train_labels, cv=5
            )
            np.savez(learning_curve_path, train_sizes, train_scores, val_scores)
        helper_plot_learning(figure_save_path, train_sizes, train_scores, val_scores)
        self.logger.info("Learning curve plot creaed.")

    def plot_confusion_matrix(self, figure_save_path):
        self.logger.info("plot_confusion_matrix plotting...")
        helper_plot_confusion_matrix(figure_save_path, self.conf_matrix)
        self.logger.info("Confusion matrix plot creaed.")

    def plot_grid_c(self, figure_save_path):
        self.logger.info("plot_grid_c preparing...")
        C = [self.results["C"] for self.results in self.grid.cv_results_["params"]]
        mean_scores = self.grid.cv_results_["mean_test_score"] * 100
        self.logger.info("plot_grid_c plotting...")
        helper_plot_grid_c(figure_save_path, C, mean_scores)
        self.logger.info("C parameter plot creaed.")

    def plot_grid_gamma(self, figure_save_path):
        self.logger.info("plot_grid_gamma preparing...")
        gamma = [self.results["gamma"] for self.results in self.grid.cv_results_["params"]]
        mean_scores = self.grid.cv_results_["mean_test_score"] * 100
        self.logger.info("plot_grid_gamma plotting...")
        helper_plot_grid_gamma(figure_save_path, gamma, mean_scores)
        self.logger.info("Gamma parameter plot creaed.")
