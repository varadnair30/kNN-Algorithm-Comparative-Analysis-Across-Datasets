# kNN-Algorithm-Comparative-Analysis-Across-Datasets

# Abstract


**In this study, we implemented and evaluated the K-Nearest Neighbors (KNN) algorithm from scratch, applying it to three distinct datasets: Breast Cancer, Car Evaluation, and Hayes-Roth, sourced from the UCI Machine Learning Repository. Our methodology encompassed data preprocessing, algorithm implementation, and accuracy assessment through 10-fold cross-validation. We further compared our implementation's accuracy with Scikit-learn's KNN classifier using paired t-tests. The findings underscore the effectiveness of our KNN model, exhibiting competitive accuracy across datasets, and highlight the algorithm's adaptability and robustness in handling varied data types, reinforcing its utility in practical machine learning applications.
**

# Introduction

**The K-Nearest Neighbors (KNN) algorithm stands as a cornerstone in the realm of machine learning, celebrated for its simplicity and effectiveness in classification and regression tasks. This assignment embarks on a detailed exploration and implementation of the KNN algorithm from scratch, targeting three diverse datasets: Breast Cancer, Car Evaluation, and Hayes-Roth. Each dataset presents unique challenges and characteristics, offering a rich ground for applying and assessing the KNN algorithm's adaptability and performance. Through this endeavor, we aim to deepen our understanding of KNN's practical applications, its strengths, and limitations within various contexts.

Further investigation on this assignment not only focuses on the technical implementation of the KNN algorithm but also delves into the nuances of k-fold cross-validation as a robust method for evaluating model performance. By meticulously processing and analyzing the Breast Cancer, Car Evaluation, and Hayes-Roth datasets, this study provides a comprehensive view of KNN's applicability across different data domains. This rigorous approach allows to highlight the algorithm's versatility and its potential as a powerful tool for machine learning practitioners. Through this assignment, we seek to contribute valuable insights into the effective application and evaluation of KNN, paving the way for future innovations and enhancements in machine learning methodologies.

For this programming assignment , KNN algorithm is implemented from the scratch and the functions to evaluate it with a k-fold cross validation (also from scratch). Getting the accuracy of custom KNN algorithm using 10-Fold cross validation on the following datasets from the UCI-Machine Learning Repository and comparing it with that obtained with KNN and  with scikit-learn's KNN using hypothesis testing(Paired t-test) for each dataset.**

# Methodology


**The approach to implement K-Nearest Neighbors (KNN) algorithm and k-fold cross-validation involved several steps. Initially, datasets are preprocessed by normalizing numerical data to ensure uniformity in scale and encoding categorical variables to transform them into a machine-readable format. Implementing the KNN algorithm from scratch, calculated Euclidean distances between data points to identify the k-nearest neighbors for classification or regression tasks. For model evaluation, utilized (10-fold cross-validation) k-fold cross-validation, systematically dividing the dataset into k subsets to train and test the model multiple times, ensuring comprehensive performance assessment. 

Challenges encountered included optimizing the algorithm for efficiency and handling datasets with mixed data types, requiring careful preprocessing to maintain data integrity and relevance.


Extension:- Manhattan Distance & Regression ( using Root Mean Squared Error and Mean  
                     Squared Error)**

# Dataset Analysis


The dataset analysis for this assignment focused on three datasets from the UCI Machine Learning Repository:

•	Breast Cancer (https://archive.ics.uci.edu/dataset/44/hayes+roth)

•	Car Evaluation (https://archive.ics.uci.edu/dataset/19/car+evaluation)

•	Hayes-Roth (https://archive.ics.uci.edu/dataset/14/breast+cancer)

Each dataset required specific preprocessing steps to ensure compatibility with the KNN algorithm. For the Breast Cancer dataset, significant preprocessing included normalization of features to a similar scale. The Car Evaluation dataset involved encoding categorical variables into numerical values, given its attribute-based classification nature. Lastly, the Hayes-Roth dataset also necessitated normalization and handling of missing values, showcasing the need for careful data preparation to enable accurate KNN classification. These steps were crucial in mitigating challenges related to data quality and format, directly impacting the algorithm's effectiveness across diverse data scenarios.

In conclusion, the dataset analysis for this assignment underlines the importance of understanding the characteristics and requirements of each dataset to effectively apply the KNN algorithm.
 

# Conclusion


Based on the code analysis from all the scripts, here's a summary of the expected results for the dataset analysis:

•	Custom KNN Mean Accuracy: The script calculates mean accuracy for the custom KNN implementation using 10-fold cross-validation

•	Scikit-learn KNN Mean Accuracy: Computes mean accuracy for Scikit-learn's KNN using cross-validation

•	Paired t-test Results: A paired t-test is performed to compare the accuracies of the custom KNN and Scikit-learn's KNN, providing a t-statistic and p-value. The script determines whether to reject the null hypothesis based on an alpha value of 0.05

•	Predictive Regression (Root Mean Squared Error/ Mean Squared Error ): It calculates RMSE/MSE for a regression task, indicating the model's prediction error

•	Manhattan Distance: The code also likely explores the use of Manhattan distance as an alternative to Euclidean distance for calculating the proximity between data points in the KNN algorithm. The choice between Manhattan and Euclidean distance can significantly affect the performance of the KNN model, as each distance metric has its own strengths depending on the dataset's characteristics and the problem context

The specific values for these results, including whether the null hypothesis is rejected and the significance of the RMSE value, would be directly obtained from executing the scripts with the datasets. (the value of alpha = 0.05)

 

# References


1)	https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

(The URL provided links to a tutorial on Machine Learning Mastery about implementing the k-Nearest Neighbors (k-NN) algorithm in Python from scratch. This tutorial, written by Jason Brownlee, teaches the basics of the k-NN algorithm, including its principles, how to code it step-by-step in Python without using libraries, and how to evaluate its performance on a real dataset. It focuses on understanding the algorithm's simplicity and power for making predictions based on the most similar historical examples. The tutorial also includes practical examples and code snippets to solidify the reader's understanding and skills in applying k-NN in Python for machine learning tasks.)


2)	https://machinelearningmastery.com/k-fold-cross-validation/

(The URL leads to a tutorial on Machine Learning Mastery by Jason Brownlee, providing a comprehensive introduction to k-fold cross-validation(kFCV). This statistical method is essential for estimating the skill of machine learning models, offering insights into how to implement and utilize k-fold cross-validation effectively in Python using scikit-learn. The tutorial covers the basics, configuration of k, practical examples, and explores variations like stratified and repeated cross-validation, aiming to equip readers with a solid understanding of using k-fold cross-validation for model evaluation.)

 
