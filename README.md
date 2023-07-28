# Customer Acquisition Cost Predictor

This repository contains the code and resources for a Customer Acquisition Cost (CAC) predictor project. The aim of this project is to develop a system that can predict the cost of acquiring new customers for a business. The prediction is based on various customer details and uses machine learning regression techniques.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Selection](#model-selection)
- [Streamlit Web App](#streamlit-web-app)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

## Introduction
Customer Acquisition Cost (CAC) refers to the cost incurred by a business to acquire a new customer. Predicting the CAC helps businesses plan and allocate their marketing budgets effectively. This project focuses on building a model that can estimate the CAC based on customer details, enabling businesses to make informed decisions regarding customer acquisition strategies.

## Dataset
The dataset used for training and evaluation is available at [this link](https://www.kaggle.com/datasets/ramjasmaurya/medias-cost-prediction-in-foodmart). It contains relevant information such as customer demographics, marketing channels, advertising costs, and the corresponding CAC values. The dataset will be used to train the predictive model.

## Model Selection
Various regression models were evaluated for predicting the CAC, including XGBoost, linear regression, and Lasso regression. However, the Random Forest Regressor was selected for its simplicity and high accuracy. The model achieved a remarkable 99% accuracy and demonstrated a very low R-squared score, indicating its effectiveness in estimating the customer acquisition cost.

## Streamlit Web App
The project includes a user-friendly web application built using Streamlit. The web app provides a convenient interface for users to input customer details and obtain the predicted customer acquisition cost. With this interactive tool, businesses can quickly estimate the cost associated with acquiring new customers based on their specific characteristics.

## Usage
To use the Customer Acquisition Cost Predictor, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/CAC-predictor.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Download the dataset from [this link](https://www.kaggle.com/datasets/ramjasmaurya/medias-cost-prediction-in-foodmart) and place it in the appropriate directory.
4. Run the Streamlit web app: `streamlit run app.py`
5. Access the web app through the provided URL and enter the required customer details.
6. Obtain the predicted customer acquisition cost based on the input.

## Acknowledgments
Special thanks to the authors of the [dataset](https://www.kaggle.com/datasets/ramjasmaurya/medias-cost-prediction-in-foodmart) for providing the valuable data for this project.
## 1.1 Overview
Customer acquisition can be quite a challenging task in today’s world, mostly due to the fact that businesses face intense competitions from rival companies for the same target audience. To fight the competition, they have to formulate marketing strategies and tactics in order to stand out and attract customers. Besides, it can be quite challenging to entice a potential customer to buy your product from a wide range of choice offered to him / her. This project aims to use Machine Learning algorithms to predict the cost of acquiring a new customer based on the data of about 60,000 existing customers based on multiple factors such as the products sold, store features, income of the customer, etc.
## 1.2 Purpose
The purpose of this project is to analyse existing consumer behaviour and understand what appeals to them and on the basis of this information gathered, predict the cost of acquiring new customers.

## Demo Video
https://drive.google.com/drive/folders/17kKvgFMHHo-u6LgZrz24VUJRBNh25IuI?usp=drive_link

## Note

We have successfully submitted our assignments and project files along with the full project report on zoho forms with our personal github repo link with public access and also uploaded the link of the demo video on the smartinternz portal as per the instructions 

Thank you!
