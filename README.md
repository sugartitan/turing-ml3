# Lending Club Data Analysis
3rd Project for Turing College ML Module

Deployed model link: https://streamlit-3kybr4m6ua-lz.a.run.app/

## About this Part
Congrats! You have reached the last Part of this Sprint. In this Part, you will put what you learned during this and the previous Sprints into practice. As the final assignment of this Sprint, you solve a lending automation problem for LendingClub. You will have to apply all you have learned about training and deploying machine learning models to complete this task. Our expectation is that you'll use your own judgment on how to perform the analysis and select the most important avenues of modeling, statistical testing, and exploration. You'll have to iteratively try to find patterns in the data, raise hypotheses and use your data analysis skills to get answers.

P.S. we don't expect this project to be perfect - you will continue to improve your skills and there will be many projects for you to apply your newly gained skills in the future. For now just use what you have learned and try your best!

## Context
Imagine that you are a data scientist who was just hired by the LendingClub. They want to automate their lending decisions fully, and they hired you to lead this project. Your team consists of a product manager to help you understand the business domain and a software engineer who will help you integrate your solution into their product. During the initial investigations, you've found that there was a similar initiative in the past, and luckily for you, they have left a somewhat clean dataset of LendingClub's loan data. The dataset is located in a public bucket here: https://storage.googleapis.com/335-lending-club/lending-club.zip (although you were wondering if having your client data in a public bucket is such a good idea). In the first meeting with your team, you all have decided to use this dataset because it will allow you to skip months of work of building a dataset from scratch. In addition, you have decided to tackle this problem iteratively so that you can get test your hypothesis that you can automate these decisions and get actual feedback from the users as soon as possible. For that, you have proposed a three-step plan on how to approach this problem. The first step of your plan is to create a machine learning model to classify loans into accepted/rejected so that you can start learning if you have enough data to solve this simple problem adequately. The second step is to predict the grade for the loan, and the third step is to predict the subgrade and the interest rate. Your team likes the plan, especially because after every step, you'll have a fully-working deployed model that your company can use. Excitedly you get to work!

## Objectives for this Part
* Practice downloading datasets from external sources.
* Practice performing EDA.
* Practice applying statistical inference procedures.
* Practice using various types of machine learning models.
* Practice building ensembles of machine learning models.
* Practice using hyperparameter tuning.
* Practice using AutoML tools.
* Practice deploying machine learning models.
* Practice visualizing data with Matplotlib & Seaborn.
* Practice reading data, performing queries, and filtering data.
## Requirements
* Download the data from here.
* Perform exploratory data analysis. This should include creating statistical summaries and charts, testing for anomalies, checking for correlations and other relations between variables, and other EDA elements.
* Perform statistical inference. This should include defining the target population, forming multiple statistical hypotheses and constructing confidence intervals, setting the significance levels, conducting z or t-tests for these hypotheses.
* Apply various machine learning models to predict the target variables based on your proposed plan. You should use hyperparameter tuning, model ensembling, the analysis of model selection, and other methods. The decision where to use and not to use these techniques is up to you, however, they should be aligned with your team's objectives.
* Deploy these machine learning models to Google Cloud Platform. You are free to choose any deployment option you wish as long as it can be called an HTTP request.
* Provide clear explanations in your notebook. Your explanations should inform the reader what you are trying to achieve, what results you got, and what these results mean.
* Provide suggestions about how your analysis and models can be improved.
## Evaluation Criteria
* Adherence to the requirements. How well did you meet the requirements?
* Depth of your analysis. Did you just skim the surface, or did you explored the dataset in-depth?
* Model's performance. How well did your model perform the predictions?
* Model's deployment. How performant, robust, and scalable your model deployment is?
* Visualization quality. Did you use charts effectively to visualize patterns in the data? Are your visualizations properly labeled? Did you use colors effectively? Did you adhere to the principle of proportional ink?
* Code quality. Was your code well-structured? Did you use the appropriate levels of abstraction? Did you remove commented-out and unused code? Did you adhere to the PEP8?
* Code performance. Did you use suitable algorithms and data structures to solve the problems?
