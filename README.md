# Nasdaq Predictive Model

### V0.1.0

![NASDAQ Image](/Images/Nasdaq-copy.jpg)

## Purpose and Goal of this Project/Repository

As markets become increasingly more volatile, and trading becomes more automated thanks to algorithmic trading, it has become more significant to take in a larger volume of information to make better informed decisions regarding investments in markets around the globe — more information than can be reasonably expected of a person to parse through and analyze individually. The goal of this project is to create a model that will analyze data fed into it by creating a model based off of historical data and using it to predict returns on future data based on trends. 

This predictive random forest model will focus on NASDAQ companies, and will use data from every single ticker in the exchange over the course of a year from July 2019 to July 2020 to train and test the data — over 900,000 data points. Our primary goal is to identify stocks that can be expected to outperform the NASDAQ, and our seconday goal with this is to practice more in the creation of models for forecasting purposes, as well as to familiarize ourselves more with AWS infrastructure, given that we will be using Sagemaker to collaborate on the project. 

### Team 

- Javier Mendez
- Reuben Lopez
- Scott Andersen 
- Sara Jankovic

### Contents of Repository

1. handler — Contains the necessary libraries that are required for the model to operate in AWS infrastructure, as well as an older version of the model (handler.py) which runs on a local machine but not AWS.
2. docker_deploy — Contains the most up-to-date model that was executed on AWS Lambda (handler.py), as well as the file necessary for deployment on AWS (Makefile).
3. nasdaq-historical-data — Contains historical data for the NASDAQ that the model is training and testing on, both as individual days and aggregated + cleaned. 
4. Images — Images used in the repository.

### Tools Used

- Excel
- Python
- Pandas (Python Library)
- VS Code
- AWS Sagemaker
- AWS Lambda
- AWS S3
- AWS CloudWatch

## Process of Creating the Model 

As all models do (as we have found out), it all begins with finding quality data that can be used to test and train it. We were able to find historical data for every NASDAQ ticker (for a small fee!). We have uploaded a year of this data to this repository so that it can be used by anyone wishing to test the model for themselves - the code is already written to intepret it right away. 

After the data was found, we cleaned it in such a fashion so that we were working with only a "returns" column for every ticker on every day from July 2019 to July 2020. We chose this data for two reasons: 
  
  1. This is *a lot* of data already for a model to train on. over 900,000 individual percentages for this model to learn on. We figured this was more than enough to create something that was accurate enough for our liking. 
  2. The data includes moments:
    A) Before COVID
    B) During the COVID Crash
    C) During the COVID Recovery
   This allows the model to learn from different types of "moments" in the stock market (stability, reduction, and growth).

From here on out, we were able to create the model, which suprisingly enough, often-times can be the easy part (compared to data-aggregation). This was all done on AWS Sagemaker, a service provided by Amazon that allows you to work on Jupyter Notebooks on their servers. This provides access to their massive amounts of processing power which is useful for working on a model with this much data, as well as allows us to more easily collaborate on the code. Here is an example of what some of the code looks like: 

![Code Example](/Images/code-example.gif)

Once we had the model created and running at a level that we liked it, we then go through the process of turning the model wihch is currently split up into multiple Jupyter Notebook cells into a variety of functions, each being a key aspect of the model. This is done so that we are able to connect it to SNS, the Amazon service we are going to be using to send the tickers via email to any users who wish to subscribe to the service the model provides. To get an idea of what this looks like, here is a chunk of the code in cells compared to what it looks like in a function:

![Code Example 2](/Images/transformation.PNG)

With SNS, we can then add emails that wish to recieve the data and the messages with the tickers can be sent whenever the code is run (as of V0.1.0)

## Deploying the Model

The model will do the analysis to create a 1-day forecast. With all the data points we fed into it, it "learned" ways to identify ticker movements and we can ask for it to return us the tickers that it believes will move in a positive direction the following day, and will return us the highest movers. 

For the model to be useful, it needs to be deployable. For this we opted to use AWS lambda to be able to execute the model once a day. We ran into a potential issue with our file sizes, in particular the data and imported libraries. To solve this we uploaded the data separately on S3, and then we used CloudWatch to configure the daily emails. After debugging, we were able to get it deployed and the email could be requested.

## Results
The results of the accuracy of the model are below:
![Model Accuracy](/Images/model-accuracy.png)

Additionally, here is an example of what the email subscribers receive:

![Email](/Images/email.png)

## How this is useful for you

There are a variety of ways you can trade with this knowledge, the two most common being:
 1. Momentum Trading (short-term): Day-by-day trading of stocks in the NASDAQ Composite. This is useful for day-traders who are trying to ride waves or identify trends quickly. Anyone looking to beat the market in the short-term can use this model to generate a list of tickers to keep on eye on for the following trading day.  
 2. Options Trading (Momentum, short-term or long-term): Quick gains with much higher risk. This is useful for momentum trading stock options intra-day.

## Resources

[Useful Article on Random Forest's and how they work.](https://en.wikipedia.org/wiki/Random_forest)

[Data Cleaning Tips and Tricks in Python](https://towardsdatascience.com/data-cleaning-in-python-the-ultimate-guide-2020-c63b88bf0a0d?gi=dd7bd10c80c6)

[AWS Sagemaker Tutorial](https://www.youtube.com/watch?v=8Vj7OaR4DcA)

[Random Forest Regression Article](https://towardsdatascience.com/random-forest-and-its-implementation-71824ced454)

[AWS Lambda Best Practices](https://docs.aws.amazon.com/lambda/latest/dg/best-practices.html)

[Know Ryan](https://www.linkedin.com/in/ryan-bacastow/)
