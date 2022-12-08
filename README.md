# Model Evaluation on Financial Tweets and Stock Trend Prediction
Group: Yunhan Zhang, Kaiyue Wei, Ningyuan Zhou, Qiaojuan Tu

**Special thanks to Dr. Nakul Padalkar from Geogretown DSAN program for helping us train the bert pre-training model on personal computer.**

## Background and Motivation
Before the internet, there could be a significant lag time between the unfolding of an event and the public’s awareness of it. With the advent of social media, information could spread more quickly. Twitter stands out among social media sites as a valuable source of up-to-date breaking news, alerts, and tips that can inform trading decisions. A single investor can simply monitor Twitter for major announcements and, lately, financial market trends instead of various websites and other publication channels. Posts on the social media platform can influence stock returns, according to research led by a West Virginia University financial expert.

## Proposal and Goals
Through the project, we would like to know whether Twitter posts really affect the stock trend and have built and trained several models based on financial tweets and stock market price index change (Rise/Fall). We will also use the model with the best performance to predict the next stock index movement within some time frame by using financial text content. 

This topic interests us since we hypothesize that current social media affects the markets and thus leads to a change in the stock index. The models used on financial tweets to reflect and predict the next stock index movement trend within some time frame [Rise or Fall classification] are Logistic Regression, Naive Bayes, and BERT models. This project evaluates the performance of different models and uses Python codes with a github repository to deliver the results and predictions.

## Data Description
For the stock index change variable, we will use two commonly followed equity indices of the stock market that people generally refer to. The first one is the Standard and Poor's 500, or simply the S&P 500, a stock market index tracking the stock performance of 500 large companies listed on stock exchanges in the United States. The other one is the New York Stock Exchange, or simply the NYSE. The two indexes consist of different groups of companies, some of which may overlap.
For the financial tweet content data, an article that we found lists the 24 best Twitter profiles that people who are interested in finance and the stock market have to follow. It also lists Goldman Sachs and Morgan Stanley, which are reliable. We scraped and generated tweets from all 24 accounts through the Twitter API.

## Packages
Following are the main Python packages that were used to complete this project: 
1. Transformers
2. Scikit-learn 
3. PyTorch 
4. Keras

## Workflow
After data generation, the next step is data cleaning for both the tweet text data and the two stock index datasets. For the financial tweet text data, we perform traditional text processing such as lowercase text, removing stop words, hyperlinks, emails, numbers, extra space, and punctuation. After cleaning the text data, we tokenized the text by using three different tools: the plain CountVectorizer, Ngram CountVectorizer, and TF-IDF with N-gram CountVectorizer. By comparing three tokenizing tools, we choose the best possible numerical representation of the text strings for running models. We chose N-grams from 1 to 4 because 4 can be the most relevant term size to our financial topic. Those tokenized words are also lemmatized and stemmed in order to yield a good result.

For the index data, we calculated the weekly stock price mean difference and last day difference using the mean of the current week minus the mean of the previous week. If the number is greater than 0, it is labeled "RISE". If it is less than zero, it is designated as "FALL".

After cleaning the data, we did EDA to get some basic ideas about our dataset. Then we used the tokenizing matrix as input to train and test models and evaluated those models through accuracy, f1-score, and confusion matrices.

## Github Content Guide

**Code:** All analysis is coded using Python. All visualizations are included in the _visualization_ folder. 

**Data:** Raw data and cleaned data. 

**Presentation Slide:** Presentation slides include detailed analysis and results for this project.

## Coding References: 

Code from ANLY 580 Labs was used to complete this project along with internet resources. Following are a few websites that were referred to for the coding section.

1. https://mccormickml.com/2019/07/22/BERT-fine-tuning/ 

2. https://huggingface.co/bert-base-uncased




