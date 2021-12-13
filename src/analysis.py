# Imports
import pandas as pd
import numpy as np
import pydotplus
import collections

# sklearn packages
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Load the files
tweet_scores = pd.read_csv('data/scored-tweets.csv', sep=';')
stock_price = pd.read_csv('data/AAPL.csv')

# convert the timestamp column to a datatime type
tweet_scores.timestamp = pd.to_datetime(tweet_scores.timestamp)
# only need the day part of the datetime
tweet_scores['Date'] = pd.to_datetime(tweet_scores.timestamp.dt.date)

# Convert Date column for stock price too
stock_price['Date'] = pd.to_datetime(stock_price.Date)

# todays open minus yesterdays open
stock_price['open_change'] = stock_price.Open - stock_price.Open.shift(1) 
# Generate the difference between last nights close and today's open
stock_price['overnight_change'] = stock_price.Open - stock_price.Close.shift(1)

# these catagories will be used for both tweet sentiment classes and stock price change classes, but with diffrent cutoffs
labels = ['very_negative', 'negative', 'neutral', 'positive', 'very_positive']

# Label the tweet sentiment scores into the 5 classes. 
tweet_scores['sentiment_cat'] =pd.cut(
    tweet_scores.sentiment_score,
    [-np.inf,-.5, -.01, .01, .5, np.inf],
    labels=labels
)

# I want totals of each sentiment class per day. I make 5 columns, one for each class
# Then put a 1 in the column each tweet's sentiment class falls into and a 0 in the others
# This lets me just sum the columns when I group by day later
for label in labels:
    tweet_scores[label] = tweet_scores.sentiment_cat.apply(lambda x: 1 if x ==label else 0)

# get sentiment counts per day, this takes advantage of the step above
sentiment_by_day = tweet_scores.groupby([tweet_scores.timestamp.dt.date]).sum()

# the group by makes an index, I don't need it
sentiment_by_day.reset_index(level=0, inplace=True)
# add one day to the scores, this makes yesterdays tweets, todays sentiment when it is joined in the next step
sentiment_by_day['yesterday_timestamp'] =  pd.to_datetime(sentiment_by_day.timestamp) + pd.Timedelta(days=1)
# join to stock price
stock_price_w_score =pd.merge(stock_price, sentiment_by_day, left_on="Date", right_on="yesterday_timestamp")

# Don't need sentiment_score or tweet_id
stock_price_w_score = stock_price_w_score.drop('tweet_id',axis=1)
stock_price_w_score = stock_price_w_score.drop('sentiment_score',axis=1)

# Classify the overnight change into the same 5 classes, but with diffrent cutoffs
stock_price_w_score['price_change_cat'] = pd.cut(
    stock_price_w_score.overnight_change,
    [-np.inf,-1, -.01, .01, 1, np.inf],
   labels=labels
)

# set the prediction column and the feature columns for modeling
prediction_col = 'price_change_cat'

# the totaled tweet classes per day are the feature columns
feature_cols = labels

# pull out x and y
x = stock_price_w_score[feature_cols].values
y = stock_price_w_score[prediction_col].values

#split the dataset into the train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

# Decision tree
tree_model = tree.DecisionTreeClassifier()
tree_model.fit(x_train, y_train)

# Random forest
forest_model = RandomForestClassifier()
forest_model.fit(x_train, y_train)

# visualize the decision tree
# This comes from the week 4 lecture of MSDS680_C70 Machine Learning Regis course
dot_data = tree.export_graphviz(tree_model,
                                feature_names=feature_cols,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)
colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)
for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('out/stock_tree.png')

#gathering the predictions
tree_preds = tree_model.predict(x_test)
print ("Tree model accuracy score: ", accuracy_score(y_test,tree_preds))

#gathering the predictions
forest_preds = forest_model.predict(x_test)
print ("Forest model score: " , forest_model.score(x_test,y_test))

# write out the stock price with sentiment score for making some graphs in excel
stock_price_w_score[['Date','price_change_cat']].to_csv('out/Stock_price_change.csv')
# write out sentiment by day also to make graphs
sentiment_by_day.to_csv('out/sentiment_by_day.csv')

# Write out decision tree confusion matrix
plt.subplots(figsize=(6, 6))
mat = confusion_matrix(y_test, tree_preds, labels=labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=labels, yticklabels=labels)
plt.xlabel('True Change')
plt.ylabel('Predicted Change')
plt.savefig('out/tree_model_confusion.png', dpi=300)

mat = confusion_matrix(y_test, forest_preds, labels=labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=labels, yticklabels=labels)
plt.savefig('out/forest_model_confusion.png', dpi=300)

# as discussed in the presentation, I decided to see if the models above do any better than the meaningless 
# approach of using tomorrows tweet sentiment to predict today's price. If they both preform about the same, 
# then using sentiment to predict stock is not much better than random
# write out random forest confusion matrix

# subtract one day from the scores timestamp, this makes tomorrows tweets, todays sentiment
sentiment_by_day['tomorrows_timestamp'] =  pd.to_datetime(sentiment_by_day.timestamp) + pd.Timedelta(days=-1)
# join to stock price
stock_price.Date = pd.to_datetime(stock_price.Date)
stock_price_w_score_tomorrow =pd.merge(stock_price, sentiment_by_day, left_on="Date", right_on="tomorrows_timestamp")
stock_price_w_score_tomorrow = stock_price_w_score_tomorrow.drop('tweet_id',axis=1)
stock_price_w_score_tomorrow = stock_price_w_score_tomorrow.drop('sentiment_score',axis=1)
stock_price_w_score_tomorrow['price_change_cat'] = pd.cut(
    stock_price_w_score_tomorrow.overnight_change,
    [-np.inf,-1, -.01, .01, 1, np.inf],
   labels=labels
)

# this will result in the row to contain NaN, so drop it
stock_price_w_score_tomorrow = stock_price_w_score_tomorrow.dropna()

# set the prediction column and the feature columns for tree
prediction_col = 'price_change_cat'

feature_cols = labels
x = stock_price_w_score_tomorrow[feature_cols].values
y = stock_price_w_score_tomorrow[prediction_col].values

#split the dataset into the train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

# Decision tree
tree_model = tree.DecisionTreeClassifier()
tree_model.fit(x_train, y_train)

# Random forest
forest_model = RandomForestClassifier()
forest_model.fit(x_train, y_train)

#gathering the predictions
tree_preds = tree_model.predict(x_test)
print ("Tree model accuracy score, using tomorrows tweets: ", accuracy_score(y_test,tree_preds))

#gathering the predictions
forest_preds = forest_model.predict(x_test)
print ("Forest model score, using tomorrows tweets: : " , forest_model.score(x_test,y_test))

# Write out decision tree confusion matrix
plt.subplots(figsize=(6, 6))
mat = confusion_matrix(y_test, tree_preds, labels=labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=labels, yticklabels=labels)
plt.xlabel('True Change')
plt.ylabel('Predicted Change')
plt.savefig('out/tomorrow_tree_model_confusion.png', dpi=300)

mat = confusion_matrix(y_test, forest_preds, labels=labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=labels, yticklabels=labels)
plt.savefig('out/tomorrow_forest_model_confusion.png', dpi=300)


# Next I tried some k-means clustering.

# calculate the next 5 days offsets 
stock_price['Date-1'] = pd.to_datetime(stock_price.Date)- pd.Timedelta(days=1)
stock_price['Date-2'] = pd.to_datetime(stock_price.Date)- pd.Timedelta(days=2)
stock_price['Date-3'] = pd.to_datetime(stock_price.Date)- pd.Timedelta(days=3)
stock_price['Date-4'] = pd.to_datetime(stock_price.Date)- pd.Timedelta(days=4)
stock_price['Date-5'] = pd.to_datetime(stock_price.Date)- pd.Timedelta(days=5)

# Now I can join each tweet to the open_change value for the next 5 days
tweet_scores['open_change+1'] = pd.merge(tweet_scores, stock_price[['open_change', 'Date-1']], left_on='Date', right_on='Date-1')['open_change']
tweet_scores['open_change+2'] = pd.merge(tweet_scores, stock_price[['open_change', 'Date-2']], left_on='Date', right_on='Date-2')['open_change']
tweet_scores['open_change+3'] = pd.merge(tweet_scores, stock_price[['open_change', 'Date-3']], left_on='Date', right_on='Date-3')['open_change']
tweet_scores['open_change+4'] = pd.merge(tweet_scores, stock_price[['open_change', 'Date-4']], left_on='Date', right_on='Date-4')['open_change']
tweet_scores['open_change+5'] = pd.merge(tweet_scores, stock_price[['open_change', 'Date-5']], left_on='Date', right_on='Date-5')['open_change']

# The last five days of stock data end up with NaNs because the data set cuts off. 
# this results in some NaNs
tweet_scores = tweet_scores.dropna()
tweets_to_cluster = tweet_scores[['sentiment_score','open_change+1','open_change+2','open_change+3','open_change+4','open_change+5']]

# cluster
i=10
random_state=34
tweet_scores['cluster'] = KMeans(n_clusters=i, random_state=random_state).fit_predict(tweets_to_cluster).tolist()

# write out the cluster numbers for manual analysis
tweets_to_cluster.to_csv('out/clustered_tweets.csv')

# Look at cluster counts per day to see if cluster just matches by day. Some do
print (tweet_scores[['Date','cluster','tweet_id']].groupby(['Date','cluster']).count())

# That's as far as I got with the clustering

