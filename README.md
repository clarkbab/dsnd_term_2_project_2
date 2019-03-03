# Disaster Response Pipeline Project

### Description

This project uses supervised learning techniques to classify tweets sent during natural disasters. We can run new tweets through our model and get 

### Instructions:
We first must run an ETL pipeline to clean our data and store it in a database. To do so, run:

```
$ python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

We then run a machine learning pipeline on the data, to train our classifier. This also saves the trained classifier to a `.pkl` file.

```
$ python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

There is also a web application that can predict message categories, run it with:

```
$ python app/run.py
```

And view the application by visiting `localhost:3001` from your browser.
