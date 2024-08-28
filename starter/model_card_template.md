# Model Card

This model was trained on the census.csv dataset. The goal of the model is to predict whether the income of a person is higher or lower than 50k dollars based on the attributes of the person. The model will return value 1 if it is the case, otherwise it will return value 0.


## Model Details

The model that was trained on the census.csv data is a RandomForest Classifier with basic parameters and random state equal to 42.

## Intended Use

The intended use of the model is to perform a request to a fast API application which will return 0 or 1 depending on the fact whether is the income is higher than 50k dollars or not. The input should look as follows:

"age": 23,
"workclass": "Private",
"education": "HS-grad",
"maritalStatus": "Divorced",
"occupation": "Other-service",
"relationship": "Own-child",
"race": "Black",
"sex": "Male",
"hoursPerWeek": 30,
"nativeCountry": "United-States"



## Training Data

The used training data is the census.csv file which was cleaned before using it for model training. All spaces were removed and the data which wasn't complete was removed from the dataset.

## Evaluation Data

The evaluation data was obtained with the train_test_split functions from sklearn. This evaluation dataset contains 20% of the original dataset.

## Metrics
The used metrics are precision, recall and fbeta. On the test dataset the results are as follows:
precision: 0.7498081350729087
recall: 0.6385620915032679
fbeta: 0.6897282033180374

## Ethical Considerations

Ethical considerations with this ML model are linked to collecting new data for model inference: Certainly in Europe privacy legislations and the European AI act have to be taken in consideration.

## Caveats and Recommendations

If you would like to use another ML model you can change this in the train_model function. LogisticRegression or SVM can be easily implemented with the provided code.

The data is heavily skewed which has an influence on model slice performance. This should be taken into account during model inference.


