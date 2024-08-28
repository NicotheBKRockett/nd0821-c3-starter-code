

filename = "my_model.pickle"
    model = pickle.load(open(filename, "rb"))
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    data = pd.DataFrame(data)
    print(data)
    X, y, encoder, lb = process_data(data, categorical_features=[], label=None, training=False, encoder=None, lb=None)
    print(X)
    predictions = inference(model, X)