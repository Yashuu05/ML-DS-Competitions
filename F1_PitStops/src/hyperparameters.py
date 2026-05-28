LogisticRegression={
    "model__c":[0.001, 0.01, 0.1, 10, 100],
    "model__penalty":["11", "12"],
    "model__solver":["liblinear", "saga"],
    "model__max_iter":[1000,2000]
}

RandomForest={
    "model__n_estimators":[
        100,200,300
    ],
    "model__max_depth":[7,10,20],
    "model__min_samples_split":[2,5,10],
    "model__min_samples_leaf":[1,2,4],
}

LightGBM={
    "model__learning_rate":[0.01, 0.05, 0.1],
    "model__num_leaves":[31,50,100],
    "model__n_estimators":[100,200,500],
    "model__boosting_type":["gbdt","dart"]
}