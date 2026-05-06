# Phase 2 - Classical ML Models

This phase builds on the Phase 1 feature vectors and tries a few normal machine learning models instead of just the rule-based Arch Index cutoff.

What I learned the hard way:
- Accuracy by itself is not enough for a medical-ish problem
- Sensitivity matters because missing a flatfoot case is a bad failure
- Specificity matters because I do not want to keep calling normal feet flatfoot
- Subject-independent splitting matters because trial-level random splits can make the result look better than it really is

Files in this folder:
- `dataset_builder.py` scans a folder of CSV files, runs the Phase 1 extraction, and writes a tabular dataset
- `svm_model.py` trains an RBF SVM with StandardScaler and permutation importance
- `random_forest_model.py` trains a 100-tree random forest and saves feature importance plots
- `knn_model.py` sweeps k from 3 to 11 and plots the elbow curve
- `logistic_model.py` trains an interpretable logistic regression model and prints coefficients
- `evaluate.py` saves the confusion matrix, ROC curve, and a text summary
- `common.py` is just shared glue code so I did not repeat the same split logic everywhere

Typical workflow:

```powershell
python dataset_builder.py --folder "../data/raw" --output "./output/features_dataset.csv"
python svm_model.py --dataset-csv "./output/features_dataset.csv"
python random_forest_model.py --dataset-csv "./output/features_dataset.csv"
python knn_model.py --dataset-csv "./output/features_dataset.csv"
python logistic_model.py --dataset-csv "./output/features_dataset.csv"
```

The shared evaluator prints:
- confusion matrix
- accuracy
- sensitivity
- specificity
- F1 score
- AUC

I kept the scripts separate instead of making one huge trainer because that was easier to compare while I was experimenting.
