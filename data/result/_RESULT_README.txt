This folder contains the results of validation and evaluation tests. I have left them here since they are small.

1. FULL_Training_results_v3.txt
This is a record of all the best epochs for each hyperparameter combination.  
It includes the training and validation metrics, which are compared to check for over/under fitting.

2. BEST_Training_results.txt
The chosen model to evaluate, based on overall performance during training.

3. EllipticEvaluationResults.txt
When the {modelid} was evaluated against each of the downstream {classifier} (logistic , randomforest etc), the metrics for that model is saved in this file.
