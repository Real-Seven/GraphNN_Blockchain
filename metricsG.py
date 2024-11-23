from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, roc_curve
#from sklearn.metrics import precision_recall_curve, auc, accuracy_score

from sklearn.linear_model import LogisticRegression 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit

import numpy as np
import torch

class GraphMetrics:


        """
        A utility class Metrics
        """

        def __init__(self, args, _modelid):
                self._args = args
                self._init()

        def _init(self):
                #args = self._args
                self._device = torch.device('cpu')
                #self._modelid=args._modelid



        def get_validation_metrics(self, features, labels, phase='valid'):
            """
            Evaluate metrics
            """
            sf = ShuffleSplit(5, test_size=0.4, random_state=0)
            clf = OneVsRestClassifier(
                         LogisticRegression(solver='liblinear'), n_jobs=1)# originally -1, but parallel caused issues.
            f1_micro_res = []
            f1_illict_res = []
            recall_illicit_res = []
            precision_illicit_res = []
            features = StandardScaler().fit_transform(features)

            for train_index, test_index in sf.split(features, labels):
                train_x = features[train_index]
                train_y = labels[train_index]
                test_x = features[test_index]
                test_y = labels[test_index]
                
                # fit the logistic regression model
                clf.fit(train_x, train_y)
                pred = clf.predict(test_x)


                f1_illict=f1_score(test_y, pred, average='binary', pos_label=1, zero_division=0)  # Illicit
                f1_micro=f1_score(test_y, pred, average='micro', zero_division=0)  # Accuracy
                recall_illicit = recall_score(test_y, pred, average='binary',pos_label=1, zero_division=0)        # For positive class
                precision_illicit = precision_score(test_y, pred, average='binary',pos_label=1, zero_division=0)  # For positive class

                f1_illict_res.append(f1_illict)
                f1_micro_res.append(f1_micro)
                recall_illicit_res.append(recall_illicit)
                precision_illicit_res.append(precision_illicit)

            return np.mean(f1_illict_res), np.std(f1_illict_res) \
                ,np.mean(f1_micro_res), np.std(f1_micro_res) \
                ,np.mean(recall_illicit_res), np.std(recall_illicit_res) \
                ,np.mean(precision_illicit_res), np.std(precision_illicit_res)





        def get_evaluation_metrics (self, labels, pred, phase='test'):
                f1_illict=f1_score(labels, pred, average='binary', pos_label=1, zero_division=0)  # Illicit
                f1_micro=f1_score(labels, pred, average='micro', zero_division=0)  # Accuracy
                recall_illicit = recall_score(labels, pred, average='binary',pos_label=1, zero_division=0)       
                precision_illicit = precision_score(labels, pred, average='binary',pos_label=1, zero_division=0)  
                if phase=='test':
                        # find best classification threshold
                        fpr, tpr, thresholds = roc_curve(labels, pred,  pos_label=1) 
                        J = tpr - fpr
                        ix = np.argmax(J)

                        best_thresh = thresholds[ix]
                        print(f"Best ovo Threshold for classificatio from this test: {best_thresh}")
                return [f1_illict, f1_micro, recall_illicit, precision_illicit]


        # def get_metrics_torch (self, labels, preds, phase='valid'):
        #         mets = {
        #                 "f1": f1,
        #                 "f1_micro": f1_micro,
        #                 "recall_illicit": recall_illicit,
        #                 "precision_illicit": precision_illicit
        #                 }
