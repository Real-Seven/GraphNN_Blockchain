from  augmentation import Augmentation
from  metricsG import GraphMetrics
from ellipticData import Dataset
import model
import utils
import time
from sklearn.preprocessing import MinMaxScaler 
import numpy as np
import torch
import os.path as osp


"""
Utility class for evaluating models during test phase only
Loads the augmented data, representations (using state dictionaries)
Chooses the downstream model (Random Forest etc)
Asseess against provided data

"""
        
def test_evaluate(args, best_model):
    """
    Evaluation
    """
    eval = Evaluator(args)
    eval.load_best_model(best_model)
    eval.load_embeddings()
    eval.evaluate()

def final_evaluate(args, best_model):
    """
    Evaluation
    """
    eval = Evaluator(args)
    eval.load_best_model(best_model)
    eval.evaluate_fullset("Logistic")

class Evaluator:
    
    """
    A utility class for evaluating models against each classifier
    """

    def __init__(self, args):
        self._args = args
        self._init()

    def _init(self):
        args = self._args
        self._device = torch.device('cpu')
        self._aug = Augmentation(method=args.aug)
        self._noaug = Augmentation(method="none")

        self._modelid=utils.get_global_model_id(args)
        self._dataset = Dataset(root="./data" ,augumentation=self._aug)
        self._originalDataset = Dataset(root="./data" ,augumentation=self._noaug)
        self._metricsG=GraphMetrics(args,self._modelid)

        hidden_layers = [int(l) for l in args.layers]
        layers = [self._dataset.data.x.shape[1]] + hidden_layers
        #only want the encodings from earlier in the model without the predictor
        self._model=model.Encoder(layer_config=layers, gnn_type=args.model, dropout=args.dropout).to(self._device)
        print(self._model)
        print(f"ModelID:  {self._modelid}")
      
    def load_best_model(self, filename):
        path=osp.join(self._dataset.model_dir, filename)
        #  warm starting the model, with strict=False
        self._model.load_state_dict(torch.load(path, map_location=self._device), strict=False)
 
        
    def load_embeddings(self):
        """
        Extract the embeddings from the saved model.
        Set the Test and Train sets for representations
        Set the Test and Train sets for original features
        Use MinMaxScaler to bring all features to [0,1] for trad models mostly
        """
        
        #outputs = []
        self._model.eval()
        self._X_train = self._Y_train = None
        self._X_test = self._Y_test= None
        self._X_trainOrig = self._X_testOrig =None
        scaler = MinMaxScaler()
        
        augmented_data=self._dataset.to(self._device)
        # only the embeddings from the online network are used, as per BYOL
        onlineX, _= self._model(x=augmented_data.x,
            edge_index=augmented_data.edge_index,
            edge_weight=augmented_data.edge_attr)

        emb = onlineX.detach()
    
        Y = augmented_data.y.detach()
        emb_dim, _ = emb.shape[1], Y.unique().shape[0]

        train_m = augmented_data.train_mask.detach()
        valid_m = augmented_data.val_mask.detach()  
        test_m  = augmented_data.test_mask.detach()

        ## put train and validation sets togerth
        train_mask=torch.logical_or(train_m, valid_m) 
        test_mask=test_m

        # test train split of embeddings
        self._X_train = emb[train_mask]
        self._X_test = emb[test_mask]
        self._X_train = scaler.fit_transform(self._X_train)
        self._X_test= scaler.transform( self._X_test)

        
        #  test train split of Labels
        self._Y_train = Y[train_mask]
        self._Y_test = Y[test_mask]


        #test train split of original features
        self._X_trainOrig = self._originalDataset.x[train_mask]
        self._X_testOrig = self._originalDataset.x[test_mask]

        self._X_trainOrig = scaler.fit_transform(self._X_trainOrig)
        self._X_testOrig= scaler.transform(self._X_testOrig)


        self._embeddings = emb
        self._originalX= self._originalDataset.x

 

    def evaluate(self):
        """
        This utility does a loop against each type of evaluator
        It also adds the results using original features
        """
        print("Evaluating ...")
        for modelname in ['KNN', 'Saga', 'IsolationForest', 'NaiveBayes','XGBClassifier', 'RandomForest','Logistic']:
            #aug_metrics = orig_metrics=[]
            tmean_aug=tstd_aug=tmean_orig=tstd_orig=0.0
            
            # Evaluation for each type of model
            aug_metrics, orig_metrics=self.evaluate_model(modelname=modelname)
            
            # rsults on representations
            tmean_aug=np.mean(aug_metrics, axis=0).tolist()
            tstd_aug=np.std(aug_metrics,axis=0).tolist()

            #results on original features
            tmean_orig=np.mean(orig_metrics, axis=0).tolist()
            tstd_orig=np.std(orig_metrics,axis=0).tolist()

            # save the models to file
            filename=f"EllipticEvaluationResults.txt"
            path = osp.join(self._dataset.result_dir, filename)
            with open(path, 'a') as f:
                f.write(f"{self._modelid}\
                        ,{modelname}\
                        ,{tmean_aug[0]:.5f},{tstd_aug[0]:.5f}\
                        ,{tmean_aug[1]:.5f},{tstd_aug[1]:.5f}\
                        ,{tmean_aug[2]:.5f},{tstd_aug[2]:.5f}\
                        ,{tmean_aug[3]:.5f},{tstd_aug[3]:.5f}\
                        ,{tmean_orig[0]:.5f},{tstd_orig[0]:.5f}\
                        ,{tmean_orig[1]:.5f},{tstd_orig[1]:.5f}\
                        ,{tmean_orig[2]:.5f},{tstd_orig[2]:.5f}\
                        ,{tmean_orig[3]:.5f},{tstd_orig[3]:.5f}\n")      
       
            print(f'{modelname} :: Reps f1: { tmean_aug[0]:.5f} std: { tstd_aug[0]:.5f} : Orig f1: {tmean_orig[0]:.5f} std: {tstd_orig[0]:.5f}'
                                 f'  Reps acc: { tmean_aug[1]:.5f} std: { tstd_aug[1]:.5f} : orig acc: {tmean_orig[1]:.5f} std: {tstd_orig[1]:.5f}'
                            )

         

    def evaluate_model (self, modelname):
        test_metrics_reps=[]
        test_metrics_orig=[]
                    
        iterations=10

        if  modelname in('IsolationForest','XGBClassifier','RandomForest', 'Logistic', 'NaiveBayes','KNN', 'NaiveBayes' , 'Saga'):
  
            test_metrics_reps=self.fit_and_predict(modelname,reps=True, iterations=iterations)
            test_metrics_orig=self.fit_and_predict(modelname,reps=False ,iterations=iterations)

        # Some models do not experience variance so could just run once
        # However for completeness, many runs are carried out instead
        # Straitified Kfold was intitially used, but used too much resource/time
            
        # elif modelname in( 'KNN', 'NaiveBayes' , 'Saga'):
        #     std of these are 0 , therefore iiteration can be 1
        #     test_metrics_reps=self.fit_and_predict(modelname,reps=True, iterations=1)
        #     test_metrics_orig=self.fit_and_predict(modelname,reps=False ,iterations=1)


        else:
            print ('No Model selected')
    
        return test_metrics_reps   , test_metrics_orig 
         

    def fit_and_predict (self,modelname, reps=True, iterations=2):
        metrics=[]

        # labels remain static
        Ytrain=self._Y_train
        Ytest=self._Y_test

        if reps: # representations
            Xtrain=self._X_train
            Xtest=self._X_test
            writemodel=modelname+"reps"

        else: # Original features
            Xtrain=self._X_trainOrig
            Xtest=self._X_testOrig
            writemodel=modelname+"orig"

        timer_start = time.perf_counter()
        for i in range(iterations): 
            clf=model.get_evaluation_model (modelname)
            clf.fit(Xtrain, Ytrain)
            Ypred= clf.predict (Xtest)

            if modelname=='IsolationForest':
                Ypred[Ypred == 1] = 0
                Ypred[Ypred == -1] = 1    
            mets=self._metricsG.get_evaluation_metrics(Ypred, Ytest,phase='test') 
            metrics.append(mets)
        timer_end = time.perf_counter()
        utils.write_timings_eval(writemodel,timer_end - timer_start)

        return metrics   

    def evaluate_model_logistic (self, modelname):
        test_metrics_reps=[]
        test_metrics_orig=[0,0,0,0,0,0,0,0]
                    
        iterations=10

        if  modelname in('Logistic'):
  
            test_metrics_reps=self.fit_and_predict(modelname,reps=True, iterations=iterations)
            return test_metrics_reps   , test_metrics_orig                                                                                

    def evaluate_fullset(self, modelname):
        #outputs = []
        modelname="Logistic"
        self._model.eval()
        self._X = self._Y = None
        scaler = MinMaxScaler()
        

        # The final score. Train on the training set, and evaluate on the fullset.
        augmented_data=self._dataset.to(self._device)
        # only the embeddings from the online network are used, as per BYOL
        onlineX, _= self._model(x=augmented_data.x,
            edge_index=augmented_data.edge_index,
            edge_weight=augmented_data.edge_attr)

        emb = onlineX.detach()
    
        Y = augmented_data.y.detach()
        train_m = augmented_data.train_mask.detach()
        valid_m = augmented_data.val_mask.detach()  
        test_m  = augmented_data.test_mask.detach()
        ## put train and validation sets togerth
        train_mask=torch.logical_or(train_m, valid_m) 
        test_mask=test_m
        # test train split of embeddings
        self._X_train = emb[train_mask]
        self._X = emb
        self._X_train = scaler.fit_transform(self._X_train)
        self._X = scaler.transform( self._X )
        #  test train split of Labels
        self._Y_train = Y[train_mask]
        self._Y=Y

        self._embeddings = emb
        tmean_aug=tstd_aug=tmean_orig=tstd_orig=0.0
        metrics=[]


        for i in range(10): 
            clf=model.get_evaluation_model (modelname)
            clf.fit(self._X_train, self._Y_train)
            Ypred= clf.predict (self._X)
            mets=self._metricsG.get_evaluation_metrics(Ypred, self._Y,phase='test') 
            metrics.append(mets)

        
        # rsults on representations
        tmean_aug=np.mean(metrics, axis=0).tolist()
        tstd_aug=np.std(metrics,axis=0).tolist()

        
       
        print(f'{modelname} :: Reps f1: { tmean_aug[0]:.5f} std: { tstd_aug[0]:.5f}'
                                 f'  Reps acc: { tmean_aug[1]:.5f} std: { tstd_aug[1]:.5f} '
                            )

    