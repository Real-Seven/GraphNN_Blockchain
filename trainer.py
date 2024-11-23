from  augmentation import Augmentation
from  metricsG import GraphMetrics
from ellipticData import Dataset
import model
import utils

from torch_geometric.data  import DataLoader

import numpy as np
import torch
import os.path as osp
import sys
import glob
import time

"""
Utility class for training models during train/valid phase only
Loads the augmented data, builds the network,
trains the model. Saves the state dictionaries.

"""

def train_validate(args):
    """
    Parameter Search
    """
    timer_start = time.perf_counter()

    t = Trainer(args)
    t.train()
    t.search_best_epoch()
    timer_end = time.perf_counter()

    utils.write_timings(args,timer_end - timer_start)
        


class Trainer:
    
    """
    A utility class for training
    """

    def __init__(self, args):
        self._args = args
        self._init()

    def _init(self):
        args = self._args
        self._device = torch.device('cpu')
        self._aug = Augmentation(method=args.aug)
        self._tau= 0.99

        self._modelid=utils.get_global_model_id(args)
        self._dataset = Dataset(root="./data" ,augumentation=self._aug)
        self._loader = DataLoader(dataset=self._dataset) 
        self._metricsG=GraphMetrics(args,self._modelid)

        hidden_layers = [int(l) for l in args.layers]
        layers = [self._dataset.data.x.shape[1]] + hidden_layers

        self._model = model.GBYOL(
            layer_config=layers, 
            dropout=args.dropout,  
            gnn_type=args.model,
            tau=self._tau
        ).to(self._device)
        
        print(f"ModelID:  {self._modelid}")
        print(self._model)
        
        #scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.01, patience=20, verbose=True)
        pos_weight=torch.tensor([420/45])  
        self._criterion = torch.nn.BCEWithLogitsLoss( pos_weight=pos_weight)
        self._optimizer = torch.optim.Adam(
            params=self._model.parameters(), lr=args.lr, weight_decay=1.5e-6)
        
      
    def load_best_model(self, filename):
        path=osp.join(self._dataset.model_dir, filename)
        self._model.load_state_dict(torch.load(path, map_location=self._device), strict=False)


    def train(self):
        """
        Training the cross-applied network
        """
        early_stopping = EarlyStopping( patience=5, min_delta=1e-5)
        self._model.train()
        for epoch in range(self._args.epochs):
            for bc, batch_data in enumerate(self._loader):
                batch_data.to(self._device)
     
                v1_output, v2_output, loss = self._model(
                    x1=batch_data.x, x2=batch_data.x2, edge_index_v1=batch_data.edge_index, edge_index_v2=batch_data.edge_index2,
                    edge_weight_v1=batch_data.edge_attr, edge_weight_v2=batch_data.edge_attr2)
                self._optimizer.zero_grad() 
                loss.backward() 
                self._optimizer.step()

                # Update the target network with a moving average
                self._model.update_moving_average()
                sys.stdout.write('\rEpoch {}/{}, batch {}, loss {:.4f}'.format(epoch + 1, self._args.epochs, bc + 1, loss.data))
                sys.stdout.flush()
      
            if (epoch + 1) % 10 == 0:
                path = osp.join(self._dataset.model_dir,f"{self._modelid}.ep.{epoch + 1}.pt")
                torch.save(self._model.state_dict(), path)
            
            early_stopping(loss.data.item())
            if early_stopping.early_stop:
                print("Early stopping ***** epoch: " , epoch + 1)
                path = osp.join(self._dataset.model_dir,f"{self._modelid}.ep.{epoch + 1}.pt")
                torch.save(self._model.state_dict(), path)
                break
            
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': self._model.state_dict(),
            #     'loss': loss.data
            #     }, path)
           

        print()        
        
    def infer_embeddings(self):
        """
        Infers node embeddings from the model. 
        """
        
        #outputs = []
        self._model.train(False)
        self._embeddings = self._labels = None
        self._train_mask = self._val_mask = self._test_mask = None
        
        for bc, batch_data in enumerate(self._loader):
            batch_data.to(self._device)
            v1_output, v2_output, _ = self._model(
                x1=batch_data.x, x2=batch_data.x2,
                edge_index_v1=batch_data.edge_index,
                edge_index_v2=batch_data.edge_index2,
                edge_weight_v1=batch_data.edge_attr,
                edge_weight_v2=batch_data.edge_attr2)
            
            emb = torch.cat([v1_output, v2_output], dim=1).detach()

            y = batch_data.y.detach()

            train_m= batch_data.train_mask.detach()
            valid_m = batch_data.val_mask.detach()  
            test_m = batch_data.test_mask.detach()
            if self._embeddings is None:
                self._embeddings, self._labels = emb, y
                self._train_mask, self._val_mask, self._test_mask = train_m, valid_m, test_m
            else:
                self._embeddings = torch.cat([self._embeddings, emb])
                self._labels = torch.cat([self._labels, y])
               
                self._train_mask = torch.cat([self._train_mask, train_m])
                self._val_mask = torch.cat([self._val_mask, valid_m])
                self._test_mask = torch.cat([self._test_mask, test_m])


    def search_best_epoch(self):
        """
        Searches for the best epoch that leads to the best validation accuracy. 
        Used for hyperparameter tuning
        
        """

        pattern=self._dataset.model_dir + "/"+ self._modelid + ".ep.*.pt"
        print(pattern)
        model_files = glob.glob(pattern)
 
        results = []
        best_epoch = -1, (0,), (0,), (0,) 


        for i, model_file in enumerate(model_files):
            if model_file.endswith(".pt"):
                substr = model_file.split(".")
                epoch = int(substr[substr.index("ep") + 1])

                train_metrics, valid_metrics, _ = self.epoch_validation(epoch)
                
                results.append([epoch, train_metrics, valid_metrics, _])

                if train_metrics[0] > best_epoch[1][0]:  ## looking for illicit_f1, on training data
                    best_epoch = epoch, train_metrics, valid_metrics, _

        res={}
        res['epoch']  = best_epoch[0]
        res['train'] =  best_epoch[1]
        res['valid'] =  best_epoch[2]


        print(f"The best epoch is: {best_epoch[0]}") 
        print(f"with training illicit f1: {res.get('train')[0] } std: {res.get('train')[1]}")
        print(f"with valid illicit f1: {res.get('valid')[0] } std: {res.get('valid')[1]}")
    
        path = osp.join(self._dataset.result_dir, "training_results_v3.txt")
       
        line= f"{self._modelid},{res['epoch']},{res['train']},{res['valid']}\n"
        with open(path, "a") as training_file:
            training_file.write(                line                )

    def epoch_validation(self, epoch):
        """
        Evaluates SelfGNN saved after a given training epoch
        :param epoch: The epoch to be evaluated
        :return: The validation and test accuracy of the model saved at a given epoch
        """
        train_metrics=[]
        valid_metrics=[]
        test_metrics=[]
        path = osp.join(self._dataset.model_dir, f"{self._modelid}.ep.{epoch}.pt")
        self._model.load_state_dict(torch.load(path, map_location=self._device))
        self.infer_embeddings()
        train_metrics = self.evaluate_semi(split="train")
        valid_metrics = self.evaluate_semi(split="valid")
        test_metrics= self.evaluate_semi(split="test")
        return train_metrics, valid_metrics, test_metrics

    def evaluate_semi(self, split="valid"):
        """
        Evaluate against the test/train/valid splits
        """
        if split == "train":
            mask = self._train_mask
        elif split == "valid":
            mask = self._val_mask
        else:
            mask = self._test_mask
        
        features = self._embeddings[mask].detach().cpu().numpy()
        labels = self._labels[mask].detach().cpu().numpy()

        return self._metricsG.get_validation_metrics(features, labels,split)
        

class EarlyStopping:
    def __init__(self, mode='min', patience=5, min_delta=0):
        """
        Args:
            patience (int): How long to wait after last time  loss improved.
                            Default: 5
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                                Default: 0
        """
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.min_delta *= 1 if self.mode == 'min' else -1

    def __call__(self, metric):
        if self.best_score is None:
            self.best_score = metric
        elif (self.mode == 'min' and metric > self.best_score - self.min_delta) or \
             (self.mode == 'max' and metric < self.best_score + self.min_delta):
            self.counter += 1
            print(metric,self.best_score )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = metric
            self.counter = 0
     
