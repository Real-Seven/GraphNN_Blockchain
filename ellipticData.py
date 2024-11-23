import os
import pandas as pd
import torch
from typing import List
#import os.path as osp
import numpy as np
import torch.nn.functional as F


from torch_geometric.data import Data, InMemoryDataset

from tqdm import tqdm

class Dataset(InMemoryDataset):

    """
    When called 
    1. Identify raw files
    2. Check o/s for pytoch structure graph data, with the correct augmentations
    3. If file exists, load and return the data
    4. If not exists, load raw files, generate augs, generate graph, return clean and aug data

    """
    def __init__(self, root=".\\data",   augumentation=None, transform=None,
                pre_transform=None, force_reload=False):


        self.augumentation = augumentation
        super().__init__(root, transform, pre_transform,force_reload)
        self.load(self.processed_paths[0])

    @property
    def num_classes(self) -> int:
        return 1

    @property
    def processed_file_names(self) -> str:
        pre_augmented='EllipticData_'+self.augumentation.method +'.pt'
        return pre_augmented
    
    @property
    def raw_file_names(self) -> List[str]:
        return [
            'elliptic_txs_features.csv',
            'elliptic_txs_edgelist.csv',
            'elliptic_txs_classes.csv',
        ]

    @property
    def model_dir(self):
        if not os.path.exists(os.path.join(self.root, "model")): 
            os.makedirs(os.path.join(self.root, "model")) 
        return os.path.join(self.root, "model")

    @property
    def result_dir(self):
        if not os.path.exists(os.path.join(self.root, "result")): 
           os.makedirs(os.path.join(self.root, "result")) 

        return os.path.join(self.root, "result")

    @property
    def dirs(self):
        return [self.root, self.raw_dir, self.processed_dir, self.model_dir, self.result_dir]


    def process(self) :
    
        vanilla=self.csv_to_torch()

        data=self.add_augmentation(vanilla)
 
        self.save([data], self.processed_paths[0])


    def csv_to_torch(self):
        f_path=self.raw_dir
        print (f_path)

        f_feat = os.path.join(f_path, self.raw_file_names[0])
        f_edge = os.path.join(f_path, self.raw_file_names[1])
        f_classes =  os.path.join(f_path, self.raw_file_names[2])

        # Load the elliptic data from filesystem
        try:
            edge_df = pd.read_csv(f_edge) # skiprows=1
            feat_df = pd.read_csv(f_feat , header=None)
            class_df= pd.read_csv(f_classes)
            print("Files loaded successfully.")
        
        except Exception as e:
            print(f"An error occurred during file load: {e}")

        # Transform the data into torch_geometric.data
        columns = {0: 'txId', 1: 'time_step'}
        feat_df = feat_df.rename(columns=columns)

        # take only local features, no time step. Total93 features
        x = torch.from_numpy(feat_df.loc[:, 2:94].values).to(torch.float) 

        print ('xshape', x.shape)


        # There exists 3 different classes in the dataset:
        # 0=licit,  1=illicit, 2=unknown
        # change to one-hot type encoding
        mapping =  {'unknown': -1, '1': 1, '2': 0}
        class_df['class'] = class_df['class'].map(mapping)
        y = torch.from_numpy(class_df['class'].values)
        
        # renaming columns, 
        col_names ={0: 'id',1:'ts','txId': 'id', 'txId1': 'id1', 'txId2' :'id2'}
        feat_df = feat_df.rename(columns=col_names)
        edge_df = edge_df.rename(columns=col_names)
        class_df=class_df.rename(columns=col_names)

        tx_mapping = {idx: i for i, idx in enumerate(feat_df['id'].values)}
        feat_df['id'] = feat_df['id'].map(tx_mapping)
        edge_df['id1'] = edge_df['id1'].map(tx_mapping)
        edge_df['id2'] = edge_df['id2'].map(tx_mapping)
        class_df['id'] = class_df['id'].map(tx_mapping)

        # create masks
        # Timestamp based split
        # train_mask: 1 - 34 time_step, test_mask: 35-49 time_step
        time_step = torch.from_numpy(feat_df['time_step'].values)
        edge_index = torch.from_numpy(edge_df.values).t().contiguous()
        edge_attr = torch.ones(edge_index.shape[1]) 
        # recomended 1/3 for testingg
        # add random sampling here
        ##  add function here for different splits
        train_ts=22
        val_ts=32
        train_mask=(time_step<train_ts) & (y!=-1)  
        val_mask=(time_step>=train_ts) & (time_step<val_ts) & (y!=-1)
        test_mask=(time_step>=val_ts) & (y!=-1)

        # update all labeles of -1 to 0, to ensure 2 classes only for models
        # in validation training 
        y[y==-1]=0
    
        ####################################################################
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, train_mask=train_mask, 
                    test_mask=test_mask, val_mask=val_mask)
        

        return graph

    def add_augmentation(self, view1data):
        """
        From SelfGNN 
        Augmented view data generation using the full-batch data.

        :param view1data:
        :return:
        """
        print("Adding augmentation")
        view2data = view1data if self.augumentation is None else self.augumentation(view1data)
 
        diff = abs(view2data.x.shape[1] - view1data.x.shape[1])

        if diff > 0:
            """
            Data augmentation on the features could lead to mismatch between the shape of the two views,
            hence the smaller view should be padded with zero. (smaller_data is a reference, changes will
            reflect on the original data)
            """
            smaller_data = view1data if view1data.x.shape[1] < view2data.x.shape[1] else view2data
            smaller_data.x = F.pad(smaller_data.x, pad=(0, diff))
            view1data.x = F.normalize(view1data.x)
            view2data.x = F.normalize(view2data.x)
        
        nodes = torch.tensor(np.arange(view1data.num_nodes), dtype=torch.long)

        data = Data(nodes=nodes, edge_index=view1data.edge_index, edge_index2=view2data.edge_index,
                    edge_attr=view1data.edge_attr,
                    edge_attr2=view2data.edge_attr, x=view1data.x, x2=view2data.x, y=view1data.y,
                    train_mask=view1data.train_mask,
                    val_mask=view1data.val_mask, test_mask=view1data.test_mask, num_nodes=view1data.num_nodes)
        return data  # was originally  [data]  -- list

