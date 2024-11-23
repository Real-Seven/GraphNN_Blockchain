import torch
import numpy as np 

class Augmentation:

    """
    A utility for graph data augmentation
    Adapted from SelfGNN; Kefato, Z.T. and Girdzijauskas, S., (2021)
    """

    def __init__(self, method="none"):
        methods = { "split", "splitpc", "splitpc2", "zscore","rotate", "permute" ,"logarithmic", "none"}
        assert method in methods
        self.method = method
  

    @staticmethod   
    def _drop_feat(data, droppc=0.8):
        """
        Drop fearures,, smaller feature space
        """
        x = data.x.clone()
        drop_mask = torch.empty((x.size(1),), dtype=torch.float32).uniform_(0, 1) < droppc
        newdata = data.clone()
        x[:, drop_mask] = 0
        newdata.x=x
        return newdata


    @staticmethod
    def _splitpc(data, ratio=0.3):
        """
        Split 30/70, will zero out the rest, smaller feature space
        """
        perm = torch.randperm( data.x.shape[1]) 
        x = data.x.clone()
        x = x[:, perm]
        size = x.shape[1] // 2
        print(type(size))
        size1=int(ratio*x.shape[1])
        print(type(size1))
        size2=int((1-ratio)*x.shape[1])
        x1 = x[:, :size1]
        x2 = x[:, size2:]
        new_data = data.clone()
        data.x = x1
        new_data.x = x2
        return new_data

    @staticmethod
    def _splitpc2(data, ratio=0.25):
        """
        Split 25/75, will zero out the rest, smaller feature space
        """
        perm = torch.randperm( data.x.shape[1]) 
        x = data.x.clone()
        x = x[:, perm]
        size = x.shape[1] // 2
        print(type(size))
        size1=int(ratio*x.shape[1])
        print(type(size1))
        size2=int((1-ratio)*x.shape[1])
        x1 = x[:, :size1]
        x2 = x[:, size2:]
        new_data = data.clone()
        data.x = x1
        new_data.x = x2
        return new_data


    @staticmethod
    def _logarithmic(data):
        """
        Applies a log transform on node features 
        """
        x = data.x

        min_x, _ = x.min(dim=0)
        max_x, _ = x.max(dim=0)

        x1= (( x - min_x )/ (max_x - min_x))
        x1 = x1 + 1e-9
        logn=torch.log(x1)
        new_data = data.clone()
        new_data.x =      logn
        return new_data
    
    @staticmethod
    def _rotate(data):
        """
        Applies a rotation of feature

        :param data: The data to be augmented
        :return: a new augmented instance of the input data
        """

        slice=20
        x = data.x
        print(x.shape)
        new_x = torch.cat((x[:, -slice:], x[:, :-slice]),axis=1)
        print(new_x.shape)
        new_data = data.clone()
        new_data.x =      new_x
        return new_data
    
    @staticmethod  
    def _drop_node(data, droppc=0.3):   
        node_num, _ = data.x.size()
        drop_num = int(node_num * droppc)

        idx_mask = np.random.choice(node_num, drop_num, replace = False).tolist()
        new_data=data.clone()
        new_data.x=data.x[idx_mask] = 0

        return new_data
    
    """
    Adapted from SelfGNN; Kefato, Z.T. and Girdzijauskas, S., (2021). 
    """
    
    @staticmethod
    def _standardize(data):
        """
        Applies a zscore node feature  augmentation.

        """
        
        x = data.x
        mean, std = x.mean(dim=0), x.std(dim=0)
        new_data = data.clone()
        new_data.x = (x - mean) / (std + 10e-7)
        return new_data

    @staticmethod
    def _split(data):
        """
        Shuffle and split into 2 - resulting in 2 smaller sets

        """
        perm = torch.randperm( data.x.shape[1]) 
        x = data.x.clone()
        x = x[:, perm]
        size = x.shape[1] // 2
        x1 = x[:, :size]
        x2 = x[:, size:]
        new_data = data.clone()
        data.x = x1
        new_data.x = x2
        return new_data
    
    @staticmethod
    def _permute(data):
        """
        Shuffle only

        """
        x= data.x.clone()
        x = data.x[torch.randperm(x.size(0))]
        newdata=data.clone()
        newdata.x=x
        return newdata

    def __call__(self, data):
        """
        Applies different data augmentation techniques
        """
        if self.method == 'split':
            return self._split(data)
        
        elif self.method == 'splitpc':
            return self._splitpc(data)
        
        elif self.method == 'splitpc2':
            return self._splitpc2(data)
        
        elif self.method == 'rotate':
            return self._rotate(data)
        
        elif self.method == 'permute':
            return self._permute(data)
             
        elif self.method == "zscore":
            return self._standardize(data)
        
        elif self.method == "drop_feat":
            return self._drop_feat(data)
        
        elif self.method == "logarithmic":
            return self._drop_feat(data)     

        elif self.method == "none":
            return data
