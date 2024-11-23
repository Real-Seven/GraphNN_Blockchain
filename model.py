from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression as SciLogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import torch.nn.functional as F
import torch.nn as nn
import torch

from functools import wraps
import copy

"""
Utility class for building all models 
components, training and evaluating.
Parallel network from BYOL and SelfGNN

"""

class EarlyStopping:
    def __init__(self, mode='min', patience=5, min_delta=0):
        """
        Args:
            mode (string): Are we Min(imising) or Max(imising)the metric?
            patience (int): How many epoch to wait after loss 
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
        elif (self.mode == 'min' and metric > self.best_score + self.min_delta) or \
             (self.mode == 'max' and metric < self.best_score - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = metric
            self.counter = 0
     

def get_evaluation_model (modelname, **kwargs):
    """
        Return mostly defaulf parameters.
        Adjustment made to overcome training errors
        Or to refect Anomoly Detection aspect
        
    """
    if modelname == 'IsolationForest':
        return  IsolationForest(n_estimators=100, contamination=0.3)
    elif modelname=="RandomForest":
        #weight_d={0:9, 1:4}
        return RandomForestClassifier (n_estimators=100, class_weight='balanced_subsample')
    elif modelname=='XGBClassifier':
        return XGBClassifier(n_estimators=50, max_depth=20, learning_rate=0.1, objective='binary:logistic') 
    elif modelname=='Logistic':
        return SciLogisticRegression(solver='liblinear')
    elif modelname=='Saga':
        return SciLogisticRegression(max_iter=300,solver='saga')  
    elif modelname=='KNN':
        # Gridsearch too slow, settle on 2
        # parameters = {"n_neighbors": range(2, 4)}
        # gridsearch = GridSearchCV(KNeighborsClassifier(), parameters)
        return KNeighborsClassifier(n_neighbors=2) 
    elif modelname=='NaiveBayes':    
       return ComplementNB()
    else:
        return None


"""
Following taken from  BYOL;  Grill, J.B. et al., (2020)
"""


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def loss_fn(x, y):
    ## define this loss . SS
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn


def update_moving_average(ema_updater, ma_model, current_model):
    
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):

        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

"""
END of BYOL
"""


"""
Adapted from SelfGNN; Kefato, Z.T. and Girdzijauskas, S., (2021). 
"""

class Normalise(nn.Module):
    def __init__(self, dim=None, method="batch"):
        super().__init__()
        method = None if dim is None else method
        if method == "batch":
            self.norm = nn.BatchNorm1d(dim)
        else:  # No norm => identity
            self.norm = lambda x: x

    def forward(self, x):
        return self.norm(x)
    

class Encoder(nn.Module):

    def __init__(self, layer_config, gnn_type, dropout=None):
        super().__init__()
        rep_dim = layer_config[-1]
        self.gnn_type = gnn_type
        self.dropout = dropout
        #self.project = "batch" 
        self.stacked_gnn = get_encoder(layer_config=layer_config, gnn_type=gnn_type)
        self.encoder_norm = Normalise(dim=rep_dim, method="batch")

        self.projection_head = nn.Sequential(
            nn.Linear(rep_dim, rep_dim),
            Normalise(dim=rep_dim, method='batch'),
            nn.ReLU(inplace=True), nn.Dropout(dropout))

    def forward(self, x, edge_index, edge_weight=None):
        for i, gnn in enumerate(self.stacked_gnn):
            if self.gnn_type == "gat" or self.gnn_type == "sage":
                x = gnn(x, edge_index)
            else:
                x = gnn(x, edge_index, edge_weight=edge_weight)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.encoder_norm(x)
        return x, (self.projection_head(x) )


class GBYOL(nn.Module):

    def __init__(self, layer_config, dropout=0.5, tau=0.99, gnn_type='gcn'):
        super().__init__()

        self.online_encoder = Encoder(layer_config=layer_config, gnn_type=gnn_type, dropout=dropout)
        self.target_encoder = None
        
        self.target_ema_updater = EMA (tau)

        rep_dim = layer_config[-1]

        self.online_predictor = nn.Sequential(
            nn.Linear(rep_dim, rep_dim), 
            Normalise(dim=rep_dim, method="batch"),
            nn.ReLU(inplace=True), 
            nn.Dropout(dropout))

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def encode(self, x, edge_index, edge_weight=None, encoder=None):
        encoder = self.online_encoder if encoder is None else encoder
        encoder.train(self.training)
        return encoder(x, edge_index, edge_weight)

    def forward(self, x1, x2, edge_index_v1, edge_index_v2, edge_weight_v1=None, edge_weight_v2=None):
        """
        
        Apply online network on both views
        
        v<x>_rep is the output of the stacked GNN
        v<x>_online is the output of the online projection head, if used, otherwise is just a reference to v<x>_rep
        """
        v1_enc = self.encode(x=x1, edge_index=edge_index_v1, edge_weight=edge_weight_v1)
        v1_rep, v1_online = v1_enc if v1_enc[1] is not None else (v1_enc[0], v1_enc[0])
        
        v2_enc = self.encode(x=x2, edge_index=edge_index_v2, edge_weight=edge_weight_v2)
        v2_rep, v2_online = v2_enc if v2_enc[1] is not None else (v2_enc[0], v2_enc[0])
        
        """
        Apply the online predictor both views using the outputs from the previous phase 
        (after the stacked GNN or projection head - if there is one)
        """
        v1_pred = self.online_predictor(v1_online)

        v2_pred = self.online_predictor(v2_online)
        
        """
        Apply the same procedure on the target network as in the online network except the predictor.
        """
        with torch.no_grad():
            target_encoder = self._get_target_encoder()
            v1_enc = self.encode(x=x1, edge_index=edge_index_v1, edge_weight=edge_weight_v1, encoder=target_encoder)
            v1_target = v1_enc[1] if v1_enc[1] is not None else v1_enc[0]
            
            v2_enc = self.encode(x=x2, edge_index=edge_index_v2, edge_weight=edge_weight_v2, encoder=target_encoder)
            v2_target = v2_enc[1] if v2_enc[1] is not None else v2_enc[0]

        """
        Compute symmetric loss (once based on view1 (v1) as input to the online and then using view2 (v2))
        """
        loss1 = loss_fn(v1_pred, v2_target.detach())
        loss2 = loss_fn(v2_pred, v1_target.detach())

        loss = loss1 + loss2
        return v1_rep, v2_rep, loss.mean()


def get_encoder(layer_config, gnn_type, **kwargs):
    """
    Builds the GNN base as required, from SelfGNN
    """
    if gnn_type == "gcn":
        return nn.ModuleList([GCNConv(layer_config[i-1], layer_config[i]) for i in range(1, len(layer_config))])
    elif gnn_type == "sage":
        return nn.ModuleList([SAGEConv(layer_config[i-1], layer_config[i]) for i in range(1, len(layer_config))])
    elif gnn_type == "gat":
        heads = [8] * len(layer_config)
        return nn.ModuleList([GATConv(layer_config[i-1], layer_config[i] // heads[i-1], heads=heads[i-1], concat=True)
                              for i in range(1, len(layer_config))])

class LogisticRegression(nn.Module):
    """
    NN evaluator
    """
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

        torch.nn.init.xavier_uniform_(self.linear.weight.data)
        self.linear.bias.data.fill_(0.0)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, y):
        logits = self.linear(x)
        loss = self.cross_entropy(logits, y)
        return logits, loss
    
"""
End SelfGNN; Kefato, Z.T. and Girdzijauskas, S., (2021). 
"""
