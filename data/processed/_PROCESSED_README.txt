The processed folder contained the dataset in torch graph format.

The raw data is loaded from the csv files, cleaned, selected and transformed into graph format. The augmentation is then applied to a copy of the node features and added to the graph.  It is then saved in this *.pt format.

The file is named according to the augmentation it contains.  

This means that when reading the data into memory, when training or evaluating, we first check to see if the data exists in torch format. If its here, it skips loading/augmenting. If its not here, it will load the csv, and process from scratch.


This is the list of graph data sets, one each for the augmentations.

EllipticData_logarithmic.pt
EllipticData_none.pt
EllipticData_permute.pt
EllipticData_rotate.pt
EllipticData_split.pt
EllipticData_splitpc.pt
EllipticData_splitpc2.pt
EllipticData_zscore.pt

These are torch files in case transforms and filters are required.
pre_filter.pt
pre_transform.pt
