from sklearn.model_selection import ParameterGrid
import re
import os

"""
Utility class 
reading and writing files
generating unique model id'd
generating hyperparameters/permutations

"""


def generate_args():
    model=["gcn","gat", "sage"]
    aug=[ "split","zscore","none", "logarithmic", "rotate", "permute", "splitpc2", "splitpc"]
    layers=[[ 32, 32],[64,64] ,[128,128]]
    lr=[0.0005, 0.001,0.0001]  
    dropout=[0.5] 
    epochs=[150] 

    all_params= ParameterGrid(dict( model=model,    aug=aug, layers=layers, lr=lr, dropout=dropout, epochs=epochs))
    return all_params

def get_global_model_id(params):
    model_id=''.join(key+"."+str(value)+"." for (key,value) in params.__dict__.items())
    model_id= re.sub(r"[^a-zA-Z0-9.]","",model_id)
    print(model_id)
    return model_id                    

def write_timings (args, duration):
    timings_loc=".\\data\\timings"
    filename="EllipticTimings005.txt"
    if not os.path.exists(timings_loc):
        os.makedirs(timings_loc)

    filename=os.path.join(  timings_loc, filename)
    with open(filename, 'a') as f:
        f.write(f"{args},{duration:.2f}"+ "\n")

def write_timings_eval (modelname, duration):
    timings_loc=".\\data\\timings"
    filename=f"{modelname}.txt"
    if not os.path.exists(timings_loc):
        os.makedirs(timings_loc)

    filename=os.path.join(  timings_loc, filename)
    with open(filename, 'a') as f:
        f.write(f"{modelname},{duration:.2f}"+ "\n")


