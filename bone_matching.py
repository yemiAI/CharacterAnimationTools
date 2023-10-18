import numpy as np
import argparse

#indexing bones from this array but we rows with array 

parser= argparse.ArgumentParser()

parser.add_argument("index", type= int)
parser.add_argument("--frame", type= int , default=1)



args = parser.parse_args()

salsa = np.load("data/salsa_1_stageii.npz")
outarr = np.load("output.npz")

bones_from = outarr['poses'][args.frame].reshape([-1,3])

bones_to = salsa['poses'][args.frame].reshape([-1,3])

bone_search = bones_from[args.index]

for i, bone in enumerate(bones_to):
    
    if (np.linalg.norm(bone - bone_search) <= 1e-5) :
        
        print (i) 
    




