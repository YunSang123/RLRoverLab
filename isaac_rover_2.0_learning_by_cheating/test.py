import pickle
import omni.isaac.core.utils.prims as prim_utils

with open('./data/env.pkl', 'rb') as f:
    data = pickle.load(f)