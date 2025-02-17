import pickle

with open('./data/env.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data)