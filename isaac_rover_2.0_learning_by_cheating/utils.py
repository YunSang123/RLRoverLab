import os
import torch
from tqdm import tqdm


def sort_data(path):
    path = path
    files = os.listdir(path)
    print
    if not "data.pt" in files:
        print(f"files[0] : {files[0]}")
        data_path = os.path.join(path, files[0])
        print(f"data_path : {data_path}")
        data_full = torch.load(data_path)
        print(f"data_full : {data_full.keys()}")
        info = data_full["info"]
        data = data_full["data"]
        for i in (pbar := tqdm(range(1, len(files)))):
            pbar.set_description("Processing data")
            data_path = os.path.join(path, files[i])
            data_temp = torch.load(data_path)["data"]
            data = torch.cat((data, data_temp), dim=0)
        
        file = {"info": info,
                "data": data}
        torch.save(file, path + "data.pt")