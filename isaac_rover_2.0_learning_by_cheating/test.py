import torch

t = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [1, 2, 3]   # 중복된 행
])

# NumPy로 변환 후 row 기준으로 중복 확인
import numpy as np
t_np = t.numpy()
unique_rows = np.unique(t_np, axis=0)

has_duplicates = len(unique_rows) != len(t)
print(has_duplicates)  # True