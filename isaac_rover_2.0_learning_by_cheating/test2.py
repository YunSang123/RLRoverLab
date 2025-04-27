import torch
import matplotlib.pyplot as plt

# 1. 랜덤값 생성
rand = torch.empty(676).normal_(mean=0, std=0.1)

# 2. 히스토그램 그리기
plt.hist(rand.cpu().numpy(), bins=100, color='blue', edgecolor='black')  # bins 개수 조절 가능
plt.title('Histogram of rand values')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)

# 3. 저장
plt.savefig('rand_histogram.png')
print("Saved rand_histogram.png!")