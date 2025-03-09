import torch

# 파일 로드
state_history = torch.load("state_history.pth")

torch.set_printoptions(threshold=torch.inf)

# 상태 정보 확인
print(f"Total states saved: {len(state_history)}")
print(f"Shape of first state: {state_history[0].shape}")
print("First state tensor:")
print(state_history[0])