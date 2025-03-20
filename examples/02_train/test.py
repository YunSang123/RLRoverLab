import torch

# 파일 로드
agent = torch.load("agent_300.pt", weights_only=True)
# ageent_226000 = torch.load("agent_226000.pt", weights_only=True)

# 상태 정보 확인
print(f"{agent}")
# print(f"agent_226000\n{ageent_226000}")