import torch

# 파일 로드
agent = torch.load("best_agent.pt", weights_only=True)
# ageent_226000 = torch.load("agent_226000.pt", weights_only=True)

# 상태 정보 확인
# print(f"{agent}")

for k,v in agent.items():
    print(k)