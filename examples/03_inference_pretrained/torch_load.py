import torch

# 체크포인트 파일 로드 (CPU로 매핑)
checkpoint = torch.load("./student_load/osr_t278800.pt", map_location=torch.device('cpu'))

# checkpoint가 딕셔너리 형태인지 확인 후 key 출력 및 optimizer 값 출력
if isinstance(checkpoint, dict):
    print("파일의 key 값:")
    for key in checkpoint.keys():
        print(key)
    
    # optimizer 키의 값을 출력
    if "optimizer" in checkpoint:
        print("\noptimizer 키의 값:")
        print(checkpoint["optimizer"])
    else:
        print("\noptimizer 키가 존재하지 않습니다.")
else:
    print("파일이 딕셔너리 형태가 아닙니다. 출력:", checkpoint)
