import h5py

# h5 파일 열기
with h5py.File('best_agent_0.h5', 'r') as file:
    # 파일 내의 모든 그룹 및 데이터셋 확인
    for key in file.keys():
        print("Dataset:", key)
        data = file[key][:]
        print(data)
