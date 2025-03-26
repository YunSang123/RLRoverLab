import csv
import matplotlib.pyplot as plt

# CSV 불러오기
point_distribution = []
with open("point_distribution.csv", mode="r") as file:
    reader = csv.reader(file)
    next(reader)  # 헤더 건너뛰기
    for row in reader:
        point_distribution.append([float(x) for x in row])

# x, y 좌표만 추출
xs = [p[0] for p in point_distribution]
ys = [p[1] for p in point_distribution]

# 시각화
plt.figure(figsize=(6, 6))
plt.scatter(xs, ys, s=10)
plt.title("Point Distribution (x, y)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.axis("equal")

# show 대신 파일로 저장
plt.savefig("point_distribution2.png")
print("Saved plot to point_distribution.png")