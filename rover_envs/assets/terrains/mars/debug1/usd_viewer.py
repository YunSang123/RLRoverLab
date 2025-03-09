from pxr import Usd, UsdGeom

def create_plane_usd(file_path="plane.usd"):
    # 새로운 USD stage 생성
    stage = Usd.Stage.CreateNew(file_path)
    
    # "/World/Plane" 경로에 평면 메쉬 생성
    plane = UsdGeom.Mesh.Define(stage, "/World/Plane")
    
    # 평면의 정점 좌표 (x, y, z) - 여기서는 y=0인 평면
    points = [
        (-1.0, 0.0, -1.0),  # 왼쪽 아래
        ( 1.0, 0.0, -1.0),  # 오른쪽 아래
        ( 1.0, 0.0,  1.0),  # 오른쪽 위
        (-1.0, 0.0,  1.0),  # 왼쪽 위
    ]
    
    # 면을 구성하는 인덱스와 각 면의 정점 개수 지정
    # 여기서는 4개의 정점을 가진 하나의 면(quad)을 생성합니다.
    plane.CreatePointsAttr(points)
    plane.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    plane.CreateFaceVertexCountsAttr([4])
    
    # USD 파일 저장
    stage.GetRootLayer().Save()
    print(f"USD 파일이 '{file_path}'에 생성되었습니다.")

if __name__ == '__main__':
    create_plane_usd()
