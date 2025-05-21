import airsim
import numpy as np
import cv2
import threading
import time
import os
from scipy.spatial.transform import Rotation as R

# ---------- 配置 ----------
DATA_ROOT = "./nuscene_sim/scene_000"
FPS = 10                    # 采集频率 Hz
vehicles = ["UAV1", "UAV2", "UAV3"]
hover_height = -10          # UAV1 悬停高度
offset_z    = -1            # UAV2/UAV3 比 UAV1 高 1m

# 创建保存目录
for v in vehicles:
    os.makedirs(f"{DATA_ROOT}/samples/CAM_{v}", exist_ok=True)
os.makedirs(f"{DATA_ROOT}/ego_pose", exist_ok=True)

# ---------- 主客户端（采集用） ----------
main_client = airsim.MultirotorClient()
main_client.confirmConnection()

# 起飞并移动到指定高度
def takeoff_and_move(client, name, z):
    client.takeoffAsync(vehicle_name=name).join()
    client.moveToZAsync(z, 2, vehicle_name=name).join()

for v in vehicles:
    main_client.enableApiControl(True, vehicle_name=v)
    main_client.armDisarm(True, vehicle_name=v)

takeoff_and_move(main_client, "UAV1", hover_height)
takeoff_and_move(main_client, "UAV2", hover_height + offset_z)
takeoff_and_move(main_client, "UAV3", hover_height + offset_z)

# ---------- 巡航线程（各用独立 client） ----------
def follow_loop(name, center, size, height):
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True, vehicle_name=name)
    client.armDisarm(True, vehicle_name=name)
    # 方形轨迹
    square = [
        (center[0]-size, center[1]-size),
        (center[0]-size, center[1]+size),
        (center[0]+size, center[1]+size),
        (center[0]+size, center[1]-size),
    ]
    idx = 0
    while True:
        wp = square[idx % 4]
        client.moveToPositionAsync(wp[0], wp[1], height, 3, vehicle_name=name)
        time.sleep(5)
        idx += 1

threading.Thread(target=follow_loop, args=("UAV2", (10,0), 5, hover_height+offset_z), daemon=True).start()
threading.Thread(target=follow_loop, args=("UAV3", (-10,0), 5, hover_height+offset_z), daemon=True).start()

# ---------- 工具函数 ----------
def get_image_and_pose(client, name):
    # 压缩返回 PNG，方便解码
    resp = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, True)],
                               vehicle_name=name)
    if not resp or not resp[0].image_data_uint8:
        return None, None
    buf = np.frombuffer(resp[0].image_data_uint8, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return None, None

    pose = client.simGetVehiclePose(vehicle_name=name)
    pos, ori = pose.position, pose.orientation
    quat = np.array([ori.w_val, ori.x_val, ori.y_val, ori.z_val])
    rot = R.from_quat(quat[[1,2,3,0]]).as_matrix()
    T = np.eye(4); T[:3,:3] = rot; T[:3,3] = [pos.x_val, pos.y_val, pos.z_val]
    return img, T

def project_img(img, T_rel, size):
    # 简化仿射投影，仅用于可视化对齐
    try:
        R_ = T_rel[:3,:3]; t_ = T_rel[:3,3]
        A = R_[:2,:2]; b = t_[:2]
        M = np.hstack([A, b.reshape(2,1)])
        return cv2.warpAffine(img, M, (size[1], size[0]))
    except:
        return np.zeros(size, dtype=np.uint8)

# ---------- 主循环：采集 + 保存 + 显示 ----------
cv2.namedWindow("UAV Projection", cv2.WINDOW_NORMAL)
print("开始数据采集，按 'q' 键退出并保存数据…")

frame = 0
interval = 1.0 / FPS
start = time.time()

while True:
    imgs = {}
    poses = {}
    # 1. 采集三机图像与姿态
    for v in vehicles:
        img, T = get_image_and_pose(main_client, v)
        imgs[v], poses[v] = img, T

    # 如果任意一图或位姿为空，跳过这一帧
    if any(imgs[v] is None or poses[v] is None for v in vehicles):
        continue

    # 2. 保存原始图像 & 位姿
    for v in vehicles:
        # 图像
        img_path = f"{DATA_ROOT}/samples/CAM_{v}/{frame:06d}.png"
        cv2.imwrite(img_path, imgs[v])
        # 位姿
        pose_path = f"{DATA_ROOT}/ego_pose/{v}_{frame:06d}.txt"
        np.savetxt(pose_path, poses[v], fmt="%.6f")

    # 3. 投影并显示
    T1 = poses["UAV1"]
    p_imgs = []
    for v in ["UAV2","UAV3"]:
        T_rel = np.linalg.inv(T1) @ poses[v]
        p_imgs.append(project_img(imgs[v], T_rel, imgs["UAV1"].shape))

    top = np.hstack((imgs["UAV1"], p_imgs[0]))
    bottom = np.hstack((p_imgs[1], np.zeros_like(imgs["UAV1"])))
    disp = np.vstack((top, bottom))

    cv2.imshow("UAV Projection", disp)
    frame += 1

    # 4. 退出条件
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("检测到 'q'，退出并保存完成。 共采集帧数:", frame)
        break

    # 保持固定采集频率
    elapsed = time.time() - start
    target = frame * interval
    if target > elapsed:
        time.sleep(target - elapsed)

cv2.destroyAllWindows()
