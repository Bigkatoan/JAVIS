# PLAN.md — Kế hoạch tổng thể hệ thống JAVIS

Tài liệu quy hoạch kiến trúc + luồng vận hành cho toàn bộ robot JAVIS (2
bánh tự cân bằng, điều khiển bằng policy RL train trong mjlab, triển khai
thật qua ROS2 trên Jetson Orin Nano). Viết trước khi code driver ODrive/
policy inference, để các phần ăn khớp với nhau ngay từ đầu thay vì phải sửa
lại sau.

Trạng thái các quyết định/phát hiện tham chiếu trong tài liệu này đến thời
điểm 2026-08-07. Xem thêm: `SIM2REAL.md` (checklist đo đạc thật),
`MKS_XDRIVE_MINI.md` (kiến thức phần cứng board động cơ), `README.md`
(tổng quan repo + known gaps).

---

## 1. Bức tranh toàn cảnh

```
                    ┌─────────────────────────────────────────┐
                    │              HUẤN LUYỆN (offline)         │
                    │  Onshape CAD → URDF → mjlab (MuJoCo-Warp)  │
                    │  javis/velocity_task.py — train policy RL  │
                    │  Output: policy weights (.pt/.onnx)        │
                    └─────────────────┬───────────────────────┘
                                      │ deploy (copy weights lên Jetson)
                                      ▼
┌──────────────────────────── ROBOT THẬT (Jetson Orin Nano) ─────────────────────────────┐
│                                                                                           │
│   ┌────────────────┐   ┌──────────────────┐   ┌───────────────────────────────────┐   │
│   │  Cảm biến (đã   │   │  Nhận thức/định   │   │        Điều khiển chuyển động       │   │
│   │  có sẵn)        │   │  vị (đã có sẵn)   │   │             (CẦN XÂY MỚI)          │   │
│   │                 │   │                   │   │                                      │   │
│   │ bmx160_bmp388_  │   │ orin_vslam_       │   │  ┌─────────────────────────────┐   │   │
│   │ driver (IMU)    │   │ bringup (D435 →   │   │  │ policy_inference_node        │   │   │
│   │  → /bmx160_     │   │ cuVSLAM + nvblox) │   │  │ (chạy policy RL đã train)     │   │   │
│   │    bmp388_node/ │   │  → /odom, mesh/   │   │  │  sub: /cmd_vel (mong muốn),   │   │   │
│   │    imu/data_raw │   │    ESDF cho nav   │   │  │       /wheel_states,          │   │   │
│   │                 │   │                   │   │  │       imu/data               │   │   │
│   └────────────────┘   └──────────────────┘   │  │  pub: /wheel_left/cmd_vel,    │   │   │
│                                                  │  │       /wheel_right/cmd_vel    │   │   │
│                                                  │  └──────────────┬────────────────┘   │   │
│                                                  │                 │                     │   │
│                                                  │  ┌──────────────▼────────────────┐   │   │
│                                                  │  │ javis_odrive_driver_node        │   │   │
│                                                  │  │  sub: /wheel_{left,right}/cmd_vel│   │   │
│                                                  │  │  pub: /wheel_states              │   │   │
│                                                  │  │  giao tiếp USB (odrive python)   │   │   │
│                                                  │  └──────────────┬────────────────┘   │   │
│                                                  └─────────────────┼─────────────────────┘   │
└────────────────────────────────────────────────────────────────┼─────────────────────────┘
                                                                     │ USB (2 cổng)
                                              ┌──────────────────────┼──────────────────────┐
                                              ▼                                             ▼
                                   ┌─────────────────────┐                     ┌─────────────────────┐
                                   │  MKS xDrive Mini      │                     │  MKS xDrive Mini      │
                                   │  bánh TRÁI            │                     │  bánh PHẢI            │
                                   │  USB serial            │                     │  USB serial            │
                                   │  318236823335)         │                     │  3676365D3335)         │
                                   └─────────────────────┘                     └─────────────────────┘
```

**Nguyên tắc thiết kế chính**: tách rõ 2 lớp —
1. **Lớp thấp (low-level)**: `javis_odrive_driver_node` — chỉ biết nói
   chuyện với phần cứng ODrive qua USB, không biết gì về policy/RL. Nhận
   lệnh vận tốc từng bánh (rad/s), trả về trạng thái encoder.
2. **Lớp cao (policy)**: `policy_inference_node` — chạy model RL đã train,
   dịch "ý định di chuyển" (twist mong muốn) + cảm biến (IMU, encoder) thành
   lệnh vận tốc từng bánh. Đây là bộ não thay thế cho 1 bộ điều khiển động
   học vi sai (differential-drive kinematics) cổ điển — vì robot này là cấu
   hình tự cân bằng kiểu Segway (2 bánh đồng trục), không đứng vững thụ động
   được, nên cần policy học được cả việc giữ thăng bằng lẫn bám vận tốc.

Tách 2 lớp này để: (a) test được `javis_odrive_driver_node` độc lập bằng
lệnh tay (không cần policy), (b) đổi/tune policy sau này không đụng vào code
giao tiếp phần cứng.

---

## 2. Quyết định kiến trúc đã chốt

| Quyết định | Lý do | Ngày |
|---|---|---|
| Giao tiếp Jetson↔ODrive: **USB**, không CAN | Chỉ 2 động cơ, lợi thế bus-dùng-chung của CAN không đáng kể; 1 board bị hỏng chip CAN transceiver phần cứng, tốn nhiều giờ debug không ra | 2026-08-07 |
| Driver ODrive viết bằng **ROS2 (rclpy)**, không phải script độc lập | Jetson đã có sẵn ROS2 Humble + toàn bộ cảm biến publish qua topic — viết driver dạng node để tích hợp tự nhiên, dùng `Timer` của ROS2 để kiểm soát tần số vòng điều khiển chính xác | 2026-08-07 |
| Điều khiển vận tốc (`VELOCITY_CONTROL`), không phải vị trí | Khớp sẵn với `WHEEL_ACTUATOR_CFG`/`javis/velocity_task.py` trong sim | 2026-08-06 |
| Action space policy: **vận tốc từng bánh riêng** (không phải Twist chung) | Khớp `javis/velocity_task.py` hiện tại; đơn giản hơn khi driver không cần biết động học robot | (kế thừa từ sim) |

---

## 3. Các ROS2 node cần xây (chưa có node nào tồn tại)

### 3.1. `javis_odrive_driver` (ưu tiên 1 — làm trước)

**Vai trò**: cầu nối duy nhất giữa ROS2 và phần cứng ODrive qua USB.

- **Ngôn ngữ**: Python (`rclpy`) — tái sử dụng trực tiếp logic đã kiểm chứng
  trong `scripts/setup_odrive.py` (kết nối theo serial number, calibrate+
  verify+retry).
- **Lúc khởi động node**:
  1. Kết nối 2 board qua USB bằng serial number cố định (không dùng
     `find_any()` không tham số — dễ nhầm board nếu USB enumerate khác thứ
     tự mỗi lần cắm lại, đã từng gặp trong lúc debug).
  2. Chạy `calibrate_encoder_with_retry` + `verify_and_fix_calibration` cho
     cả 2 board (tái dùng nguyên hàm từ `scripts/setup_odrive.py`, không viết
     lại).
  3. Vào `AXIS_STATE_CLOSED_LOOP_CONTROL` cho cả 2 axis0.
  4. Chỉ sau khi cả 2 board sẵn sàng mới bắt đầu nhận lệnh — publish
     `/odrive_driver/ready` (`std_msgs/Bool`) để node khác biết khi nào an
     toàn để gửi lệnh.
- **Topic**:
  - Subscribe `/wheel_left/cmd_vel`, `/wheel_right/cmd_vel` (`std_msgs/
    Float32`, đơn vị rad/s).
  - Publish `/wheel_states` (`sensor_msgs/JointState`, `name=["left_wheel",
    "right_wheel"]`, `position`=turns×2π (rad), `velocity`=turns/s×2π
    (rad/s)).
  - Publish `/odrive_driver/right_wheel/error`,
    `/odrive_driver/left_wheel/error` (`std_msgs/UInt32`) — mirror
    `axis0.error` mỗi tick, để giám sát/log lỗi trong lúc vận hành thật
    (không chỉ lúc setup).
- **Timer**: 1 timer ROS2 chạy ở tần số cố định (mặc định đề xuất **50Hz**,
  cần đo/tune thật — xem mục 7 "Việc cần đo") — mỗi tick: đọc
  `axis0.controller.input_vel` mới nhất từ buffer nội bộ (ghi bởi
  subscriber callback), set xuống ODrive, đọc encoder, publish state.
- **Watchdog an toàn**: nếu không nhận được lệnh mới trên CẢ 2 topic
  `cmd_vel` trong quá X ms (đề xuất 200ms = gấp 10 lần chu kỳ điều khiển
  50Hz) → tự set `input_vel = 0` cho cả 2 bánh. Không tin tưởng
  `watchdog_timeout`/`enable_watchdog` nội bộ của ODrive (hiện đang tắt,
  `enable_watchdog=False` — xem `odrive_config_dump.txt`) làm lớp an toàn
  duy nhất.
- **Xử lý mất kết nối USB khi đang chạy**: bắt exception từ mọi lệnh
  Fibre/USB, nếu mất kết nối → publish `/odrive_driver/ready=false`, thử
  kết nối lại theo chu kỳ (backoff), KHÔNG crash cả node (1 board rớt không
  nên làm chết luôn board còn lại).

### 3.2. `policy_inference_node` (ưu tiên 2 — làm sau khi driver ODrive chạy ổn)

**Vai trò**: chạy policy RL đã train, sinh lệnh vận tốc từng bánh từ quan
sát thật.

- **Chưa quyết định**: format export policy (TorchScript `.pt` hay ONNX?),
  runtime trên Jetson (`torch` trực tiếp hay `onnxruntime`/TensorRT để tận
  dụng GPU?). Cần quyết định khi có policy đầu tiên train xong đủ tốt để
  deploy thử — chưa cần chốt ngay bây giờ.
- **Observation cần khớp 1-1 với `javis/velocity_task.py`** (thứ tự, đơn
  vị, scale phải giống hệt lúc train — sai 1 chỗ là policy chạy sai hoàn
  toàn mà không báo lỗi rõ ràng):
  - `base_lin_vel`, `base_ang_vel` — từ IMU (`/bmx160_bmp388_node/imu/
    data_raw`, hoặc `/imu/data` nếu đã bật `imu_filter_madgwick`).
  - `projected_gravity` — tính từ orientation IMU (cần orientation đã fusion,
    tức PHẢI bật `imu_filter_madgwick`, xem README/SIM2REAL.md — hiện tại
    launch file có điều kiện tuỳ package cài chưa).
  - `wheel_vel` — từ `/wheel_states` (`javis_odrive_driver` publish).
  - `actions` (action bước trước) — policy tự lưu trong node, không phải từ
    ROS2 topic.
  - `twist command` — từ đâu? Xem mục 3.3 bên dưới, đây là input cấp cao
    (người lái/autonomy) chưa có nguồn thật.
- **Output**: `/wheel_left/cmd_vel`, `/wheel_right/cmd_vel` — đúng những gì
  `javis_odrive_driver` đang subscribe.
- **Tần số chạy**: PHẢI khớp đúng tần số control loop lúc train
  (`decimation × SimulationCfg.mujoco.timestep` trong
  `javis/velocity_task.py`) — hiện đang là số giả định (50Hz), cần đối
  chiếu lại khi chốt số đo thật ở mục 7.

### 3.3. Nguồn lệnh "ý định di chuyển" (chưa có, cần quyết định sau)

`policy_inference_node` cần 1 twist command đầu vào (robot muốn đi hướng
nào/nhanh cỡ nào) — hiện chưa có nguồn nào tạo ra tín hiệu này trên robot
thật. Các lựa chọn (chưa quyết định, ghi lại để không quên):
- Teleop tay (joystick/bàn phím qua `teleop_twist_keyboard` — có sẵn trong
  hệ sinh thái ROS2, dễ tích hợp nhất để test ban đầu).
- Autonomy cấp cao dùng dữ liệu từ `orin_vslam_bringup` (nav2 stack, dùng
  mesh/ESDF từ nvblox để tránh vật cản) — việc làm sau, phức tạp hơn nhiều.

**Đề xuất cho giai đoạn đầu**: dùng teleop tay trước để test toàn bộ chuỗi
driver→policy→motor chạy đúng, autonomy tính sau.

---

## 4. Luồng vận hành (từ lúc bật nguồn tới lúc robot di chuyển)

```
1. Bật nguồn Jetson + pin động lực 2 board ODrive
2. systemd tự khởi động: bmx160_bmp388_driver, (orin_vslam_bringup nếu cần)
3. Khởi động javis_odrive_driver:
   a. Kết nối USB 2 board theo serial number
   b. Hiệu chuẩn encoder (calibrate_encoder_with_retry) — có thể quay nhẹ
      từng bánh trong bước này, ⚠️ ĐÃ CÓ TIỀN LỆ động cơ quay bất ngờ lúc
      hiệu chuẩn (xem SIM2REAL.md mục 5b) — PHẢI đảm bảo bánh xe không chạm
      đất/an toàn trước khi service này chạy lúc khởi động thật
   c. Tự kiểm tra (verify_and_fix_calibration), retry nếu phát hiện lỗi
   d. Vào closed-loop control, publish /odrive_driver/ready=true
4. Khởi động policy_inference_node — chờ /odrive_driver/ready=true mới bắt
   đầu publish lệnh (tránh gửi lệnh khi ODrive chưa sẵn sàng)
5. Khởi động nguồn lệnh (teleop hoặc autonomy) — người dùng/hệ thống bắt
   đầu ra lệnh di chuyển
6. Vòng lặp vận hành (mỗi tick timer, tần số cố định):
   twist command + IMU + wheel_states → policy → wheel cmd_vel
   → javis_odrive_driver → ODrive USB → động cơ quay
   → encoder mới → wheel_states → (vòng lặp tiếp)
```

**Dừng khẩn cấp (chưa thiết kế chi tiết, cần làm trước khi chạy thật)**:
- [ ] Cơ chế e-stop phần cứng (nút bấm ngắt nguồn động lực) độc lập với
      phần mềm — không phụ thuộc Jetson/ROS2 còn sống hay không.
- [ ] Cơ chế e-stop phần mềm (topic `/estop` hoặc tương tự) — khi kích hoạt,
      `javis_odrive_driver` chuyển ngay `AXIS_STATE_IDLE` cho cả 2 board,
      không chỉ set vận tốc 0 (set vận tốc 0 vẫn ở closed-loop, có thể vẫn
      sinh mô-men giữ vị trí).

---

## 5. Message/topic contract (bảng tổng hợp — nguồn chân lý duy nhất)

| Topic | Type | Publisher | Subscriber | Tần số |
|---|---|---|---|---|
| `/wheel_left/cmd_vel` | `std_msgs/Float32` (rad/s) | `policy_inference_node` (hoặc lệnh tay lúc test) | `javis_odrive_driver` | = tần số control loop |
| `/wheel_right/cmd_vel` | `std_msgs/Float32` (rad/s) | nt | `javis_odrive_driver` | nt |
| `/wheel_states` | `sensor_msgs/JointState` | `javis_odrive_driver` | `policy_inference_node` | = tần số control loop |
| `/odrive_driver/ready` | `std_msgs/Bool` | `javis_odrive_driver` | `policy_inference_node` | khi đổi trạng thái |
| `/odrive_driver/{left,right}_wheel/error` | `std_msgs/UInt32` | `javis_odrive_driver` | (giám sát/log) | = tần số control loop |
| `/bmx160_bmp388_node/imu/data_raw` | `sensor_msgs/Imu` | `bmx160_bmp388_driver` (đã có) | `policy_inference_node` | 50Hz (đã cấu hình) |
| `/cmd_vel` (tên tạm, chưa chốt) | `geometry_msgs/Twist` | teleop/autonomy (chưa có) | `policy_inference_node` | tuỳ nguồn |
| `/estop` (chưa làm) | `std_msgs/Bool` | (chưa có nguồn) | `javis_odrive_driver` | khi kích hoạt |

⚠️ Bảng này là **kế hoạch**, không phải đã cài đặt — cần khớp đúng khi code
thật, và cập nhật lại bảng nếu đổi tên/type lúc code (tránh tài liệu lệch
code).

---

## 6. Ánh xạ sim ↔ thật (điểm dễ sai nhất, kiểm tra kỹ khi deploy)

| Trong sim (`javis/velocity_task.py`) | Trên robot thật | Rủi ro nếu lệch |
|---|---|---|
| `imu_ang_vel`, `imu_lin_vel` (sensor site, sạch/không nhiễu) | IMU thật (có nhiễu/bias, xem SIM2REAL.md mục 5) | Policy "quá tin" cảm biến sạch → phản ứng sai với nhiễu thật |
| `wheel_vel` đọc thẳng từ `data.qvel` (chính xác tuyệt đối) | Encoder thật qua ODrive (có độ trễ, có thể có lỗi SPI chập chờn — SIM2REAL.md 5b) | Nếu log lúc có `encoder.error` lẫn vào observation → policy nhận input rác |
| Action → `data.ctrl[...]` áp ngay lập tức (không độ trễ) | Lệnh qua USB → ODrive → PWM động cơ (có độ trễ thật, chưa đo — mục 7) | Policy train với 0 độ trễ có thể mất ổn định khi có độ trễ thật đáng kể |
| `decimation × timestep` cố định, chính xác tuyệt đối | Timer ROS2 (có jitter, nhất là nếu Python/rclpy dưới tải) | Tần số control loop thật dao động quanh giá trị train, chưa biết ảnh hưởng bao nhiêu |
| Reward/domain randomization (mass, friction...) chỉ có trong sim | Không có gì cả — robot thật chỉ chạy inference, không train tiếp | Đây là lý do domain randomization lúc train quan trọng — bù cho sai lệch này |

---

## 7. Việc cần đo/quyết định trước khi deploy thật (tổng hợp từ SIM2REAL.md)

- [ ] **Tần số control loop thật** — hiện đề xuất 50Hz nhưng chưa đo/test gì
      trên phần cứng thật với ROS2 timer. Cần benchmark: chạy
      `javis_odrive_driver` với timer 50Hz, đo jitter thật (dùng
      `rclpy.time`/log timestamp mỗi tick), xem có ổn định không trước khi
      chốt số này ngược lại vào `decimation` trong sim.
- [ ] **Độ trễ USB thật** (lệnh gửi → động cơ phản hồi) — chưa đo, xem
      SIM2REAL.md mục 3.
- [ ] **Format export policy + runtime inference trên Jetson** — chưa quyết
      định (TorchScript/ONNX, CPU/GPU).
- [ ] **Nguồn twist command** (mục 3.3) — quyết định teleop trước, autonomy
      sau.
- [ ] **Thiết kế e-stop phần cứng + phần mềm** (mục 4) — chưa làm, cần
      trước khi chạy thật có tải trọng/di chuyển thật.
- [ ] **Watchdog timeout cụ thể** (mục 3.1) — đề xuất 200ms, chưa kiểm chứng
      thực nghiệm có đủ nhạy/không quá nhạy.

---

## 8. Thứ tự triển khai đề xuất

1. **`javis_odrive_driver`** — làm trước, test độc lập bằng lệnh tay qua
   `ros2 topic pub /wheel_left/cmd_vel ...` (không cần policy) để xác nhận
   driver + watchdog + calibrate-on-boot hoạt động đúng trên robot thật.
2. **Benchmark tần số/độ trễ thật** (mục 7) — cần số thật trước khi tinh
   chỉnh lại `decimation` trong sim cho khớp, tránh phải sửa qua lại nhiều
   lần.
3. **Export policy đầu tiên từ mjlab, viết `policy_inference_node`** — dùng
   teleop làm nguồn twist command để test end-to-end.
4. **Thiết kế + cài e-stop** — trước khi cho robot chạy có người/vật xung
   quanh.
5. **Autonomy/nav2** (dùng `orin_vslam_bringup`) — làm sau cùng, không chặn
   các bước trên.
