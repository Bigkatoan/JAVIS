# Sim-to-real checklist

Mục tiêu: một policy/controller tune trong mô phỏng (`javis/robot_constants.py`
+ mjlab) chạy được trên robot thật mà gần như không phải sửa gì. Muốn vậy,
mọi con số hiện đang là *placeholder* trong code phải được thay bằng số đo
thật, và khoảng cách còn lại (những thứ không đo được chính xác 100%) phải
được mô phỏng bằng domain randomization thay vì bỏ qua.

Điền số liệu trực tiếp vào các mục "Giá trị đo được" bên dưới rồi gửi lại —
tôi sẽ cập nhật vào `robot_constants.py`. Đánh dấu `[x]` khi đã đo xong.

---

## 1. Khối lượng & trọng tâm

Đây là nhóm quan trọng nhất — kết quả mô phỏng hiện tại (robot 2 bánh bị đổ
khi thả tự do, xem README mục "known gaps") là do khối lượng đang được ước
lượng theo *thể tích* mesh CAD (giả định mật độ đều), không phải khối lượng
thật, nên trọng tâm tính ra gần chắc chắn sai.

- [ ] **Tổng khối lượng robot đã lắp ráp hoàn chỉnh** (kg): __________
      → vẫn cần, để đối chiếu/kiểm tra lại `CHASSIS_MASS_KG` (
      `javis/robot_constants.py`) một khi biết đủ: tổng thật ≈
      `CHASSIS_MASS_KG` + 2 × 2.936 (khối lượng bánh, đã đo).
- [x] **Khối lượng từng bánh xe**: đo 2026-08-06, mỗi bánh **2936 g**
      (giả định 2 bánh bằng nhau — báo lại nếu khác). Đã cập nhật thành
      `WHEEL_MASS_KG` riêng trong `robot_constants.py`, tách hẳn khỏi khối
      lượng thân (không còn dùng `settotalmass` chia theo thể tích chung
      như trước).
- [ ] **Vị trí trọng tâm (CoM) của toàn robot**, đo bằng 1 trong 2 cách:
  - Cách đơn giản: đặt robot lên 2 điểm tựa cân riêng (VD 2 cân nhỏ dưới 2
    bánh, hoặc dưới đáy trước/sau nếu có chân chống) → tính CoM dọc trục X
    (trước-sau) từ tỉ lệ trọng lượng 2 đầu.
    Kết quả (offset theo trục X tính từ tâm trục bánh, mét): __________
  - Cách chính xác hơn: treo robot tự do bằng dây tại 2 điểm khác nhau, CoM
    nằm trên giao điểm 2 đường dây dọi kéo dài — làm 2 lần theo 2 hướng khác
    nhau để ra tọa độ 3D.
    Kết quả (x,y,z so với gốc `body` — xem `robot.urdf`, gốc `body` nằm ở
    khoảng tâm trục bánh): __________
- [ ] **Khối lượng các cụm linh kiện nặng** (giúp đối chiếu/tinh chỉnh
      `CHASSIS_MASS_KG` — hiện chỉ là 1 số tổng đoán 6.0kg cho toàn bộ 343
      part gộp trong link `body`, biết thêm phân bố sẽ giúp trọng tâm đúng
      hơn, quan trọng vì robot đang bị đổ trong mô phỏng):
  - [x] Pin: **3423 g**. Đo 2026-08-06. Vị trí lắp: __________
  - [ ] Cụm Jetson + giá đỡ: __________ kg
  - [ ] Camera D435: __________ kg
  - [ ] Board(s) MKS ODrive Mini — **cho biết thực tế có bao nhiêu board**
    (mesh CAD lặp tên 7 lần, nghi là 1-2 board thật được tham chiếu nhiều
    lần): số lượng thật: __________, khối lượng mỗi board: __________ kg
  - [x] Cụm hộp số + bánh răng: không phải bộ truyền lực — xác nhận
    2026-08-06 đây là cơ cấu dời nam châm encoder lên board MKS ODrive Mini
    (xem mục 3), khối lượng của nó vẫn tính trong link `body` như bình
    thường, không cần tách riêng.

## 2. Hình học / CAD

- [ ] **Gán vật liệu (material/density) cho từng part trong Onshape.** Đây
  là cách sửa tận gốc thay vì phải patch `TOTAL_MASS_KG` mãi mãi — một khi
  material được gán trong CAD, `onshape-to-robot` sẽ tự xuất khối lượng và
  quán tính đúng cho từng part khi chạy lại `venv/bin/onshape-to-robot javis`.
  Nếu làm được bước này thì mục 1 ở trên coi như xong tự động, không cần
  cân tay nữa (vẫn nên cân tổng robot thật 1 lần để đối chiếu/verify).
- [ ] **Đường kính + bề rộng bánh xe thật** (mét): đường kính __________,
      bề rộng __________
      → đối chiếu với `assets/wheel.stl` (hiện đo được bán kính ~0.098m,
      bề rộng ~0.067m từ mesh CAD) để xác nhận CAD khớp bánh thật đang dùng.
- [ ] **Robot thật có chân chống / đuôi / điểm tựa thứ 3 nào không?** Nếu
      có, cần thêm vào `robot.urdf` (CAD) — hiện tại mô phỏng chỉ có 2 bánh
      chạm đất, không có điểm tựa nào khác, nên nếu robot thật có thiết kế
      tự cân bằng thuần 2 bánh (không chân chống) thì đây **không phải lỗi**,
      mà là do robot cần bộ điều khiển cân bằng chủ động (xem README).
      Có chân chống/đuôi không: __________
- [x] **`gearbox.stl` + `spur_gear__20_teeth`/`__30_teeth` trong CAD dùng để
      làm gì?** Xác nhận 2026-08-06: **không phải truyền lực** — dùng để dời
      vị trí đặt nam châm encoder lên gần chip cảm biến trên board MKS
      ODrive Mini (chip encoder nằm trên board, không nằm ngay trục bánh,
      nên cần bộ bánh răng để "chuyển tiếp" chuyển động quay tới đúng chỗ
      đặt nam châm). Hệ quả quan trọng: xem mục 3 bên dưới — encoder vì vậy
      KHÔNG quay cùng tốc độ với bánh xe.

## 3. Truyền động (actuator / drivetrain)

> Xem **`MKS_XDRIVE_MINI.md`** trước — tài liệu riêng tổng hợp toàn bộ kiến
> thức về chính con board đang dùng (ghost axis1, lỗi CAN transceiver, cảnh
> báo brick firmware, vấn đề hiệu chuẩn encoder lúc boot...), nguồn từ tài
> liệu cộng đồng + kiểm chứng thực tế. Mục này chỉ còn checklist đo đạc.

- [x] **Board thật chỉ dùng 1 axis/board** (`axis0`, `axis1` là "ghost" —
      không có động cơ thật). Robot dùng 2 board riêng, mỗi board 1 bánh —
      không phải axis0/axis1 chung 1 board như giả định ban đầu. Xác nhận
      2026-08-06, xem `MKS_XDRIVE_MINI.md`.
- [x] **Chế độ dẫn động motor–bánh**: hub motor dẫn động **trực tiếp**
      (direct-drive), gear_ratio = 1:1. Xác nhận 2026-08-06.
- [x] **Loại encoder**: SPI tuyệt đối (absolute), đúng như
      `odriveconfig.txt` mẫu (`ENCODER_MODE_SPI_ABS_AMS`) — không phải Hall
      tích hợp trong hub motor. Xác nhận 2026-08-06.
- [x] **Tỉ số bánh răng bộ dời-nam-châm-encoder**: 2 tầng nối tiếp 30:20 rồi
      20:30 → triệt tiêu nhau (30/20 × 20/30 = 1). **Encoder đo đúng 1:1 với
      trục bánh xe** — không cần quy đổi gì khi đọc số ODrive (turns,
      vel_limit, input_pos... đã đúng là của bánh xe luôn). Xác nhận
      2026-08-06.
- [x] **Chế độ điều khiển ODrive**: chốt **VELOCITY_CONTROL** (2026-08-06),
      `input_mode = INPUT_MODE_PASSTHROUGH`. Đã cập nhật `odriveconfig.txt`
      (bỏ `pos_gain`/`trap_traj.config.*` của position mode cũ). Khớp sẵn
      với `WHEEL_ACTUATOR_CFG` và action space của `javis/velocity_task.py`
      — không cần đổi code sim.
- [x] **`motor.config.pole_pairs`**: đếm thật 2026-08-06 — 30 nam châm rotor
      / 2 = **15**. Stator 27 răng (cấu hình 30 cực/27 răng, phổ biến ở hub
      motor hoverboard). Đã điền vào `odriveconfig.txt`.
- [x] **Cấu hình ODrive thật, chạy trực tiếp trên board (axis0)**: đã áp
      config + hiệu chuẩn thành công 2026-08-06 qua
      `scripts/setup_odrive.py --axis 0` — motor calibration OK (phase
      resistance 0.240Ω, inductance 0.507mH), encoder offset calibration
      OK, closed-loop control OK, test quay 1 turn/s đo được 1.03 turn/s,
      không lỗi. Pin thật: 6S Li-ion Lishen SK21700 (18-25.2V) — đã đặt
      undervoltage_trip=19V, overvoltage_trip=28V (số mẫu cũ 8V/56V hoàn
      toàn sai cho hệ 24V này). `current_lim=15A` (≤16A liên tục theo
      datasheet). **axis1 (bánh còn lại) chưa chạy** — dùng lại
      `scripts/setup_odrive.py --axis 1` khi đấu dây xong bánh thứ 2.
- [x] **Torque constant (Kt) đo thật** 2026-08-06 trên axis0/bánh phải:
      **0.207 N·m/A** (ODrive firmware trước đó để mặc định 0.04 — chưa
      từng calibrate thật). Đo bằng cách quay bánh tự do (biết trước quán
      tính từ CAD+khối lượng thật), lệnh step vận tốc, log dòng đo được
      (Iq_measured) + vận tốc ở ~1.5kHz, tính Kt = I×Δω / ∫Iq dt qua nhiều
      cửa sổ thời gian. Đối chiếu chéo: KV ngầm suy ra ≈ 40 rpm/V → tốc độ
      không tải ở 23.5V đo được ≈ 939 RPM, hợp lý so với "800 RPM" trên
      datasheet (khác điện áp/tải nên không khớp tuyệt đối). Đã áp lên board
      thật (`motor.config.torque_constant`) và vào `scripts/setup_odrive.py`.
- [x] **`effort_limit`/`damping` trong `WHEEL_ACTUATOR_CFG`**: cập nhật theo
      số đo thật — `effort_limit = Kt × current_lim = 0.207×15 ≈ 3.1 N·m`;
      `damping = 0.028 N·m/(rad/s)` từ `scripts/calibrate_actuator.py fit`
      trên log thật vừa đo. Log chỉ dài 0.2s, khá nhiễu, **chưa đạt tới vận
      tốc ổn định** (RMS fit error 1.86 rad/s) — đủ tốt hơn hẳn số đoán từ
      datasheet mâu thuẫn trước đây, nhưng chưa phải số tune cuối cùng.
- [ ] **Log step-response dài hơn, sạch hơn** (ưu tiên cao — thay bản log
      0.2s vừa đo): lệnh vài mức vận tốc cố định, mỗi mức giữ đủ lâu để đạt
      trạng thái ổn định (không chỉ đoạn đầu tăng tốc), có ít nhất 1 mức đủ
      lớn để dòng bão hoà ở `current_lim` (giúp xác định `effort_limit`
      đáng tin hơn). Chạy lại `scripts/calibrate_actuator.py fit`.
- [ ] **Tune lại `vel_gain`/`vel_integrator_gain` trên ODrive thật** — đang
      để `vel_gain=0.02` (số mẫu cũ, chưa tune), theo `damping=0.028` vừa
      fit thì gain tương đương lý thuyết cho vòng vận tốc ODrive rơi vào
      khoảng `0.028 × 2π / 0.207 ≈ 0.85 A/(turn/s)` — cao hơn nhiều so với
      0.02 hiện tại, khớp với việc phản hồi thực đo được khá "mềm"/chậm
      (0.2s chỉ lên được ~10/31.4 rad/s mục tiêu). Đây là điểm khởi đầu để
      thử, không phải giá trị an toàn đảm bảo — tăng dần thực nghiệm trên
      robot thật (tăng tới khi hơi dao động rồi lùi lại ~30-50%), **không
      tự áp trực tiếp** vì có thể gây dao động nếu tăng đột ngột.
- [ ] **Độ trễ giao tiếp** (communication/bus latency) giữa lệnh gửi đi và
      lúc động cơ thực sự phản hồi (VD qua CAN bus — thấy
      `odrv0.can.set_baud_rate(500000)` trong config mẫu). Đo bằng cách so
      thời điểm gửi lệnh vs thời điểm bắt đầu thấy encoder thay đổi trong
      log ở trên.
      Độ trễ đo được (ms): __________
      → set qua `ActuatorCfg.delay_min_lag`/`delay_max_lag` (đơn vị: số
      bước physics timestep) khi định nghĩa actuator, để policy train trong
      sim không bị bất ngờ bởi độ trễ thật khi chạy trên robot.

## 4. Ma sát & tiếp xúc

- [ ] **Hệ số ma sát bánh xe với mặt sàn thực tế robot sẽ chạy** (sàn nhà,
      thảm, ngoài trời...). Cách đo đơn giản: đặt robot (tắt động cơ, bánh tự
      do) trên mặt phẳng nghiêng, tăng dần góc nghiêng tới khi bắt đầu trượt
      → hệ số ma sát tĩnh ≈ tan(góc đó).
      Góc bắt đầu trượt: __________ độ → μ ≈ __________
      → thay vào `ROBOT_COLLISION.friction` trong `robot_constants.py`
      (hiện đang để `(1.0, 0.005, 0.0001)`, placeholder).
- [ ] **Độ nảy/đàn hồi (restitution) của bánh và sàn**, nếu quan trọng cho
      use-case (thường bỏ qua được với robot di chuyển chậm trên mặt phẳng).

## 5. Cảm biến

### IMU (⚠️ đã sửa: không phải DFRobot — là BMX160 + BMP388, xác nhận qua Jetson thật 2026-08-06)

Kiểm tra trực tiếp trên Jetson Orin Nano (`orin@10.42.0.147`) cho thấy IMU
thật là **Bosch BMX160** (accel+gyro+mag 9 trục) + **BMP388** (áp suất/nhiệt
độ), có driver ROS2 riêng đã viết sẵn:
`~/ros2_ws/src/bmx160_bmp388_driver` (node C++ `bmx160_bmp388_node`, chạy
như service `bmx160-bmp388.service`, `WantedBy=multi-user.target`).

- I2C bus: `/dev/i2c-7`, BMX160 addr `0x68`, BMP388 addr `0x76`.
- `publish_rate_hz: 50.0`, `frame_id: "imu_link"`.
- Topic: `/bmx160_bmp388_node/imu/data_raw` (raw accel+gyro, chưa fusion).
  `imu_filter_madgwick` được wire sẵn (nếu cài) để ra `/imu/data` (quaternion
  đã fusion) — cần `sudo apt-get install ros-humble-imu-filter-madgwick`
  trên Jetson nếu chưa có, kiểm tra bằng `ros2 topic list | grep imu`.
- Không có magnetometer riêng — BMX160 tích hợp mag trong cùng chip, dùng
  cho `imu_filter_madgwick` (`use_mag: True`, `world_frame: 'enu'`).

- [ ] **Cập nhật tên cảm biến trong `robot_constants.py`/CAD nếu đang ghi
      "DFRobot"** — sai, cần đổi thành BMX160/BMP388 cho nhất quán tài liệu.
- [ ] **Xác nhận pose lắp đặt thật khớp CAD.** Code đang lấy thẳng từ
  `robot.urdf` (`IMU_POS`/`IMU_RPY` trong `robot_constants.py`) — đúng theo
  thiết kế, nhưng nếu lắp tay có sai lệch so với CAD (đặc biệt là hướng/trục)
  thì cần đo lại và cập nhật.
- [ ] **Noise/bias thật của cảm biến** (lấy từ datasheet BMX160, hoặc đo trực
      tiếp: để robot đứng yên, `ros2 topic echo /bmx160_bmp388_node/imu/data_raw`
      vài chục giây, tính độ lệch chuẩn và giá trị trung bình lệch khỏi
      0/9.81):
      Gyro noise density: __________, bias: __________
      Accelerometer noise density: __________, bias: __________
      → dùng để cấu hình noise model qua `mjlab.utils.noise` khi định nghĩa
      observation terms đọc `imu_ang_vel`/`imu_lin_acc`, để policy train với
      domain randomization không "quá tin" vào cảm biến sạch như trong sim.
- [ ] **Chưa hiệu chuẩn extrinsic IMU↔camera** (comment trong
      `orin_vslam_bringup` gọi đây là "Phase B", `enable_imu_fusion` đang cố
      định `False` trong cuVSLAM) — cần gắn cứng cơ khí + đo lại trước khi
      bật fusion IMU vào VSLAM.

### Camera Intel RealSense D435 (xác nhận qua Jetson thật 2026-08-06 — không phải D435i, camera KHÔNG có IMU riêng)

Có package ROS2 bringup riêng cho camera này:
`~/ros2_ws/src/realsense_splitter` + `~/workspaces/isaac_ros-dev/src/orin_vslam_bringup`
(dùng cho Isaac ROS **cuVSLAM** (visual odometry) + **nvblox** (3D mesh/ESDF),
không phải chỉ để lấy ảnh cho policy).

- **D435 (không phải D435i)** — xác nhận trong config
  (`d435_single_stream.yaml`): `enable_accel: false`, `enable_gyro: false`
  ("D435 has no gyro/accel hardware"). IMU của robot là BMX160 rời (ở trên),
  không phải từ camera.
- **Config vận hành thật hiện tại** (`config/sensors/d435_single_stream.yaml`):
  - Color: `640x480 @ 30fps`
  - Depth: `848x480 @ 30fps`, `clip_distance: 2.75m` (cắt ở driver do độ
    chính xác D435 giảm rõ sau ~3m)
  - Infra1/Infra2: bật, cùng profile với depth, cấp thẳng cho cuVSLAM
  - **Emitter dot pattern giữ CỐ ĐỊNH TẮT** — firmware của chính con D435 này
    (`5.17.0.10`) từ chối lệnh `depth_module.emitter_on_off` (xác nhận bằng
    test thật), nên **không dùng được kiểu nhấp nháy emitter +
    `realsense_splitter` demux** như ví dụ mặc định của nvblox_examples —
    `orin_vslam_bringup` phải tự viết remapping thẳng vào topic gốc của
    `realsense2_camera` để lách qua giới hạn này. **Ghi nhớ: nếu build sim
    camera stream có kèm mô phỏng structured-light dot pattern, robot thật
    sẽ KHÔNG có pattern đó** (ảnh IR sạch, passive stereo).
  - Node thật tên `camera` (không phải theo `camera_name` param) → topic ở
    `/camera0/camera/...`, không phải `/camera0/...` — cần khớp đúng nếu sau
    này subscribe trực tiếp từ code robot.
- [ ] **Intrinsics thật** (không dùng số FOV chung chung 42° đang để tạm) —
      lấy qua `ros2 topic echo /camera0/camera/infra1/camera_info --once` (đã
      có node chạy sẵn qua `d435_realsense.launch.py`), hoặc `pyrealsense2`:
      `profile.get_stream(...).as_video_stream_profile().get_intrinsics()`.
      fx, fy, cx, cy (RGB): __________
- [ ] **Độ trễ pipeline camera** (capture → có data cho policy), quan trọng
      hơn với các tác vụ phản ứng nhanh.

## 5b. ⚠️ Encoder "mất giao tiếp SPI" — chập chờn thật, có tài liệu xác nhận + đã giảm thiểu, CHƯA hết gốc (2026-08-06)

**Cập nhật quan trọng**: sau khi tra tài liệu cộng đồng về chính board MKS
xDrive Mini (xem **`MKS_XDRIVE_MINI.md`**), xác nhận đây **không phải hiện
tượng lạ/chỉ xảy ra với board của mình** — là vấn đề đã được cộng đồng ghi
nhận: board này có lỗi hiệu chuẩn encoder không ổn định lúc khởi động
("noise-related encoder errors during initialization"). Vẫn chưa có giải
thích kỹ thuật sâu (EMI hay gì chính xác) từ phía MKS, nhưng đã biết đây là
đặc điểm chung của board, không phải lỗi riêng của 2 board robot này —
giảm bớt lo ngại rằng phần cứng bị lỗi/hỏng.

**Diễn biến quan sát trên phần cứng thật**: lỗi
(`encoder.error=128` `ERROR_ABS_SPI_COM_FAIL`, hoặc hiệu chuẩn "thành công"
nhưng hội tụ sai góc commutation — dòng cao, gần như không quay, không báo
lỗi) xảy ra ở NHIỀU thời điểm khác nhau: lúc hiệu chuẩn ban đầu, lúc hiệu
chuẩn lại, lúc vào closed-loop control — trên **cả 2 board** (không phải
đặc thù 1 board lỗi). Đã loại trừ: kẹt cơ khí (bánh quay tay rất nhẹ), dây
pha lỏng (kiểm tra tay, giắc chặt), config không lưu sạch. Cách khắc phục
cộng đồng khuyến nghị (tắt `startup_encoder_offset_calibration`, tự hiệu
chuẩn bằng phần mềm mỗi lần boot) đã áp dụng — xem `MKS_XDRIVE_MINI.md`.

**Đã làm được** (giảm thiểu, không phải sửa gốc): `scripts/setup_odrive.py`
giờ tự động retry khi lỗi xảy ra ở hiệu chuẩn encoder VÀ lúc vào closed-loop
control (2 điểm đã quan sát thấy lỗi), cộng thêm bước tự kiểm tra sau hiệu
chuẩn (`verify_and_fix_calibration`: quay thử ngắn, nếu dòng cao mà vận tốc
thấp bất thường thì tự hiệu chuẩn lại) vì ODrive không tự báo lỗi cho kiểu
hội tụ sai này. Test cuối chạy sạch từ đầu tới cuối không lỗi, nhưng **không
đảm bảo sẽ luôn như vậy** — bản chất ngẫu nhiên của lỗi này nghĩa là có thể
tái diễn bất cứ lúc nào có dòng điện thật chạy qua động cơ, kể cả lúc vận
hành thực tế sau này, không chỉ lúc setup.

- [ ] **Việc cần làm khi viết driver ROS2/phần mềm điều khiển thật** (mục 6
      bên dưới — hiện chưa có code nào): PHẢI có cơ chế phát hiện +
      tự phục hồi tương tự (`verify_and_fix_calibration`, retry vào closed
      loop) chạy MỖI LẦN robot khởi động thật, không chỉ tin
      `startup_encoder_offset_calibration` của ODrive là đủ — và nên có
      cảnh báo/log lại mỗi lần lỗi xảy ra trong vận hành thực tế để theo dõi
      tần suất, giúp đánh giá xem có cần can thiệp vật lý hay không.
- [ ] **Kiểm tra vật lý sâu hơn khi có dịp tháo máy**: khe hở/độ đồng tâm
      nam châm encoder qua bộ bánh răng 2 tầng, và cân nhắc thử shielding/
      tụ lọc gần chip encoder nếu nghi ngờ EMI — chưa làm được vì cần thao
      tác vật lý trực tiếp trên board.

- **Hệ quả cần nhớ**: bất kỳ log vận tốc thật nào ghi được trong lúc có
  `encoder.error != 0` đều không dùng được để calibrate
  (`scripts/calibrate_actuator.py fit`) hoặc so sánh sim-thực — luôn kiểm
  tra `axis0.error`/`encoder.error` sau khi log xong, bỏ log nếu có lỗi.

## 6. Vòng điều khiển / phần mềm

**Compute thật đã xác nhận** (SSH vào `orin@10.42.0.147`, 2026-08-06):
- Jetson **Orin Nano Engineering Reference Developer Kit Super** (P3767-0005),
  JetPack/L4T R36.5.2, Ubuntu 22.04, 6 CPU core, RAM 7.4GiB (~6.3GiB khả
  dụng), NVMe 467GB (còn 371GB trống).
- Máy này đang gánh **cả stack trợ lý giọng nói** (wakeword-listener,
  whisper.cpp, piper TTS, ollama, conversation-coordinator — tên các service
  systemd đã enable) **lẫn perception** (Isaac ROS cuVSLAM/nvblox, xem mục
  5) — cần tính vào ngân sách compute khi sau này thêm policy RL chạy realtime
  trên cùng máy, không phải máy trống chỉ để lái robot.

**CAN bus: đã lên (interface), nhưng CHƯA có driver nói chuyện với ODrive:**
- `can0` **UP**, 500 kbit/s, qua CAN controller onboard của chính Jetson
  (Tegra **MTTCAN**, `c310000.mttcan` — không phải qua USB-CAN adapter rời).
  Được cấu hình tự động lúc boot bằng service riêng đã viết sẵn:
  `/etc/systemd/system/can0-up.service` (`ip link set can0 type can bitrate
  500000` + `up`).
- `candump can0` (3s) **không thấy traffic nào** lúc kiểm tra — hợp lý vì 2
  board ODrive hiện không cắm điện/nối vào Jetson này (đợt cấu hình trước đó
  làm qua USB nối trực tiếp máy khác, xem `MKS_XDRIVE_MINI.md` +
  `scripts/setup_odrive.py`).
- **Grep toàn bộ home dir Jetson: không có bất kỳ code nào (ROS2 package,
  script Python) nói chuyện CAN với ODrive** — xác nhận lại đúng như README
  đã ghi "chưa có driver". Việc còn lại: viết node/script mới gửi
  `Set_Axis_Requested_State`/`Set_Input_Vel`/đọc `Heartbeat`+`Get_Encoder_Estimates`
  qua `can0`, tái sử dụng logic calibrate+verify+retry đã có trong
  `scripts/setup_odrive.py` (xem `MKS_XDRIVE_MINI.md` mục hiệu chuẩn).
- ⚠️ Chưa kiểm tra lỗi phần cứng SN65HVD230 "Listen Only" (xem
  `MKS_XDRIVE_MINI.md`) trên đường CAN thật của Jetson này — làm việc này
  đầu tiên khi bắt đầu viết driver, trước khi nghi ngờ lỗi phần mềm nếu gửi
  lệnh xuống không có tác dụng.

- [ ] **Tần số vòng điều khiển thật trên robot** (Jetson gửi lệnh xuống
      ODrive bao nhiêu Hz?): __________ Hz
      → phải khớp với `decimation` × `SimulationCfg.mujoco.timestep` khi
      định nghĩa RL task, để 1 bước hành động của policy trong sim tương ứng
      đúng khoảng thời gian thật trên robot.
- [ ] **Viết driver CAN↔ODrive** (chưa có, xem phân tích ở trên) — interface
      (đơn vị, tên biến) nên khớp 1-1 với action/observation của policy, đây
      là phần thay thế `data.ctrl[...]`/`data.sensordata[...]` trong sim
      bằng I/O thật.

## 7. Giới hạn an toàn

- [ ] **Đối chiếu giới hạn dòng/mô-men/vận tốc đã set trên ODrive thật**
      (`motor.config.current_lim`, `controller.config.vel_limit`...) với
      `effort_limit` đã calibrate trong `WHEEL_ACTUATOR_CFG` — đảm bảo sim
      không "cho phép" policy ra lệnh vượt quá những gì phần cứng thật chịu
      được, nếu không policy học được hành vi sim cho phép nhưng robot thật
      sẽ bị driver tự cắt/bảo vệ giữa chừng → hành vi khác nhau giữa sim và
      thật.

## 8. Domain randomization (bù phần không đo được chính xác 100%)

Không có phép đo nào là hoàn hảo — mục tiêu không phải làm sim khớp thật
tuyệt đối, mà là: sau khi calibrate xong (mục 1-7), train policy với
domain randomization dao động quanh các giá trị đã đo (VD ±10-20% khối
lượng, ma sát, độ trễ actuator) để policy robust với phần sai số còn lại,
thay vì cần sim khớp thật 100%. mjlab có sẵn cơ chế này qua `EventManager`
(xem các ví dụ trong `mjlab/tasks/velocity`: `foot_friction`, `encoder_bias`
theo đúng pattern này, áp dụng tương tự cho ma sát bánh/độ lệch encoder của
robot này).

## 9. Vòng lặp xác thực (đo lại định kỳ)

- [ ] Ghi log trên robot thật định kỳ (lệnh vs phản hồi thực tế của bánh,
      IMU, điện áp pin...) và so sánh lại với sim bằng đúng cách đã làm ở
      mục 3 (`calibrate_actuator.py fit`) — pin xả yếu dần, bánh mòn, ma sát
      đổi theo mặt sàn... nên đây không phải việc làm 1 lần.
