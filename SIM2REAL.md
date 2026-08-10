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

**✅ Cập nhật 2026-08-10: đã thay mô hình khối lượng.** Không còn dùng một mật
độ đồng đều cho cả 343 part nữa (cách cũ làm trọng tâm sai, vì pin chiếm 39%
thể tích thân nhưng nặng gấp ~5 lần nhựa in quanh nó). `javis/mass_model.py`
giờ chia thân xe thành các **nhóm linh kiện** và gán khối lượng riêng cho từng
nhóm; thể tích lấy trực tiếp từ mesh STL.

| nhóm | thể tích | khối lượng | nguồn |
|---|---|---|---|
| `battery` | 1683 cm³ | 3.423 kg | cân thật 2026-08-06 |
| `printed` | 2439 cm³ | 1.000 kg | 1 cuộn PLA, infill ~15% → 0.41 g/cm³ |
| `jetson` | 70 cm³ | 0.176 kg | catalog Orin Nano devkit |
| `camera` | 36 cm³ | 0.072 kg | datasheet D435 |
| `odrive` | 22 cm³ | 0.070 kg | 2 board MKS, **ước lượng — chưa cân** |
| `imu` | 1 cm³ | 0.005 kg | breakout BMX160 + BMP388 |
| `hardware` | 19 cm³ | 0.148 kg | thép 7.85 g/cm³ |
| `electronics_misc` | 24 cm³ | 0.073 kg | giả định 3.0 g/cm³ |
| `wiring_misc` | — | 0.300 kg | dây + lặt vặt, không có trong CAD, DR 0–1 kg |
| mỗi bánh | 1416 cm³ | 2.936 kg | cân thật 2026-08-06 |

→ thân **5.27 kg**, cả xe **11.14 kg**, CoM cao **0.2875 m** so với mặt đất,
lệch trước-sau chỉ **+1.4 mm**. Kiểm chứng chéo: 1 kg PLA / 2439 cm³ =
0.41 g/cm³, đúng tầm 15% infill + vỏ; và mô hình tái tạo lại đúng quán tính
bánh 0.01205 kg·m² đã đo độc lập khi hiệu chuẩn actuator.

Xem `scripts/inspect_mass.py` và `scripts/verify_mass_model.py` (32 phép kiểm
tra độc lập, gồm cả đối chiếu brute-force từng mesh và đối chiếu với MuJoCo).

- [ ] **Tổng khối lượng robot đã lắp ráp hoàn chỉnh** (kg): __________
      → để đối chiếu con số tính ra ở trên (11.14 kg). Đây là phép kiểm tra
      giá trị nhất còn lại: lệch nhiều nghĩa là một nhóm nào đó sai hẳn.
- [x] **Khối lượng từng bánh xe**: đo 2026-08-06, mỗi bánh **2936 g**
      (giả định 2 bánh bằng nhau — báo lại nếu khác). Trong sim mỗi bánh được
      random ±10% **độc lập nhau**, vì 2 hub motor không bao giờ giống hệt và
      chênh lệch đó gây nhiễu yaw thật.
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
- [ ] **Cân lại các nhóm đang là ước lượng.** Mỗi nhóm trong bảng trên có một
      dải domain randomization riêng trong `javis/sim_config.py`, đặt theo mức
      độ *thật sự biết* con số đó. Cân được cái nào thì thu hẹp dải cái đó —
      không phải sửa một sai số hệ thống.
  - [x] Pin: **3423 g**. Đo 2026-08-06. Vị trí lắp: __________
  - [ ] Cụm Jetson + giá đỡ: __________ kg (đang dùng catalog 176 g, DR ±20%)
  - [ ] Camera D435: __________ kg (đang dùng datasheet 72 g, DR ±20%)
  - [ ] Board(s) MKS ODrive Mini — **ưu tiên cao nhất trong nhóm này**: hoàn
    toàn chưa cân, các listing chỉ ghi khối lượng đóng gói. Đang giả định
    2 board × 35 g, DR ±40%. Số lượng thật: __________, mỗi board: ______ kg
  - [ ] Xác nhận bằng mắt mesh nào thuộc nhóm nào:
    `.venv/bin/python scripts/view_robot.py --color-by-group`. Riêng
    `main_board_simplified.stl` + `module_board_me*.stl` đang được **suy đoán**
    là cụm Jetson dựa vào thể tích (70 cm³), tên trong CAD không nói rõ.
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

### 🔴 ƯU TIÊN CAO NHẤT: gain vòng vận tốc hiện tại KHÔNG cân bằng nổi robot

Phát hiện từ mô phỏng 2026-08-10, nhưng đây là **kết luận về phần cứng**, không
phải lỗi sim:

- Board đang để `vel_gain = 0.25 N·m/(turn/s)` = 0.0398 N·m/(rad/s).
- Quán tính bánh (từ `mass_model`) J = 0.0122 kg·m².
- → hằng số thời gian vòng vận tốc = J/kp = **307 ms**.
- Trong khi hằng số thời gian đổ của chính con robot = √(h/g) với h = 0.2875 m
  → **171 ms**.

**Vòng trong chậm hơn cái mà nó phải ổn định.** Không có bộ điều khiển cân bằng
nào — RL hay LQR hay MPC — làm việc được qua một vòng vận tốc như vậy: bánh còn
đang trên đường đạt tốc độ đặt thì robot đã ngã xong. Riêng khâu tích phân cũng
không cứu được: `vel_integrator_gain = 0.15` cần ~4 s để tích đủ momen giữ một
góc nghiêng 5°.

**Giá trị đích đã chốt trong sim** (`javis/sim_config.py`, `DrivetrainCfg` —
viết thẳng bằng **đơn vị gốc của ODrive**, gõ vào board là xong, không phải quy
đổi gì):

```
axis0.controller.config.vel_gain            = 15.0    # hiện tại 0.25  (60×)
axis0.controller.config.vel_integrator_gain = 75.0    # hiện tại 0.15  (500×)
```

Chọn bằng `scripts/tune_sim_gains.py`, quét gain trên cả bánh có tải lẫn bánh
không tải và chấm điểm **đúng công thức `scripts/tune_wheel_pid.py` đang dùng
trên phần cứng thật**. Kết quả: vòng vận tốc còn **5.1 ms** khi bánh quay tự do
và **49.9 ms** khi đang đẩy robot — đều nằm gọn trong 171 ms mà bài toán cân
bằng cho phép, tức bánh xe trở thành một "nguồn vận tốc" gần lý tưởng, đúng
như policy đang giả định.

### Cách tune lên dần cho an toàn

Đừng nhảy một phát từ 0.25 lên 15. Với Kt = 0.207 N·m/A, `vel_gain = 15` làm
dòng chạm `current_lim` 15 A chỉ với sai số vận tốc **1.3 rad/s** — nghĩa là
sai lệch lớn một chút là ra momen tối đa ngay (đúng cái ta muốn khi cứu thăng
bằng), nhưng nhiễu encoder cũng bị khuếch đại gấp ~60 lần.

- [ ] **Kê robot lên giá, bánh không chạm đất.** Tạm hạ `current_lim` xuống
      ~5 A trong lúc dò.
- [ ] Đi từng nấc, mỗi nấc lệnh một bước vận tốc rồi nghe/nhìn:
      `0.25 → 1 → 3 → 6 → 10 → 15`. Tạm để `vel_integrator_gain = 5 × vel_gain`
      (đúng quy tắc ODrive `0.5 × bandwidth × vel_gain` ở bandwidth 10 Hz).
- [ ] Nấc nào bắt đầu **rung/hú ở trạng thái đứng yên** thì lùi lại 30–50% —
      đó là trần do nhiễu encoder, không phải do lý thuyết.
      Giá trị dừng được: vel_gain ______, vel_integrator_gain ______
- [ ] Trả `current_lim` về 15 A, thử lại có tải.
- [ ] Nếu trần thật thấp hơn 15 nhiều (VD chỉ tới 6), **báo lại để hạ
      `DrivetrainCfg.vel_gain` trong sim cho khớp** — sim phải chạy đúng con số
      phần cứng làm được, không phải ngược lại.

- [ ] **Mở rộng dải quét của `scripts/tune_wheel_pid.py`.** Hiện chỉ quét
      0.15–0.40, và quét trên bánh **quay tự do** — ở chế độ đó gain thấp luôn
      cho điểm đẹp nhất, nên bộ quét sẽ luôn chọn sai cho bài toán cân bằng.
      Cần quét lại khoảng 1–20.

> ⚠️ Lưu ý về sim: vòng PI trong `javis/mdp/actions.py` tính momen **tường minh
> mỗi bước vật lý**, nên gain và timestep không độc lập nhau (`kp·dt/J < 1`).
> Vì thế task đã đổi sang **timestep 2.5 ms (400 Hz), decimation 4** để giữ
> **100 Hz điều khiển** (mục 6 bên dưới — số theo trí nhớ, chưa đo). Board thật
> chạy vòng này ở 8 kHz nên không vướng giới hạn đó — đây thuần tuý là ràng
> buộc của mô phỏng. Đổi `CONTROL_HZ` mà không chia hết cho timestep vật lý,
> hoặc gain quá cao so với timestep, thì `OdriveVelocityAction` sẽ raise thẳng
> chứ không âm thầm phân kỳ.

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
- [ ] **Tune lại `vel_gain`/`vel_integrator_gain` trên ODrive thật** — xem ô
      🔴 đầu mục 3. Board đang ở `vel_gain = 0.25` (không phải 0.02 như ghi chú
      cũ ở đây; `scripts/setup_odrive.py` và bản dump USB đều xác nhận 0.25).
      Mục tiêu ~3.8. Tăng dần trên robot thật (tăng tới khi hơi dao động rồi
      lùi lại 30-50%), **không áp thẳng một phát** — gain gấp 15× mà đặt đột
      ngột thì gần như chắc chắn dao động.
      Ghi chú: `damping = 0.028` fit được từ log 0.2s là một **gain điều khiển**
      xấp xỉ vòng kín cũ, không phải ma sát vật lý của bánh. Sim giờ tách hai
      thứ này: PI loop mô phỏng riêng (`javis/mdp/actions.py`), còn ma sát nhớt
      vật lý của bánh là một `joint_damping` riêng, chưa đo, đang DR trong dải
      0–0.03 N·m/(rad/s).
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

## 5b. ✅ Encoder "hội tụ sai commutation" — ĐÃ TÌM RA GỐC RỄ THẬT: khe hở nam châm quá gần (2026-08-07)

**Cập nhật lớn (2026-08-07)**: sau nhiều tháng nghi là lỗi ngẫu nhiên/EMI
không rõ nguyên nhân, đã **xác định được nguyên nhân gốc thật**: khoảng
cách (air gap) giữa nam châm gắn trên trục và chip encoder AS5047P **quá
gần (< 0.5mm)** trên cả 2 board. Từ trường tại chip bị bão hòa ở khoảng
cách đó, khiến chip đọc ra góc sai **một cách hệ thống, lặp lại** — không
phải hội tụ sai ngẫu nhiên như từng nghĩ. Đây là lý do hiệu chuẩn lại nhiều
lần trước đây luôn ra cùng 1 kiểu lỗi (dòng cao ổn định ~3-9A, vận tốc≈0,
không báo lỗi) — hiệu chuẩn không sửa được vì bản thân *tín hiệu vào* đã
sai, không phải logic hiệu chuẩn sai.

**Cách xác nhận + khắc phục**: nới khoảng cách nam châm lên **~1mm** trên
từng board (thao tác tay, tháo lắp cơ khí) → hiệu chuẩn lại
(`calibrate_encoder_with_retry` + `verify_and_fix_calibration`) → **cả 2
board pass ngay lần đầu**: LEFT `vel=1.35 turn/s` (target 1.5) `Iq=0.45A`,
RIGHT `vel=1.32 turn/s` `Iq=0.53A` — hoàn toàn bình thường, dòng thấp,
tracking tốt. Đối chiếu trước/sau cực kỳ rõ ràng (trước: vel≈0, Iq kẹt
3-9A trên cả 2 board dù hiệu chuẩn lại bao nhiêu lần cũng vậy).

**Vẫn còn placeholder tài liệu cộng đồng** (mục dưới, giữ lại tham khảo):
tài liệu MKS xDrive Mini cộng đồng có nhắc tới "noise-related encoder
errors during initialization" — không rõ có phải cùng nguyên nhân khe hở
nam châm hay là vấn đề khác, vì tài liệu đó không nói rõ chi tiết cơ khí.
Coi đây là 2 khả năng riêng biệt cho tới khi có thêm bằng chứng.

- [x] **Nguyên nhân gốc encoder "hội tụ sai commutation"**: khe hở nam
      châm quá gần (<0.5mm) — đã xác nhận + khắc phục 2026-08-07 bằng cách
      nới lên ~1mm trên cả 2 board. `vel_gain`/`vel_integrator_gain` đã
      khôi phục về giá trị kiểm chứng thật (0.3/0.2) sau khi 1 lần tuning
      tự động trước đó bị nhiễu bởi dữ liệu từ lúc encoder còn lỗi (đã lưu
      đè giá trị sai 0.35/0.40 — đã sửa lại).
- [ ] **Đo lại khoảng cách chính xác tối ưu** (không chỉ "khác <0.5mm là
      được") — 1mm mới là điểm test đầu tiên thành công, chưa quét dải để
      tìm khoảng tối ưu/dung sai chấp nhận được. Nếu có datasheet AS5047P
      chính thức, đối chiếu lại dải từ trường khuyến nghị (từ trí nhớ,
      CHƯA xác nhận: khoảng 30-70mT tại chip) để tính khe hở tối ưu theo
      đúng loại nam châm đang dùng.
- [ ] **Việc cần làm khi viết driver ROS2/phần mềm điều khiển thật** (mục 6
      bên dưới): vẫn nên giữ cơ chế `verify_and_fix_calibration`
      (retry + tự kiểm tra sau hiệu chuẩn) như một lớp bảo vệ, dù nguyên
      nhân gốc đã biết — phòng trường hợp khe hở nam châm trôi lại theo
      thời gian/rung động khi robot vận hành thật (chưa biết độ bền cơ khí
      của việc gắn nam châm hiện tại).
- [ ] **Cố định chắc chắn vị trí nam châm sau khi chỉnh** (keo/vít khoá) —
      hiện tại mới chỉnh tay để test, chưa có biện pháp giữ cố định lâu
      dài chống trôi do rung động khi robot di chuyển thật.

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

**✅ Quyết định giao tiếp (2026-08-07): USB, không dùng CAN.** Sau khi phát
hiện chip transceiver CAN trên board bánh phải hỏng phần cứng (nhiều giờ
debug, xem toàn bộ lịch sử trong `MKS_XDRIVE_MINI.md`), quyết định chuyển
hẳn sang **USB** (native Fibre protocol, package `odrive` Python) làm giao
tiếp chính giữa Jetson và 2 board ODrive — lý do: chỉ có 2 động cơ, lợi thế
bus-dùng-chung của CAN không đáng để đánh đổi lấy rủi ro phần cứng đã gặp
phải. `can0` trên Jetson vẫn UP sẵn (`can0-up.service`) nhưng không còn là
hướng phát triển driver chính nữa — có thể bỏ qua phần CAN dưới đây trừ khi
sau này đổi ý.

- [ ] **Viết driver USB↔ODrive thật cho ROS2/phần mềm điều khiển** (chưa có
      code nào trên Jetson) — dùng thẳng package `odrive` Python
      (`odrive.find_any(serial_number=...)` để phân biệt 2 board theo số
      serial USB, xem `scripts/setup_odrive.py` đã dùng đúng cách này suốt
      phiên debug), gửi `axis0.controller.input_vel` mỗi bước điều khiển,
      đọc `axis0.encoder.pos_estimate`/`vel_estimate` làm observation. Tái sử
      dụng logic calibrate+verify+retry đã có trong `scripts/setup_odrive.py`
      (xem `MKS_XDRIVE_MINI.md` mục hiệu chuẩn) — PHẢI chạy lại mỗi lần robot
      khởi động thật.
      ⚠️ Cần cơ chế tự phát hiện + kết nối lại nếu board rớt khỏi USB khi
      đang vận hành (đã quan sát thấy hiện tượng này vài lần lúc debug,
      nguyên nhân chưa rõ — có thể do enumeration chập chờn khi có nhiều
      thiết bị USB cùng lúc, cần driver thật xử lý robust hơn script test).

<details>
<summary>Phần CAN cũ (không còn là hướng chính, giữ lại tham khảo)</summary>

- `can0` **UP**, 500 kbit/s, qua CAN controller onboard của chính Jetson
  (Tegra **MTTCAN**, `c310000.mttcan` — không phải qua USB-CAN adapter rời).
  Được cấu hình tự động lúc boot bằng service riêng đã viết sẵn:
  `/etc/systemd/system/can0-up.service` (`ip link set can0 type can bitrate
  500000` + `up`).
- Debug sâu (nhiều giờ, xem `MKS_XDRIVE_MINI.md`) xác nhận: board bánh trái
  phát/nhận CAN hoàn toàn bình thường, board bánh phải **không bao giờ phát
  được gì lên bus** dù config đúng 100% — cô lập được bằng cách tráo board
  qua cùng 1 dây/cùng 1 ESP32-S3 độc lập làm bộ nghe. Đo trực tiếp tại chip
  transceiver (VP230/SN65HVD230) trên board phải: VCC=3.3V (có nguồn), chân
  Rs=0V (đã mod đúng), nhưng CANH/CANL không tách ra khi ép chân D xuống GND
  — kết luận: **chip transceiver trên board phải hỏng phần cứng ở tầng lái
  ra bus**, không phải lỗi config/dây/termination.

</details>

- [ ] **Tần số vòng điều khiển thật trên robot** (Jetson gửi lệnh xuống
      ODrive bao nhiêu Hz?): __________ Hz
      → `javis/balance_task.py` (`CONTROL_HZ`) hiện để **100 Hz**, theo trí
      nhớ người dùng (USB/ROS2 round-trip thật "khoảng 100-150Hz") — **chưa
      phải số đo bằng timer ROS2 thật**, chỉ là ước lượng tạm thay cho số bịa
      50Hz trước đó. Đo xong thì sửa thẳng `CONTROL_HZ` (không phải
      `decimation` — `decimation` tự tính lại theo `CONTROL_HZ` và
      `PHYSICS_TIMESTEP_S`), rồi **train lại từ đầu** (đổi control rate làm
      thay đổi cả obs history length, không phải chỉnh nhẹ được).
      (Driver USB↔ODrive ở đầu mục này là phần thay thế
      `data.ctrl[...]`/`data.sensordata[...]` trong sim bằng I/O thật —
      driver CAN cũ không còn cần viết nữa, xem quyết định chuyển sang USB
      ở trên.)

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
