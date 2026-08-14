# DEPLOY.md — Từ checkpoint sim tới robot thật

Nhánh `simulation` không chỉ tạo ra một policy — nó khép lại vài mục "chưa
quyết định" trong `PLAN.md`, và phát hiện một điều kiện tiên quyết **bắt buộc**
mà `PLAN.md` chưa hề biết tới. Tài liệu này là bản đối chiếu: cái gì đã sẵn
sàng, cái gì đã đổi so với kế hoạch cũ, cái gì vẫn thiếu, và thứ tự làm cụ thể.

Không lặp lại toàn bộ `PLAN.md`/`SIM2REAL.md` — tài liệu này trỏ tới chúng và
chỉ nói phần đã thay đổi hoặc mới phát hiện.

---

## 🔴 Điều kiện tiên quyết duy nhất chặn mọi thứ khác: retune gain ODrive

Không liên quan gì tới ROS2, driver, hay policy — đây là việc phần cứng phải
làm trước, độc lập với toàn bộ phần mềm còn lại, và không có nó thì mọi
checkpoint train ra đều vô nghĩa trên robot thật.

Board đang chạy `vel_gain = 0.25`. Với quán tính bánh 0.0122 kg·m², đó là vòng
vận tốc **307 ms** — chậm hơn hằng số thời gian đổ của robot (**171 ms**).
Không bộ điều khiển cân bằng nào (RL, LQR, MPC) làm việc được qua một vòng
trong chậm hơn thứ nó phải ổn định. Chi tiết & đường tune an toàn:
`SIM2REAL.md` mục 3.

```
axis0.controller.config.vel_gain            = 15.0   # hiện tại 0.25  (60×)
axis0.controller.config.vel_integrator_gain = 75.0   # hiện tại 0.15  (500×)
```

Mọi checkpoint train được đều giả định drivetrain đạt gần những số này
(`javis/sim_config.py`, `DrivetrainCfg`). Nếu board thật chỉ lên được gain
thấp hơn, **đo lại số thật rồi kiểm tra bằng**:

```bash
.venv/bin/python scripts/eval_payload_sweep.py --checkpoint <ckpt mới> \
    --vel-gain <đo được> --vel-integrator-gain <đo được>
```

trước khi tin rằng policy còn cân bằng được ở gain đó.

---

## Đã giải quyết so với `PLAN.md`

| Mục trong PLAN.md | Trạng thái trước | Bây giờ |
|---|---|---|
| Mục 7: "format export policy... chưa quyết định (TorchScript/ONNX?)" | Chưa chốt | **ONNX**, `scripts/export_onnx.py`, parity torch↔onnxruntime ~2e-6 |
| "Observation cần khớp 1-1... sai 1 chỗ là chạy sai không báo lỗi" | Chỉ có trong code, phải tự đọc | Xuất ra `*_contract.json` máy đọc được, kèm cảnh báo layout term-major (xem bên dưới) — không cần đoán từ `balance_task.py` nữa |
| Khối lượng/CoM ước lượng theo thể tích đồng đều | Trọng tâm gần chắc chắn sai (xem README cũ) | Mô hình theo nhóm linh kiện thật, đối chiếu 2 nguồn độc lập, 32 phép kiểm tra tự động (`scripts/verify_mass_model.py`) |
| Mục 6 bảng "action → data.ctrl áp ngay lập tức, không độ trễ" | Sim lý tưởng hoá hoàn toàn | Vòng PI ODrive mô phỏng tường minh (`javis/mdp/actions.py`) + trễ observation ngẫu nhiên 1-3 bước — thu hẹp khoảng cách, **không xoá hết** (độ trễ USB thật vẫn chưa đo, mục 7 PLAN.md) |
| — | — | **Mới, PLAN.md chưa biết**: gain ODrive hiện tại không đủ nhanh để cân bằng — xem mục đỏ ở trên |

Vẫn **chưa làm**, đúng như PLAN.md mục 8 đã liệt kê: `javis_odrive_driver`,
`policy_inference_node`, e-stop phần cứng/phần mềm, nguồn twist command thật.
Nhánh này không đụng tới phần ROS2 — chỉ chuẩn bị artifact (policy) và tài
liệu (hợp đồng I/O) để phần đó dễ viết đúng hơn.

---

## Hợp đồng I/O chính xác

Nguồn chân lý là `*_contract.json` sinh tự động bởi `scripts/export_onnx.py`
cùng lúc với mỗi checkpoint — không gõ tay, không copy số cũ.
`checkpoints/javis_payload_flat/model_1499_contract.json` hiện đã **lỗi thời**
(384 → xem cảnh báo ở `checkpoints/README.md`) — số dưới đây là hợp đồng của
code hiện tại (`javis/balance_task.py`, `CONTROL_HZ=100`,
`OBS_HISTORY_LENGTH=24`), đúng cho checkpoint kế tiếp sẽ train.

### Observation — 384 giá trị, **TERM-MAJOR**, không phải frame-major

Đây là chỗ dễ sai nhất theo đúng cảnh báo PLAN.md đã nêu. Toàn bộ vector KHÔNG
phải là 24 khung hình liên tiếp mỗi khung 16 giá trị. Nó là:

```
[ base_lin_vel: 24 khung cũ→mới ][ base_ang_vel: 24 khung ][ projected_gravity: 24 khung ]
[ wheel_vel: 24 khung ][ actions: 24 khung ][ command: 24 khung ]
```

24 khung × 10ms/khung (100Hz) = 240ms lịch sử — cùng khoảng thời gian vật lý
như bản 12 khung × 20ms (50Hz) trước đây, chỉ đổi số khung để giữ nguyên cửa
sổ thời gian khi control rate tăng lên.

| Term | Rộng | Nguồn thật trên robot | Đã có sẵn? |
|---|---|---|---|
| `base_lin_vel` | 24×3 | **KHÔNG phải IMU** — xem mục sửa lỗi bên dưới. Cần odometry từ `orin_vslam_bringup` (cuVSLAM) | ⚠️ Pipeline có sẵn nhưng chưa nối vào `policy_inference_node` |
| `base_ang_vel` | 24×3 | `/bmx160_bmp388_node/imu/data_raw`, trường `angular_velocity` | ✅ có sẵn |
| `projected_gravity` | 24×3 | Cần **orientation đã fusion** (`imu_filter_madgwick`), không lấy thẳng từ IMU thô | ⚠️ Launch file có điều kiện, cần bật (PLAN.md mục 3.2) |
| `wheel_vel` | 24×2 | `/wheel_states` (`javis_odrive_driver` publish, **rad/s**, đổi từ `vel_estimate` turn/s: `×2π`) | ❌ driver chưa viết |
| `actions` | 24×2 | Policy tự lưu lại action bước trước, không qua ROS2 topic | — (nằm trong `policy_inference_node`) |
| `command` | 24×3 | `/cmd_vel`-kiểu twist, nguồn chưa quyết (PLAN.md mục 3.3, đề xuất teleop trước) | ❌ chưa có nguồn |

Mỗi term còn có noise huấn luyện + trễ ngẫu nhiên. Từ bản retune này,
`base_lin_vel`/`base_ang_vel`/`projected_gravity` (3 term nguồn IMU/VSLAM)
còn cộng thêm **1 độ lệch (bias) cố định suốt 1 episode**, không chỉ nhiễu
trắng mỗi bước — mô phỏng gần hơn hành vi zero-rate offset của IMU MEMS thật
trong 1 chu kỳ nguồn (xem `javis/balance_task.py`, hàm `imu_noise`). Driver
thật **không cần** giả lập trễ hay bias (cả hai vốn đã có tự nhiên trên phần
cứng thật), chỉ cần đưa đúng số đo mới nhất mỗi tick.

### Action — 2 giá trị

| | |
|---|---|
| Thứ tự | `[left_wheel, right_wheel]` |
| Ý nghĩa | Vận tốc bánh mong muốn |
| Đơn vị | rad/s, **sau khi** nhân đầu ra mạng với `scale = 5.0` |
| Quy đổi ODrive | `input_vel [turn/s] = action [rad/s] / (2π)` — hộp số 1:1, không có tỉ số truyền (SIM2REAL.md mục 3) |
| Đích publish | `/wheel_left/cmd_vel`, `/wheel_right/cmd_vel` (`std_msgs/Float32`, rad/s — đúng bảng contract PLAN.md mục 5, robot ODrive vẫn nhận turn/s nên driver phải tự đổi lần nữa) |

### Tần số điều khiển: 100 Hz — theo trí nhớ người dùng, chưa phải số đo

Trước đó giả định 50Hz (số bịa, chưa test gì). Người dùng nhớ lại USB/ROS2
round-trip thật đạt khoảng 100–150Hz — đã chốt **100Hz** (đầu thấp của
khoảng nhớ được): policy train ở tốc độ điều khiển CHẬM hơn (ít lần sửa sai
hơn) là bài toán khó hơn, nên tổng quát hoá sang phần cứng chạy nhanh hơn lúc
train sẽ an toàn hơn chiều ngược lại. Vẫn **chưa phải số đo thật** — cần
benchmark bằng timer ROS2 thật (PLAN.md mục 7) để xác nhận/điều chỉnh lại.

Đổi `CONTROL_HZ` ở đầu `javis/balance_task.py` khi có số đo thật;
`decimation` tự tính lại theo (`assert` sẽ báo lỗi rõ ràng nếu con số không
chia hết cho timestep vật lý 2.5ms, thay vì âm thầm làm tròn sai). Physics
vẫn chạy 400Hz như trước — không liên quan tới đổi này, mà do vòng PI ODrive
cần timestep mịn để ổn định ở gain cao (`SIM2REAL.md` mục 3).

---

## Sửa một chỗ sai trong PLAN.md mục 3.2

PLAN.md liệt `base_lin_vel` cùng `base_ang_vel` là "từ IMU". Không đúng: IMU
(BMX160 — accelerometer + gyro) không đo được vận tốc dài, chỉ đo được gia
tốc và vận tốc góc. Trong sim, `base_lin_vel` đọc từ site sensor
`mjSENS_VELOCIMETER` — vận tốc dài chuẩn (ground truth). Nguồn thật gần nhất
là **odometry từ cuVSLAM** (`orin_vslam_bringup`, đã chạy sẵn trên Jetson cho
mục đích khác — nav2), chưa nối vào `policy_inference_node`. Nếu pipeline
VSLAM rớt khung hình, policy mất một observation nó đã học cùng — nên cân
nhắc train thêm một biến thể bỏ hẳn `base_lin_vel` nếu độ tin cậy VSLAM là
vấn đề (chưa làm trong nhánh này).

---

## Công cụ dùng được ngay, không cần chờ driver ROS2

| Việc | Lệnh |
|---|---|
| Lái thử policy bằng bàn phím trong sim, trước khi đụng phần cứng | `scripts/teleop_keyboard.py --checkpoint <ckpt mới>` (W/S/A/D, xem `--help`) |
| Xem training trực tiếp qua trình duyệt (kể cả khi đang chạy) | `scripts/watch_training.sh` |
| Kiểm tra policy có sống nổi ở gain thật đo được (nếu thấp hơn 15) | `scripts/eval_payload_sweep.py --vel-gain <n> --vel-integrator-gain <n>` |
| Đối chiếu khối lượng/CoM tính toán với số đo tay | `scripts/inspect_mass.py --check-model` |
| Xuất lại ONNX + hợp đồng I/O sau khi train tiếp | `scripts/export_onnx.py --checkpoint <ckpt mới>` |
| Video so sánh nhiều cấu hình tải, có vector target/current | `scripts/record_payload_video.py --checkpoint <ckpt>` |

---

## Checklist triển khai theo thứ tự (bổ sung vào PLAN.md mục 8)

1. **[MỚI, chặn tất cả]** Retune gain ODrive theo bậc thang trong `SIM2REAL.md`
   mục 3, trên giá đỡ, bánh không chạm đất.
2. `javis_odrive_driver` (PLAN.md mục 3.1) — test độc lập bằng
   `ros2 topic pub /wheel_left/cmd_vel ...`, chưa cần policy.
3. Benchmark tần số/độ trễ USB thật (PLAN.md mục 7). Nếu lệch đáng kể khỏi
   100 Hz (số hiện đang dùng, theo trí nhớ chưa đo — xem mục trên), sửa
   `CONTROL_HZ` trong `javis/balance_task.py` rồi **train lại** — đừng chỉnh
   ngược ở phía driver để "giả vờ" khớp con số trong sim.
4. Bật `imu_filter_madgwick` (cho `projected_gravity`), nối odometry VSLAM
   vào `policy_inference_node` (cho `base_lin_vel`) — xem mục sửa lỗi ở trên.
5. Viết `policy_inference_node`: nạp checkpoint mới nhất (`.onnx`), lắp buffer
   lịch sử **đúng thứ tự term-major**, publish theo bảng action ở trên. Test
   lại bằng `scripts/export_onnx.py`'s parity check nếu nghi ngờ.
6. E-stop phần cứng + phần mềm (PLAN.md mục 4) — bắt buộc trước khi có tải
   trọng/di chuyển thật, dù toàn bộ các bước trên đã xong.
7. Test trên giá đỡ trước (bánh không chạm đất) → test cân bằng tĩnh không
   tải → mới tới có tải, đối chiếu với `scripts/eval_payload_sweep.py` đã
   chạy trong sim cho đúng cấu hình đó.

---

## Trung thực về những gì chưa xong

- `checkpoints/javis_payload_flat/model_1499.pt` giờ **lỗi thời, không nạp
  được vào code hiện tại** (khác shape observation) — xem
  `checkpoints/README.md`. `logs/` đã xoá sạch để train lại từ đầu với
  `CONTROL_HZ=100`.
- `Javis-Payload-Rough` (địa hình đa dạng) đang được người dùng tự train,
  chưa có checkpoint để đóng gói.
- **100 Hz control rate là theo trí nhớ, không phải số đo** — benchmark thật
  trên Jetson vẫn là việc chưa làm quan trọng nhất còn lại (PLAN.md mục 7).
- 2 board MKS ODrive chưa cân thật (đang ước lượng 35g/board trong mô hình
  khối lượng) — xem `SIM2REAL.md` mục 1.
- Độ trễ USB thật: chưa đo (PLAN.md mục 7).
- Độ lệch (bias) IMU trong observation noise là ước lượng bậc-độ-lớn, không
  phải thông số datasheet BMX160 thật — xem `javis/balance_task.py`.
- Watchdog/e-stop: chưa có dòng code nào (PLAN.md mục 4) — đây là việc an
  toàn, không phải việc tối ưu, phải làm trước khi robot di chuyển tự do.
