# MKS xDrive Mini — ghi chú phần cứng

Ghi lại kiến thức về board điều khiển động cơ đang dùng cho 2 bánh JAVIS —
tổng hợp từ tài liệu cộng đồng (không phải tài liệu chính thức MKS, xem
nguồn ở cuối) + kiểm chứng thực tế trên phần cứng của robot này
(2026-08-06). Đọc trước khi đụng vào `scripts/setup_odrive.py` hoặc
`odriveconfig.txt`.

## ✅ Quyết định giao tiếp (2026-08-07): dùng USB, không dùng CAN

Sau nhiều giờ debug CAN bus (xem toàn bộ phần dưới) phát hiện chip
transceiver CAN trên board bánh phải bị hỏng phần cứng — quyết định **bỏ
CAN làm giao tiếp chính, chuyển hẳn sang USB** (native Fibre protocol qua
package `odrive` Python, giống hệt cách `scripts/setup_odrive.py` đã dùng
suốt từ đầu).

**Lý do**: robot này chỉ có đúng 2 động cơ — lợi thế chính của CAN (bus dùng
chung cho nhiều node, độ trễ xác định cho vòng điều khiển tần số rất cao)
không phát huy tác dụng ở quy mô này. Jetson có đủ cổng USB để cắm trực
tiếp cả 2 board cùng lúc. USB đã chứng minh hoạt động ổn định suốt phiên
debug dài, trong khi CAN tốn rất nhiều thời gian vì đúng 1 lỗi phần cứng
trên 1 board.

**Rủi ro đã biết, chấp nhận**: đầu nối USB-C không bền bằng đầu vít/JST khi
robot rung lắc lúc di chuyển thật — cần cố định dây bằng dây rút khi lắp
đặt. Cũng từng thấy USB enumeration chập chờn trong lúc test (board rớt
khỏi `lsusb` vài lần không rõ lý do) — cần driver ROS2/phần mềm điều khiển
thật có cơ chế tự phát hiện + kết nối lại nếu board rớt kết nối khi vận
hành.

**Toàn bộ phần CAN bên dưới vẫn giữ lại để tham khảo** (lịch sử debug, kiến
thức phần cứng thật vẫn đúng) — chỉ là không còn là phương án giao tiếp
chính nữa. Có thể quay lại nếu sau này cần bus dùng chung cho nhiều thiết
bị hơn.

## Board này thực chất là gì

**MKS xDrive Mini = MKS ODrive Mini** (2 tên gọi cho cùng 1 board, thấy cả
2 cách gọi tuỳ nơi bán). Đây là **bản clone phần cứng ODrive v3.6**, chạy
firmware **ODrive v0.5.1 đã chỉnh sửa riêng** (không phải firmware ODrive
gốc/chính thức) — khớp chính xác với package `odrive` python (v0.5.1.post0)
đang dùng trong `venv`.

## ⚠️ CHỈ DÙNG 1 AXIS THẬT MỖI BOARD

Firmware nền tảng (ODrive v3.6) hỗ trợ 2 axis, nhưng **thiết kế phần cứng
của board này chỉ đưa ra chân cắm cho 1 động cơ** — `axis0`. `axis1` **tồn
tại trong firmware nhưng không có động cơ/encoder thật nào nối vào** — gọi
là "ghost axis".

- Robot này dùng **2 board riêng biệt, mỗi board 1 bánh** (không phải 1
  board dùng cả axis0 lẫn axis1 cho 2 bánh — đã thử và xác nhận: chạy
  calibrate trên axis1 của board đang cắm bánh phải → lỗi ngay vì không có
  động cơ thật ở đó).
- **Ghost axis1 vẫn cần cấu hình** khi dùng CAN bus: nếu để mặc định, cả 2
  board sẽ có axis1 với ID trùng nhau (hoặc trùng với axis0 thật của board
  kia), gây xung đột trên CAN bus. Cách khắc phục cộng đồng dùng:
  `axis1.config.can_node_id = 63` (ID "chỉ nghe", không dùng để giao tiếp
  thật) cho **cả 2 board**.
- `scripts/setup_odrive.py` đã tự làm việc này — luôn set `can_node_id`
  của axis1 = 63, và axis0 = ID riêng theo bánh (`--wheel right` → 0,
  `--wheel left` → 1).

## Encoder onboard

- Chip **AS5047P** (họ AMS) — SPI tuyệt đối, gắn ngay trên board (không
  phải rời/nối dây ra ngoài).
- Config đúng: `ENCODER_MODE_SPI_ABS_AMS`, `abs_spi_cs_gpio_pin = 7`.
  Khớp với số liệu đã dùng, xác nhận đúng bởi cả tài liệu cộng đồng lẫn
  thực nghiệm.

## ✅ Đã tìm ra gốc rễ (2026-08-07): khe hở nam châm-encoder quá gần

Sau nhiều lần nghi là lỗi ngẫu nhiên/EMI không rõ nguyên nhân (xem phần bên
dưới, giữ lại làm lịch sử), **đã xác nhận nguyên nhân gốc thật trên cả 2
board robot này**: khoảng cách (air gap) giữa nam châm gắn trên trục và
chip AS5047P **quá gần (<0.5mm)**, khiến từ trường tại chip bão hòa, đọc
góc sai một cách hệ thống. Nới khoảng cách lên **~1mm** trên cả 2 board →
hiệu chuẩn pass ngay lần đầu, dòng điện bình thường (<1A ở tốc độ test,
trước đó kẹt 3-9A không quay). Xem `SIM2REAL.md` mục 5b để biết đầy đủ số
liệu đối chiếu trước/sau.

**Chưa xác định**: đây có phải đúng "vấn đề đã biết" mà tài liệu cộng đồng
`Smurf/xdrive-mini-docs` nhắc tới bên dưới hay không (nguyên nhân họ ghi
không đủ chi tiết) — coi 2 phần dưới đây là bối cảnh cộng đồng, còn phần
này là kết quả kiểm chứng thật trên đúng 2 board của robot này.

## ⚠️ Vấn đề đã biết theo tài liệu cộng đồng: hiệu chuẩn encoder lúc khởi động không ổn định

Tài liệu cộng đồng (`Smurf/xdrive-mini-docs`) ghi nhận đây là **vấn đề đã
biết của chính board này**: "board sẽ không tự cấu hình đúng để dùng
encoder onboard" lúc khởi động, có lỗi liên quan tới nhiễu
(noise-related encoder errors during initialization).

**Cách khắc phục họ khuyến nghị** (đã áp dụng vào `scripts/setup_odrive.py`):
- **Để `startup_encoder_offset_calibration = False`** — KHÔNG để ODrive tự
  động hiệu chuẩn lúc boot (ngược với bản năng đầu tiên — bản mẫu gốc
  `odriveconfig.txt` "sample, not real robot" để `True`, tôi ban đầu cũng
  làm vậy trước khi tìm ra tài liệu này).
- Thay vào đó, **phần mềm điều khiển tự chạy hiệu chuẩn rõ ràng** mỗi lần
  khởi động thật, có khoảng nghỉ 100-200ms trước khi vào closed-loop
  control.
- **Xác nhận bằng thực nghiệm trên chính robot này**: kể cả sau khi hiệu
  chuẩn "thành công" (`is_ready=True`, không lỗi), động cơ đôi khi vẫn chạy
  sai (dòng cao, gần như không quay) — ODrive không tự phát hiện ra lỗi
  hội tụ sai góc commutation này. `scripts/calibrate_encoder_with_retry` +
  `verify_and_fix_calibration` trong `scripts/setup_odrive.py` xử lý cả 2
  trường hợp: lỗi cứng (`encoder.error != 0`) VÀ hiệu chuẩn "trông ổn nhưng
  chạy sai" (quay thử, đo tỉ lệ dòng/vận tốc).
- **Quan trọng cho phần mềm điều khiển thật sau này** (ROS2/driver — hiện
  chưa viết): PHẢI lặp lại đúng cơ chế hiệu chuẩn + tự kiểm tra + retry này
  mỗi lần robot khởi động thật, không được chỉ tin `is_ready=True` là đủ.

## ⚠️ Đặc tính chung của chip SN65HVD230/VP230 (⚠️ ĐÃ SỬA trích dẫn sai 2026-08-06)

**Đính chính quan trọng**: trước đây mục này khẳng định "cộng đồng đã xác
nhận lỗi thiết kế ngay trên board MKS xDrive Mini" dựa theo bài viết
Hackaday.io — kiểm tra lại nguồn gốc cho thấy đây là **trích dẫn sai**. Bài
viết đó liệt kê **"1 × SN65HVD230 CAN module"** như một linh kiện RỜI trong
danh sách BOM, dùng làm giao diện CAN phía bộ điều khiển trung tâm (vai trò
giống hệt module transceiver rời gắn phía Jetson) — **KHÔNG PHẢI** chip hàn
sẵn trên PCB của MKS xDrive Mini. Không có xác nhận cộng đồng nào cho việc
chip trên chính board MKS bị lỗi thiết kế này.

**Vẫn đúng và không đổi** (bằng chứng vật lý trực tiếp trên robot này,
2026-08-06): board MKS xDrive Mini bánh phải có 1 chip ghi **"VP230"** hàn
trên PCB — tên khác của cùng họ chip SN65HVD230 (tương thích chân, cùng
datasheet). Đặc tính **chung của cả họ chip này** (không riêng gì board
nào): chân Rs (chân 8) nối GND qua điện trở giá trị quá cao thay vì nối tắt
0 ohm sẽ khiến chip kẹt ở chế độ **"Listen Only"** (nhận được, không gửi
được) — đây là kiến thức chung về chip, áp dụng được cho bất kỳ board nào
dùng chip này, kể cả khi không có nguồn cộng đồng nào xác nhận cụ thể cho
đúng board MKS.

**Đã thử trên robot này**: tháo/nối tắt chân Rs của chip VP230 trên board
phải — không thay đổi kết quả (vẫn không phát được gì lên CAN bus, xác nhận
qua ESP32-S3 độc lập nhiều lần). Đã đo VCC chip = 3.3V (có nguồn), Rs = 0V
sau khi nối tắt (đúng), nhưng CANH/CANL vẫn không tách ra khi ép chân D
xuống GND (chưa xác nhận dứt điểm) — nghi vấn hiện tại: bản thân chip đã
hỏng hẳn (không chỉ là vấn đề Rs), cần thay chip hoặc đo thêm để xác nhận.

## ⚠️ TUYỆT ĐỐI không update firmware qua odrivetool

Board chạy firmware v0.5.1 **đã bị chỉnh sửa riêng** cho phần cứng này.
Lệnh `odrivetool` "upgrade" (hoặc bất kỳ lệnh DFU/flash firmware chuẩn nào
khác) sẽ **ghi đè bằng firmware ODrive gốc không tương thích, làm brick
board**. Nếu lỡ brick, cách khôi phục duy nhất là dùng máy nạp **ST-Link**
nạp lại firmware gốc đã dump sẵn (xem repo GitHub ở nguồn tham khảo).

## Cấu hình CAN của robot này

| | axis0 (động cơ thật) | axis1 (ghost) |
|---|---|---|
| Board bánh phải | `can_node_id = 0` | `can_node_id = 63` |
| Board bánh trái | `can_node_id = 1` | `can_node_id = 63` |

Baud rate: 500 kbit/s (`odrv0.can.set_baud_rate(500000)`), cả 2 board.

## Nguồn tham khảo

- [Getting Started — ODrive Documentation](https://docs.odriverobotics.com/) —
  tài liệu ODrive chính thức (v0.5.x/v0.6.x), dùng để đối chiếu API chung.
- [MKS XDrive Mini Guide: Set up, Tuning & Arduino (Hackaday.io)](https://hackaday.io/project/204985-mks-xdrive-mini-guide-set-up-tuning-arduino) —
  nguồn cho: xác nhận firmware v0.5.1 chỉnh sửa riêng, chip AS5047P onboard,
  cảnh báo không upgrade firmware. ⚠️ KHÔNG phải nguồn cho lỗi CAN
  transceiver trên chính board MKS (đã sửa 2026-08-06) — bài viết này mô tả
  1 module SN65HVD230 RỜI dùng phía bộ điều khiển trung tâm của tác giả,
  không phải chip hàn trên PCB MKS. Đặc tính chân Rs vẫn là kiến thức chung
  của họ chip SN65HVD230/VP230, không phải trích dẫn riêng cho board này.
- [justlovescience/MKS-XDRIVE-MINI (GitHub)](https://github.com/justlovescience/MKS-XDRIVE-MINI) —
  nguồn cho: cách xử lý ghost axis1 (`can_node_id = 63`), cảnh báo brick
  firmware + quy trình cứu bằng ST-Link.
- [Smurf/xdrive-mini-docs (GitHub)](https://github.com/Smurf/xdrive-mini-docs) —
  nguồn cho: vấn đề hiệu chuẩn encoder lúc khởi động không ổn định, cấu
  hình `abs_spi_cs_gpio_pin=7`/`ENCODER_MODE_SPI_ABS_AMS`, khuyến nghị tắt
  `startup_encoder_offset_calibration`.

Đây là tài liệu cộng đồng, không phải tài liệu chính thức từ MKS/Makerbase
— đối chiếu lại nếu có bản chính thức xuất hiện sau này.

## Liên quan trong repo này

- `scripts/setup_odrive.py` — script cấu hình + hiệu chuẩn, áp dụng toàn bộ
  ghi chú ở trên.
- `odriveconfig.txt` — bản chú thích chi tiết từng dòng config (dựa theo
  file mẫu ban đầu, nay đã lạc hậu hơn `setup_odrive.py` — ưu tiên tin
  script hơn file này nếu có mâu thuẫn).
- `SIM2REAL.md` mục 3 và 5b — checklist đo đạc thật + lịch sử debug lỗi
  encoder chập chờn trên chính robot này.
