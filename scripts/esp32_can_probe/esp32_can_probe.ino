// ESP32-S3 CAN bus probe for JAVIS's 2x MKS xDrive Mini / ODrive boards.
//
// Passively listens on the CAN bus and prints every frame it sees, with
// ODrive heartbeat frames (cmd_id 0x01) flagged specially so you can
// confirm both boards (node_id 0 = right wheel, node_id 1 = left wheel,
// ghost axis1 = node_id 63 on both -- see MKS_XDRIVE_MINI.md) are alive
// and broadcasting correctly. Useful as an independent reference when a
// second listener (e.g. a Jetson) reports total silence on the same bus --
// if this sketch also sees nothing, the problem is upstream of both
// listeners (the bus/boards themselves); if it sees heartbeats fine, the
// problem is isolated to the other listener's own CAN receive path.
//
// Wiring: TX -> GPIO10, RX -> GPIO9 -> external 3.3V CAN transceiver ->
// CAN_H/CAN_L bus, 500 kbit/s (matches scripts/setup_odrive.py's
// odrv.can.set_baud_rate(500000)).
//
// Board: ESP32-S3, Arduino core (arduino-esp32) -- uses ESP-IDF's native
// TWAI driver directly, no extra library needed.

#include "driver/twai.h"
#include <string.h>

static const gpio_num_t CAN_TX_PIN = GPIO_NUM_10;
static const gpio_num_t CAN_RX_PIN = GPIO_NUM_9;

// ODrive CAN node IDs configured in scripts/setup_odrive.py.
// Right wheel restored to its real id (0) via USB after isolation testing.
static const uint32_t NODE_ID_RIGHT_WHEEL = 0;
static const uint32_t NODE_ID_LEFT_WHEEL = 1;
static const uint32_t NODE_ID_GHOST = 63;

// ODrive CAN Simple protocol: arbitration ID = (node_id << 5) | cmd_id.
static const uint32_t CMD_HEARTBEAT = 0x01;
static const uint32_t CMD_SET_VEL_GAINS = 0x1D;

// --- Active receive test -----------------------------------------------
// The passive listen above only proves whether a board can TRANSMIT. To
// separately test whether a board can RECEIVE at all (relevant for the
// SN65HVD230 "listen only" hardware defect, which breaks TX but leaves RX
// working), this sends a harmless Set_Vel_Gains command to each node every
// 2s -- harmless because the axis is IDLE (not closed-loop), so changing a
// controller gain has zero physical effect, and it's trivially reversible.
// After running this a while, read back axis0.controller.config.vel_gain
// over USB on both boards (venv/bin/python3 + odrive.find_any(serial_number=...)):
//   - LEFT (node 1) is the control group -- it should read back ~0.31.
//     If it does NOT, the assumed cmd_id/byte layout below is wrong for
//     this firmware fork and this test is inconclusive for BOTH boards --
//     fix the encoding before drawing any conclusion about the right board.
//   - RIGHT (node 2) is the board under test. If LEFT shows 0.31 but RIGHT
//     still shows its old value (0.3), that's direct proof the right
//     board's CAN receive path is also dead, not just its transmit path.
static const float TEST_VEL_GAIN = 0.31f;
static const float TEST_VEL_INTEGRATOR_GAIN = 0.2f;  // matches setup_odrive.py default, unchanged

// Disabled by default: if NOTHING on the bus ACKs a frame, the controller
// auto-retries that same frame at the hardware level forever, occupying the
// TX slot and making every later twai_transmit() fail to queue -- this can
// make bus health harder to read, not easier. Flip to true only once plain
// passive listening below confirms real heartbeats are coming through.
static const bool ENABLE_ACTIVE_SEND_TEST = false;

void sendSetVelGains(uint32_t node_id, float vel_gain, float vel_integrator_gain) {
  twai_message_t msg = {};
  msg.identifier = (node_id << 5) | CMD_SET_VEL_GAINS;
  msg.data_length_code = 8;
  memcpy(&msg.data[0], &vel_gain, 4);
  memcpy(&msg.data[4], &vel_integrator_gain, 4);

  esp_err_t result = twai_transmit(&msg, pdMS_TO_TICKS(100));
  Serial.printf("[%lu ms] >> sent Set_Vel_Gains to node=%lu (vel_gain=%.3f): %s\n",
                millis(), (unsigned long)node_id, vel_gain,
                result == ESP_OK ? "queued ok" : "FAILED to queue");
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println();
  Serial.println("=== JAVIS CAN probe (ESP32-S3) ===");

  twai_general_config_t g_config = TWAI_GENERAL_CONFIG_DEFAULT(
      CAN_TX_PIN, CAN_RX_PIN, TWAI_MODE_NORMAL);
  // Larger RX queue so a burst of heartbeats from both boards doesn't get
  // dropped while we're busy printing the previous frame over serial.
  g_config.rx_queue_len = 32;

  twai_timing_config_t t_config = TWAI_TIMING_CONFIG_500KBITS();
  twai_filter_config_t f_config = TWAI_FILTER_CONFIG_ACCEPT_ALL();

  if (twai_driver_install(&g_config, &t_config, &f_config) != ESP_OK) {
    Serial.println("FATAL: twai_driver_install failed");
    while (true) delay(1000);
  }
  if (twai_start() != ESP_OK) {
    Serial.println("FATAL: twai_start failed");
    while (true) delay(1000);
  }

  Serial.printf("TWAI up: TX=GPIO%d RX=GPIO%d bitrate=500000\n", CAN_TX_PIN,
                CAN_RX_PIN);
  Serial.println("Listening... (expect a heartbeat every ~100ms from node 0 and node 1)");
  Serial.println();
}

// Best-effort heartbeat decode. Field layout matches the common ODrive CAN
// Simple protocol (axis_error: u32, axis_state: u8, plus a couple of flag/
// result bytes) -- if your firmware's exact byte layout differs slightly,
// the raw hex dump above this line is still the authoritative signal: what
// matters for this test is simply whether a frame with this ID arrives at
// all, not decoding every field perfectly.
void printHeartbeat(uint32_t node_id, const twai_message_t &msg) {
  if (msg.data_length_code < 5) {
    Serial.println("  (heartbeat frame too short to decode)");
    return;
  }
  uint32_t axis_error = (uint32_t)msg.data[0] | ((uint32_t)msg.data[1] << 8) |
                         ((uint32_t)msg.data[2] << 16) |
                         ((uint32_t)msg.data[3] << 24);
  uint8_t axis_state = msg.data[4];

  const char *who = "unknown node";
  if (node_id == NODE_ID_RIGHT_WHEEL) who = "RIGHT wheel (node 0)";
  else if (node_id == NODE_ID_LEFT_WHEEL) who = "LEFT wheel (node 1)";
  else if (node_id == NODE_ID_GHOST) who = "ghost axis1 (node 63, expected silent/unused)";

  Serial.printf("  ^^ HEARTBEAT from %s: axis_error=0x%08lX axis_state=%u%s\n",
                who, (unsigned long)axis_error, axis_state,
                axis_error != 0 ? "  <-- ERROR SET" : "");
}

static unsigned long last_test_send_ms = 0;

// TWAI does NOT auto-recover from BUS_OFF on its own (unlike SocketCAN's
// restart-ms) -- once tx_error_counter trips bus-off, every subsequent
// twai_transmit() fails silently forever until twai_initiate_recovery() is
// called and the controller is given time to complete the recovery sequence
// (128 occurrences of 11 consecutive recessive bits, per CAN spec). Discovered
// the hard way: this sketch's own active-test sends drove it into bus-off,
// which then looked identical to "the bus is dead" from the outside.
void recoverFromBusOffIfNeeded() {
  twai_status_info_t status;
  twai_get_status_info(&status);
  if (status.state != TWAI_STATE_BUS_OFF) return;

  Serial.println("!! TWAI is in BUS_OFF -- initiating recovery...");
  twai_initiate_recovery();
  unsigned long start = millis();
  while (millis() - start < 2000) {
    twai_get_status_info(&status);
    if (status.state != TWAI_STATE_BUS_OFF) break;
    delay(50);
  }
  if (status.state == TWAI_STATE_RUNNING) {
    Serial.println("!! Recovered, back to RUNNING.");
  } else {
    Serial.printf("!! Recovery attempt ended in state=%d (not RUNNING) -- may need twai_start() too.\n",
                  status.state);
    twai_start();
  }
}

void loop() {
  recoverFromBusOffIfNeeded();

  // Active receive test: alternate sending to the known-good left board
  // (control group) and the right board under test, every 2s.
  if (ENABLE_ACTIVE_SEND_TEST && millis() - last_test_send_ms > 2000) {
    last_test_send_ms = millis();
    sendSetVelGains(NODE_ID_LEFT_WHEEL, TEST_VEL_GAIN, TEST_VEL_INTEGRATOR_GAIN);
    sendSetVelGains(NODE_ID_RIGHT_WHEEL, TEST_VEL_GAIN, TEST_VEL_INTEGRATOR_GAIN);
  }

  twai_message_t msg;
  if (twai_receive(&msg, pdMS_TO_TICKS(1000)) == ESP_OK) {
    uint32_t node_id = msg.identifier >> 5;
    uint32_t cmd_id = msg.identifier & 0x1F;

    Serial.printf("[%lu ms] ID=0x%03lX (node=%lu cmd=0x%02lX) DLC=%u data=",
                  millis(), (unsigned long)msg.identifier,
                  (unsigned long)node_id, (unsigned long)cmd_id,
                  msg.data_length_code);
    for (int i = 0; i < msg.data_length_code; i++) {
      Serial.printf("%02X ", msg.data[i]);
    }
    Serial.println();

    if (cmd_id == CMD_HEARTBEAT) {
      printHeartbeat(node_id, msg);
    }
  } else {
    // No frame arrived in the last second -- print bus/error state so
    // silence is visible and distinguishable from "still booting".
    twai_status_info_t status;
    twai_get_status_info(&status);
    Serial.printf(
        "[%lu ms] ...no frame in 1s (state=%d tx_err=%lu rx_err=%lu bus_err_count=%lu)\n",
        millis(), status.state, (unsigned long)status.tx_error_counter,
        (unsigned long)status.rx_error_counter,
        (unsigned long)status.bus_error_count);
  }
}
