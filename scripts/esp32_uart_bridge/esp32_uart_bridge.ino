// ESP32-S3 <-> ODrive UART bridge, stage 1 (single channel).
//
// Purpose: test whether talking to the MKS xDrive Mini's ASCII protocol over
// UART (instead of direct USB from the host) avoids the intermittent
// encoder.error=4 (ENCODER_ERROR_NO_RESPONSE) seen on the left wheel's board
// when driven directly over USB -- hypothesis: heavy USB traffic on the
// ODrive's own STM32 may be delaying its internal SPI polling to the
// AS5047P encoder chip. Routing through UART instead (a different firmware
// code path on the ODrive) removes that specific interference, regardless
// of what the host (Jetson) uses upstream of this bridge.
//
// This sketch does ONE thing: transparently forwards bytes both directions
// between the ESP32-S3's native USB serial (Serial, seen by the host as a
// normal /dev/ttyACMx port) and a hardware UART (Serial1) wired to the
// ODrive's GPIO1/GPIO2. No protocol parsing -- the ODrive ASCII protocol is
// plain text, so a dumb byte pipe is all that's needed. The host runs the
// exact same commands it would send directly (e.g. "v 0 1.5 0\n",
// "r vbus_voltage\n") -- this bridge is invisible to that layer.
//
// Wiring (see MKS_XDRIVE_MINI.md for the general UART pinout background):
//   ESP32-S3 GPIO17 (TX1) -> ODrive GPIO2 (RX)
//   ESP32-S3 GPIO18 (RX1) <- ODrive GPIO1 (TX)
//   ESP32-S3 GND           - ODrive GND
//   (no VCC line -- ODrive is separately powered; only signal + GND shared)
//
// Board: ESP32-S3, Arduino core. Uses native USB CDC (Serial) for the host
// link and hardware UART1 (Serial1) for the ODrive link.

#include <HardwareSerial.h>

static const int ODRIVE_UART_TX_PIN = 17;
static const int ODRIVE_UART_RX_PIN = 18;
static const uint32_t ODRIVE_BAUD = 115200;  // matches config.uart_baudrate on the board

HardwareSerial OdriveSerial(1);  // UART1

void setup() {
  Serial.begin(115200);  // USB CDC to host (Jetson/PC)
  OdriveSerial.begin(ODRIVE_BAUD, SERIAL_8N1, ODRIVE_UART_RX_PIN, ODRIVE_UART_TX_PIN);

  // Give the host a moment to open the port before printing anything --
  // avoid polluting the byte stream the ODrive ASCII protocol expects.
  delay(300);
}

void loop() {
  // Host -> ODrive
  while (Serial.available()) {
    OdriveSerial.write(Serial.read());
  }
  // ODrive -> Host
  while (OdriveSerial.available()) {
    Serial.write(OdriveSerial.read());
  }
}
