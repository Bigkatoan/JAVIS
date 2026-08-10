#!/usr/bin/env python3
"""
JAVIS Position Control Web Test

Reference config adapted from justlovescience/MKS-XDRIVE-MINI (my_config.txt),
which the user confirmed works correctly on this exact board with a 5010 motor
(lower pole count). Corrected here for JAVIS's actual 8" hub motor:
  - pole_pairs = 15 (confirmed by counting 30 rotor magnets -- the reference
    file's pole_pairs=20 was tuned for their own, different motor)
  - vel_limit / trap_traj limits scaled way down from the reference's 120/30
    turns/s, which would be dangerous on an 8" wheel -- these are tuned for a
    small low-inertia motor, not ours.
  - current_lim, resistance_calib_max_voltage, encoder config kept the same
    as the reference (2V calib voltage, 16384 CPR SPI_ABS_AMS) since those
    aren't motor-size-dependent and match what JAVIS's setup_odrive.py uses.

This is a DIAGNOSTIC tool: this session found closed-loop velocity control
consistently fails on this board's 15-pole-pair motor (encoder proven good via
manual rotation test, current loop proven self-consistent, but no net torque
across every offset/gain/direction/bandwidth combination tried) while the same
board works with the user's other, lower-pole-count motor. Position control
shares the same encoder-offset-derived commutation angle, so it is expected to
show the same failure mode -- this tool exists to let the user verify that
directly and interactively rather than take it on faith.
"""

import asyncio
import json
import logging
import time
from aiohttp import web
import odrive
from odrive.enums import (
    AXIS_STATE_CLOSED_LOOP_CONTROL,
    AXIS_STATE_ENCODER_OFFSET_CALIBRATION,
    AXIS_STATE_MOTOR_CALIBRATION,
    AXIS_STATE_IDLE,
    CONTROL_MODE_POSITION_CONTROL,
    ENCODER_MODE_SPI_ABS_AMS,
    INPUT_MODE_TRAP_TRAJ,
    MOTOR_TYPE_HIGH_CURRENT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

LEFT_SERIAL = "318236823335"
RIGHT_SERIAL = "3676365D3335"

# --- Corrected for JAVIS's motor (see module docstring) ---
POLE_PAIRS = 15
CALIBRATION_CURRENT = 10.0
RESISTANCE_CALIB_MAX_VOLTAGE = 2.0
CURRENT_LIM = 15.0
REQUESTED_CURRENT_RANGE = 20.0

ENCODER_CS_GPIO_PIN = 7
ENCODER_CPR = 16384
ENCODER_BANDWIDTH = 3000.0
ENCODER_CALIB_RANGE = 10.0

DC_BUS_UNDERVOLTAGE = 19.0
DC_BUS_OVERVOLTAGE = 28.0
DC_MAX_POSITIVE_CURRENT = 20.0
DC_MAX_NEGATIVE_CURRENT = -1.0
BRAKE_RESISTANCE = 2.0

POS_GAIN = 20.0
VEL_GAIN = 0.3
VEL_INTEGRATOR_GAIN = 0.2
VEL_LIMIT = 8.0          # turns/s -- reference used 120, far too fast for an 8" wheel
TRAP_VEL_LIMIT = 4.0      # reference used 30
TRAP_ACCEL_LIMIT = 4.0    # reference used 5 (kept modest here too, just scaled to our vel range)
TRAP_DECEL_LIMIT = 4.0

MAX_POSITION = 8.0  # turns, safety clamp for the web slider


class ODriveController:
    def __init__(self):
        self.left_odrive = None
        self.right_odrive = None
        self.left_target_pos = 0.0
        self.right_target_pos = 0.0
        self.left_enabled = False
        self.right_enabled = False

        self.pos_gain = POS_GAIN
        self.vel_gain = VEL_GAIN
        self.vel_integrator_gain = VEL_INTEGRATOR_GAIN
        self.trap_vel = TRAP_VEL_LIMIT
        self.trap_accel = TRAP_ACCEL_LIMIT
        self.trap_decel = TRAP_DECEL_LIMIT

        self.telemetry_freq = 40.0
        self.actual_freq = 0.0
        self.busy = False
        self.status_msg = "Ready"

        self.dirty_left_target = False
        self.dirty_right_target = False

    def connect(self):
        logging.info("Connecting to ODrive boards...")
        self.status_msg = "Connecting to ODrives..."
        try:
            self.left_odrive = odrive.find_any(serial_number=LEFT_SERIAL, timeout=3)
            self.left_odrive.axis0.clear_errors()
            logging.info(f"Connected LEFT. Vbus={self.left_odrive.vbus_voltage:.2f}V")
        except Exception as e:
            logging.warning(f"LEFT connect failed: {e}")
            self.left_odrive = None
        try:
            self.right_odrive = odrive.find_any(serial_number=RIGHT_SERIAL, timeout=3)
            self.right_odrive.axis0.clear_errors()
            logging.info(f"Connected RIGHT. Vbus={self.right_odrive.vbus_voltage:.2f}V")
        except Exception as e:
            logging.warning(f"RIGHT connect failed: {e}")
            self.right_odrive = None
        self.status_msg = "ODrives connected."

    # --- full corrected config, adapted from justlovescience/MKS-XDRIVE-MINI ---
    def apply_full_config(self, board):
        if not board:
            return False
        try:
            serial = board.serial_number
        except Exception:
            logging.error("apply_full_config: could not read serial_number before erase")
            return False
        try:
            self.status_msg = f"Erasing + applying corrected config (pole_pairs=15) on {serial:x}..."
            logging.info(self.status_msg)
            try:
                board.erase_configuration()
            except Exception:
                pass  # erase reboots the board -- the call itself often drops the connection
            time.sleep(2.0)
            board = odrive.find_any(serial_number=f"{serial:012x}".upper(), timeout=15)

            board.config.brake_resistance = BRAKE_RESISTANCE
            board.config.dc_bus_undervoltage_trip_level = DC_BUS_UNDERVOLTAGE
            board.config.dc_bus_overvoltage_trip_level = DC_BUS_OVERVOLTAGE
            board.config.dc_max_positive_current = DC_MAX_POSITIVE_CURRENT
            board.config.dc_max_negative_current = DC_MAX_NEGATIVE_CURRENT
            board.config.max_regen_current = 0

            axis = board.axis0
            axis.motor.config.pole_pairs = POLE_PAIRS
            axis.motor.config.calibration_current = CALIBRATION_CURRENT
            axis.motor.config.resistance_calib_max_voltage = RESISTANCE_CALIB_MAX_VOLTAGE
            axis.motor.config.motor_type = MOTOR_TYPE_HIGH_CURRENT
            axis.motor.config.current_lim = CURRENT_LIM
            axis.motor.config.requested_current_range = REQUESTED_CURRENT_RANGE

            axis.encoder.config.mode = ENCODER_MODE_SPI_ABS_AMS
            axis.encoder.config.abs_spi_cs_gpio_pin = ENCODER_CS_GPIO_PIN
            axis.encoder.config.cpr = ENCODER_CPR
            axis.encoder.config.bandwidth = ENCODER_BANDWIDTH
            axis.encoder.config.calib_range = ENCODER_CALIB_RANGE

            axis.controller.config.control_mode = CONTROL_MODE_POSITION_CONTROL
            axis.controller.config.vel_limit = VEL_LIMIT
            axis.controller.config.pos_gain = self.pos_gain
            axis.controller.config.vel_gain = self.vel_gain
            axis.controller.config.vel_integrator_gain = self.vel_integrator_gain
            axis.controller.config.input_mode = INPUT_MODE_TRAP_TRAJ
            axis.trap_traj.config.vel_limit = self.trap_vel
            axis.trap_traj.config.accel_limit = self.trap_accel
            axis.trap_traj.config.decel_limit = self.trap_decel

            try:
                board.save_configuration()
            except Exception:
                pass  # save_configuration also reboots
            time.sleep(2.0)
            board = odrive.find_any(serial_number=f"{serial:012x}".upper(), timeout=15)

            axis = board.axis0
            axis.requested_state = AXIS_STATE_MOTOR_CALIBRATION
            t0 = time.monotonic()
            while axis.current_state != AXIS_STATE_IDLE and time.monotonic() - t0 < 30:
                time.sleep(0.2)
            if axis.error != 0:
                self.status_msg = f"Motor calibration failed: error={axis.error}"
                logging.error(self.status_msg)
                return False

            axis.requested_state = AXIS_STATE_ENCODER_OFFSET_CALIBRATION
            t0 = time.monotonic()
            while axis.current_state != AXIS_STATE_IDLE and time.monotonic() - t0 < 20:
                time.sleep(0.2)
            if axis.error != 0 or not axis.encoder.is_ready:
                self.status_msg = f"Encoder calibration failed: error={axis.error}"
                logging.error(self.status_msg)
                return False

            axis.motor.config.pre_calibrated = True
            try:
                board.save_configuration()
            except Exception:
                pass
            self.status_msg = f"Corrected config applied and calibrated on {serial:x}."
            logging.info(self.status_msg)
            return board
        except Exception:
            self.status_msg = f"apply_full_config error on {serial:x} (see log for traceback)"
            logging.exception(self.status_msg)
            return None

    def apply_full_config_both(self):
        if self.busy:
            return
        self.busy = True
        if self.left_odrive:
            result = self.apply_full_config(self.left_odrive)
            if result:
                self.left_odrive = result
        if self.right_odrive:
            result = self.apply_full_config(self.right_odrive)
            if result:
                self.right_odrive = result
        self.busy = False
        self.status_msg = "Full config sequence done -- see log for per-board results."

    def apply_live_gains(self, board):
        if not board:
            return
        try:
            c = board.axis0.controller.config
            c.pos_gain = float(self.pos_gain)
            c.vel_gain = float(self.vel_gain)
            c.vel_integrator_gain = float(self.vel_integrator_gain)
            board.axis0.trap_traj.config.vel_limit = float(self.trap_vel)
            board.axis0.trap_traj.config.accel_limit = float(self.trap_accel)
            board.axis0.trap_traj.config.decel_limit = float(self.trap_decel)
        except Exception as e:
            logging.error(f"apply_live_gains error: {e}")

    def update_gains(self, pos_gain, vel_gain, vel_integrator_gain, trap_vel, trap_accel, trap_decel):
        self.pos_gain = float(pos_gain)
        self.vel_gain = float(vel_gain)
        self.vel_integrator_gain = float(vel_integrator_gain)
        self.trap_vel = float(trap_vel)
        self.trap_accel = float(trap_accel)
        self.trap_decel = float(trap_decel)
        self.apply_live_gains(self.left_odrive)
        self.apply_live_gains(self.right_odrive)
        self.status_msg = f"Gains updated: pos_gain={self.pos_gain} vel_gain={self.vel_gain}"

    def enable_closed_loop(self):
        logging.info("Enabling position control closed loop...")
        self.clear_errors()
        for board in (self.left_odrive, self.right_odrive):
            if not board:
                continue
            try:
                if not board.axis0.encoder.is_ready:
                    board.axis0.requested_state = AXIS_STATE_ENCODER_OFFSET_CALIBRATION
                    t0 = time.monotonic()
                    while board.axis0.current_state != AXIS_STATE_IDLE and time.monotonic() - t0 < 20:
                        time.sleep(0.2)
                board.axis0.clear_errors()
                self.apply_live_gains(board)
                board.axis0.controller.input_pos = board.axis0.encoder.pos_estimate
                board.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
            except Exception as e:
                logging.error(f"enable_closed_loop error: {e}")
        self.left_enabled = self.left_odrive is not None
        self.right_enabled = self.right_odrive is not None
        if self.left_odrive:
            self.left_target_pos = self.left_odrive.axis0.encoder.pos_estimate
        if self.right_odrive:
            self.right_target_pos = self.right_odrive.axis0.encoder.pos_estimate
        self.status_msg = "Closed loop POSITION control enabled"

    def disable_idle(self):
        for board in (self.left_odrive, self.right_odrive):
            if board:
                try:
                    board.axis0.requested_state = AXIS_STATE_IDLE
                except Exception:
                    pass
        self.left_enabled = False
        self.right_enabled = False
        self.status_msg = "IDLE"

    def clear_errors(self):
        for board in (self.left_odrive, self.right_odrive):
            if board:
                try:
                    board.axis0.clear_errors()
                except Exception:
                    pass
        self.status_msg = "Errors cleared"

    def emergency_stop(self):
        logging.warning("EMERGENCY STOP")
        self.disable_idle()
        self.status_msg = "EMERGENCY STOPPED"

    def set_positions(self, left_pos, right_pos):
        new_left = max(-MAX_POSITION, min(MAX_POSITION, float(left_pos)))
        new_right = max(-MAX_POSITION, min(MAX_POSITION, float(right_pos)))
        if new_left != self.left_target_pos:
            self.left_target_pos = new_left
            self.dirty_left_target = True
        if new_right != self.right_target_pos:
            self.right_target_pos = new_right
            self.dirty_right_target = True

    def poll_and_control_step(self):
        if self.left_odrive and self.left_enabled and self.dirty_left_target:
            try:
                if self.left_odrive.axis0.error != 0:
                    self.left_odrive.axis0.clear_errors()
                    self.left_odrive.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
                self.left_odrive.axis0.controller.input_pos = self.left_target_pos
                self.dirty_left_target = False
            except Exception:
                self.left_odrive = None

        if self.right_odrive and self.right_enabled and self.dirty_right_target:
            try:
                if self.right_odrive.axis0.error != 0:
                    self.right_odrive.axis0.clear_errors()
                    self.right_odrive.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
                self.right_odrive.axis0.controller.input_pos = self.right_target_pos
                self.dirty_right_target = False
            except Exception:
                self.right_odrive = None

        data = {
            "timestamp": time.time(),
            "actual_freq": round(self.actual_freq, 1),
            "status_msg": self.status_msg,
            "busy": self.busy,
            "pos_gain": self.pos_gain,
            "vel_gain": self.vel_gain,
            "vel_integrator_gain": self.vel_integrator_gain,
            "trap_vel": self.trap_vel,
            "trap_accel": self.trap_accel,
            "trap_decel": self.trap_decel,
            "left": {"connected": False, "target_pos": round(self.left_target_pos, 3),
                     "measured_pos": 0.0, "measured_vel": 0.0, "current": 0.0,
                     "error": 0, "state": 0, "encoder_ready": False},
            "right": {"connected": False, "target_pos": round(self.right_target_pos, 3),
                      "measured_pos": 0.0, "measured_vel": 0.0, "current": 0.0,
                      "error": 0, "state": 0, "encoder_ready": False},
            "vbus": 0.0,
        }

        if self.left_odrive:
            try:
                a = self.left_odrive.axis0
                data["left"].update(connected=True,
                                     measured_pos=round(float(a.encoder.pos_estimate), 3),
                                     measured_vel=round(float(a.encoder.vel_estimate), 3),
                                     current=round(float(a.motor.current_control.Iq_measured), 2),
                                     error=int(a.error), state=int(a.current_state),
                                     encoder_ready=bool(a.encoder.is_ready))
                data["vbus"] = round(float(self.left_odrive.vbus_voltage), 2)
            except Exception:
                self.left_odrive = None

        if self.right_odrive:
            try:
                a = self.right_odrive.axis0
                data["right"].update(connected=True,
                                      measured_pos=round(float(a.encoder.pos_estimate), 3),
                                      measured_vel=round(float(a.encoder.vel_estimate), 3),
                                      current=round(float(a.motor.current_control.Iq_measured), 2),
                                      error=int(a.error), state=int(a.current_state),
                                      encoder_ready=bool(a.encoder.is_ready))
                if data["vbus"] == 0.0:
                    data["vbus"] = round(float(self.right_odrive.vbus_voltage), 2)
            except Exception:
                self.right_odrive = None

        return data


controller = ODriveController()
ws_clients = set()

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>JAVIS - Position Control Test</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
:root { --bg:#0f172a; --card:#1e293b; --blue:#38bdf8; --green:#4ade80; --orange:#fb923c; --red:#f87171; --purple:#c084fc; --text:#f8fafc; --muted:#94a3b8; }
* { box-sizing:border-box; margin:0; padding:0; font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif; }
body { background:var(--bg); color:var(--text); padding:20px; }
.header { display:flex; justify-content:space-between; align-items:center; padding-bottom:20px; border-bottom:1px solid #334155; margin-bottom:20px; }
.header h1 { color:var(--blue); font-size:1.6rem; }
.status-badge { background:#334155; padding:6px 14px; border-radius:20px; font-size:0.9rem; font-weight:bold; }
.status-online { background:#065f46; color:#34d399; }
.grid { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
@media (max-width:900px) { .grid { grid-template-columns:1fr; } }
.card { background:var(--card); border-radius:12px; padding:20px; border:1px solid #334155; }
.card h2 { font-size:1.1rem; color:var(--muted); margin-bottom:15px; border-bottom:1px solid #334155; padding-bottom:8px; }
.warn-box { background:#451a03; border:1px solid #f59e0b; color:#fbbf24; padding:12px; border-radius:8px; margin-bottom:15px; font-size:0.85rem; }
.btn-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:15px; }
button { background:#3b82f6; color:white; border:none; padding:12px; border-radius:8px; font-weight:bold; cursor:pointer; }
button:hover { opacity:0.9; }
button.btn-success { background:#10b981; }
button.btn-warning { background:#f59e0b; }
button.btn-purple { background:#8b5cf6; }
button.btn-config { background:#d97706; grid-column:span 2; border:2px solid #f59e0b; }
button.btn-danger { background:#ef4444; grid-column:span 2; font-size:1.1rem; padding:15px; }
.control-group { margin-bottom:20px; }
.control-label { display:flex; justify-content:space-between; margin-bottom:8px; font-weight:600; }
input[type=range] { width:100%; height:8px; border-radius:4px; background:#334155; }
.telemetry-row { display:flex; justify-content:space-between; padding:10px 0; border-bottom:1px solid #334155; }
.telemetry-val { font-family:monospace; font-weight:bold; }
.val-target { color:var(--blue); } .val-measured { color:var(--green); }
.chart-container { position:relative; height:320px; width:100%; }
.system-status { background:#1e1b4b; border:1px solid #6366f1; padding:12px; border-radius:8px; font-weight:600; margin-bottom:15px; color:#a5b4fc; }
.tuning-box { background:#0f172a; padding:15px; border-radius:8px; border:1px solid #334155; margin-bottom:15px; }
.tuning-box input { background:#1e293b; color:white; border:1px solid #475569; padding:6px 8px; border-radius:6px; width:70px; text-align:center; }
.tuning-box label { font-size:0.8rem; }
</style>
</head>
<body>
<div class="header">
  <div><h1>JAVIS Position Control Test</h1>
  <p style="color:var(--muted); font-size:0.85rem; margin-top:4px;">pole_pairs=15 (corrected), TRAP_TRAJ position mode</p></div>
  <div id="conn-status" class="status-badge">Connecting...</div>
</div>
<div id="status-banner" class="system-status">Status: Connecting...</div>
<div class="warn-box">Known issue this session: closed-loop control has consistently shown high current / near-zero motion on this board's 15-pole-pair motor across every offset/gain tried, even though open-loop lockin_spin proved the hardware itself is healthy. Position mode uses the same encoder-offset commutation, so expect the same symptom unless something about this control path behaves differently.</div>

<div class="grid">
  <div class="card">
    <h2>Setup &amp; Control</h2>
    <div class="btn-grid">
      <button class="btn-config" onclick="sendCommand('apply_full_config')">APPLY CORRECTED CONFIG (erase + pole_pairs=15 + calibrate)</button>
      <button class="btn-success" onclick="sendCommand('enable')">ENABLE POSITION CONTROL</button>
      <button class="btn-warning" onclick="sendCommand('idle')">SET IDLE</button>
      <button onclick="sendCommand('clear')">CLEAR ERRORS</button>
      <button onclick="reconnectODrive()">RECONNECT USB</button>
      <button class="btn-danger" onclick="sendCommand('estop')">EMERGENCY STOP</button>
    </div>

    <div class="tuning-box">
      <h3 style="font-size:0.9rem; color:var(--purple); margin-bottom:10px;">Live Gain / Trajectory Tuning</h3>
      <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
        <div><label>pos_gain: </label><input type="number" id="pg-input" step="1" value="20"></div>
        <div><label>vel_gain: </label><input type="number" id="vg-input" step="0.05" value="0.3"></div>
        <div><label>integrator: </label><input type="number" id="vgi-input" step="0.05" value="0.2"></div>
        <div><label>trap vel: </label><input type="number" id="tv-input" step="0.5" value="4"></div>
        <div><label>trap accel: </label><input type="number" id="ta-input" step="0.5" value="4"></div>
        <div><label>trap decel: </label><input type="number" id="td-input" step="0.5" value="4"></div>
        <button class="btn-purple" style="padding:6px 12px; font-size:0.85rem;" onclick="applyGains()">Apply</button>
      </div>
    </div>

    <div class="control-group">
      <div class="control-label"><span>Left Target Position: <span id="left-val-lbl" class="val-target">0.0</span> turns</span></div>
      <input type="range" id="left-slider" min="-8" max="8" step="0.1" value="0" oninput="updatePositions()">
    </div>
    <div class="control-group">
      <div class="control-label"><span>Right Target Position: <span id="right-val-lbl" class="val-target">0.0</span> turns</span></div>
      <input type="range" id="right-slider" min="-8" max="8" step="0.1" value="0" oninput="updatePositions()">
    </div>
    <div style="display:flex; gap:10px;">
      <button style="flex:1" onclick="presetPos(0,0)">Zero (0,0)</button>
      <button style="flex:1" onclick="presetPos(1,1)">+1 turn</button>
      <button style="flex:1" onclick="presetPos(-1,-1)">-1 turn</button>
      <button style="flex:1" onclick="presetPos(2,-2)">Spin apart</button>
    </div>
  </div>

  <div class="card">
    <h2>Telemetry</h2>
    <div class="telemetry-row"><span>Rate:</span><span id="loop-freq-val" class="telemetry-val" style="color:var(--purple)">0 Hz</span></div>
    <div class="telemetry-row"><span>Bus Voltage:</span><span id="vbus-val" class="telemetry-val" style="color:var(--orange)">0.0 V</span></div>

    <h3 style="margin-top:15px; margin-bottom:5px; color:var(--blue)">Left (SN 318236823335)</h3>
    <div class="telemetry-row"><span>Target / Measured Pos:</span><span class="telemetry-val"><span id="l-tgt" class="val-target">0.0</span> / <span id="l-meas" class="val-measured">0.0</span> turns</span></div>
    <div class="telemetry-row"><span>Vel / Current / Err:</span><span class="telemetry-val"><span id="l-vel">0.0</span> t/s | <span id="l-curr">0.0</span> A | <span id="l-err">0x0</span></span></div>
    <div class="telemetry-row"><span>Ready / State:</span><span class="telemetry-val"><span id="l-ready">?</span> | <span id="l-state">?</span></span></div>

    <h3 style="margin-top:15px; margin-bottom:5px; color:var(--green)">Right (SN 3676365D3335)</h3>
    <div class="telemetry-row"><span>Target / Measured Pos:</span><span class="telemetry-val"><span id="r-tgt" class="val-target">0.0</span> / <span id="r-meas" class="val-measured">0.0</span> turns</span></div>
    <div class="telemetry-row"><span>Vel / Current / Err:</span><span class="telemetry-val"><span id="r-vel">0.0</span> t/s | <span id="r-curr">0.0</span> A | <span id="r-err">0x0</span></span></div>
    <div class="telemetry-row"><span>Ready / State:</span><span class="telemetry-val"><span id="r-ready">?</span> | <span id="r-state">?</span></span></div>
  </div>
</div>

<div class="card" style="margin-top:20px;">
  <h2>Position Tracking</h2>
  <div class="chart-container"><canvas id="posChart"></canvas></div>
</div>

<script>
let ws;
const maxPts = 150;
const labels = [];
const ctx = document.getElementById('posChart').getContext('2d');
const posChart = new Chart(ctx, { type:'line', data:{ labels, datasets:[
  { label:'Left Target', borderColor:'#38bdf8', borderDash:[4,4], data:[], fill:false },
  { label:'Left Measured', borderColor:'#0284c7', borderWidth:2.5, data:[], fill:false },
  { label:'Right Target', borderColor:'#4ade80', borderDash:[4,4], data:[], fill:false },
  { label:'Right Measured', borderColor:'#16a34a', borderWidth:2.5, data:[], fill:false }
]}, options:{ responsive:true, maintainAspectRatio:false, animation:false,
  scales:{ x:{ display:true, grid:{color:'#334155'} }, y:{ display:true, title:{display:true,text:'turns',color:'#94a3b8'}, grid:{color:'#334155'} } },
  plugins:{ legend:{ labels:{ color:'#f8fafc' } } } } });

function connectWS() {
  const host = window.location.host;
  ws = new WebSocket(`ws://${host}/ws`);
  ws.onopen = () => { const b=document.getElementById('conn-status'); b.innerText='ONLINE'; b.classList.add('status-online'); };
  ws.onmessage = (e) => updateDashboard(JSON.parse(e.data));
  ws.onclose = () => { const b=document.getElementById('conn-status'); b.innerText='OFFLINE (retrying)'; b.classList.remove('status-online'); setTimeout(connectWS,2000); };
}

function updateDashboard(d) {
  document.getElementById('status-banner').innerText = 'Status: ' + d.status_msg;
  document.getElementById('vbus-val').innerText = d.vbus.toFixed(2) + ' V';
  document.getElementById('loop-freq-val').innerText = d.actual_freq.toFixed(1) + ' Hz';

  document.getElementById('l-tgt').innerText = d.left.target_pos.toFixed(2);
  document.getElementById('l-meas').innerText = d.left.measured_pos.toFixed(2);
  document.getElementById('l-vel').innerText = d.left.measured_vel.toFixed(2);
  document.getElementById('l-curr').innerText = d.left.current.toFixed(2);
  document.getElementById('l-err').innerText = '0x' + d.left.error.toString(16);
  document.getElementById('l-ready').innerText = d.left.encoder_ready ? 'Ready' : 'Not ready';
  document.getElementById('l-state').innerText = d.left.state;

  document.getElementById('r-tgt').innerText = d.right.target_pos.toFixed(2);
  document.getElementById('r-meas').innerText = d.right.measured_pos.toFixed(2);
  document.getElementById('r-vel').innerText = d.right.measured_vel.toFixed(2);
  document.getElementById('r-curr').innerText = d.right.current.toFixed(2);
  document.getElementById('r-err').innerText = '0x' + d.right.error.toString(16);
  document.getElementById('r-ready').innerText = d.right.encoder_ready ? 'Ready' : 'Not ready';
  document.getElementById('r-state').innerText = d.right.state;

  if (!document.activeElement.id.includes('pg')) document.getElementById('pg-input').value = d.pos_gain;
  if (!document.activeElement.id.includes('vg-')) document.getElementById('vg-input').value = d.vel_gain;
  if (!document.activeElement.id.includes('vgi')) document.getElementById('vgi-input').value = d.vel_integrator_gain;
  if (!document.activeElement.id.includes('tv')) document.getElementById('tv-input').value = d.trap_vel;
  if (!document.activeElement.id.includes('ta')) document.getElementById('ta-input').value = d.trap_accel;
  if (!document.activeElement.id.includes('td')) document.getElementById('td-input').value = d.trap_decel;

  const now = new Date().toLocaleTimeString().split(' ')[0] + '.' + Math.floor(new Date().getMilliseconds()/100);
  if (labels.length >= maxPts) { labels.shift(); posChart.data.datasets.forEach(ds => ds.data.shift()); }
  labels.push(now);
  posChart.data.datasets[0].data.push(d.left.target_pos);
  posChart.data.datasets[1].data.push(d.left.measured_pos);
  posChart.data.datasets[2].data.push(d.right.target_pos);
  posChart.data.datasets[3].data.push(d.right.measured_pos);
  posChart.update('none');
}

function updatePositions() {
  const l = parseFloat(document.getElementById('left-slider').value);
  const r = parseFloat(document.getElementById('right-slider').value);
  document.getElementById('left-val-lbl').innerText = l.toFixed(1);
  document.getElementById('right-val-lbl').innerText = r.toFixed(1);
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type:'set_pos', left:l, right:r }));
}

function applyGains() {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({
    type:'update_gains',
    pos_gain: parseFloat(document.getElementById('pg-input').value),
    vel_gain: parseFloat(document.getElementById('vg-input').value),
    vel_integrator_gain: parseFloat(document.getElementById('vgi-input').value),
    trap_vel: parseFloat(document.getElementById('tv-input').value),
    trap_accel: parseFloat(document.getElementById('ta-input').value),
    trap_decel: parseFloat(document.getElementById('td-input').value)
  }));
}

function presetPos(l, r) {
  document.getElementById('left-slider').value = l;
  document.getElementById('right-slider').value = r;
  updatePositions();
}

function sendCommand(cmd) {
  if (cmd === 'estop') presetPos(0,0);
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type:'cmd', action:cmd }));
}
function reconnectODrive() {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type:'cmd', action:'reconnect' }));
}
window.onload = connectWS;
</script>
</body>
</html>
"""


async def handle_index(request):
    return web.Response(text=HTML_PAGE, content_type="text/html")


async def handle_websocket(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    ws_clients.add(ws)
    logging.info(f"WS client connected. Total: {len(ws_clients)}")
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)
                t = data.get("type")
                if t == "set_pos":
                    controller.set_positions(data.get("left", 0.0), data.get("right", 0.0))
                elif t == "update_gains":
                    controller.update_gains(
                        data.get("pos_gain", POS_GAIN),
                        data.get("vel_gain", VEL_GAIN),
                        data.get("vel_integrator_gain", VEL_INTEGRATOR_GAIN),
                        data.get("trap_vel", TRAP_VEL_LIMIT),
                        data.get("trap_accel", TRAP_ACCEL_LIMIT),
                        data.get("trap_decel", TRAP_DECEL_LIMIT),
                    )
                elif t == "cmd":
                    action = data.get("action")
                    loop = asyncio.get_event_loop()
                    if action == "apply_full_config":
                        await loop.run_in_executor(None, controller.apply_full_config_both)
                    elif action == "enable":
                        await loop.run_in_executor(None, controller.enable_closed_loop)
                    elif action == "idle":
                        controller.disable_idle()
                    elif action == "clear":
                        controller.clear_errors()
                    elif action == "estop":
                        controller.emergency_stop()
                    elif action == "reconnect":
                        await loop.run_in_executor(None, controller.connect)
    except Exception as e:
        logging.error(f"WS error: {e}")
    finally:
        ws_clients.discard(ws)
        logging.info("WS client disconnected.")
    return ws


async def telemetry_loop():
    last_calc = time.time()
    n = 0
    while True:
        target_interval = 1.0 / controller.telemetry_freq
        t0 = time.time()
        telemetry = controller.poll_and_control_step()
        n += 1
        now = time.time()
        if now - last_calc >= 0.5:
            controller.actual_freq = n / (now - last_calc)
            n = 0
            last_calc = now
        if ws_clients:
            payload = json.dumps(telemetry)
            for ws in list(ws_clients):
                try:
                    await ws.send_str(payload)
                except Exception:
                    pass
        elapsed = time.time() - t0
        await asyncio.sleep(max(0.001, target_interval - elapsed))


async def start_server():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, controller.connect)
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/ws", handle_websocket)
    asyncio.create_task(telemetry_loop())
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8090)
    await site.start()
    logging.info("=" * 60)
    logging.info("JAVIS Position Control Test running on port 8090")
    logging.info("=" * 60)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_server())
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        controller.emergency_stop()
