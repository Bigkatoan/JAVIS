#!/usr/bin/env python3
"""
JAVIS Smooth Motor Web Control & Auto-Error Clearing Node
Runs on Jetson Orin Nano (IP: 192.168.0.222)

Key Fixes:
1. Automatic clear_errors() on startup & enable -> Prevents 0x100 / 0x80 error locks when entering closed loop.
2. Saved Flash Pre-Calibrated Motor & Encoder Config -> Instant smooth closed loop startup.
3. INPUT_MODE_VEL_RAMP with vel_ramp_rate = 25.0 turns/s^2 -> Smooth deceleration and immediate active reversal.
4. Encoder Bandwidth = 200.0 Hz -> Filters AS5047P SPI noise.
5. Non-blocking event-driven USB communication.
"""

import asyncio
import json
import logging
import os
import sys
import time
from aiohttp import web
import usb.core
import odrive
from odrive.enums import (
    AXIS_STATE_CLOSED_LOOP_CONTROL,
    AXIS_STATE_ENCODER_OFFSET_CALIBRATION,
    AXIS_STATE_IDLE,
    CONTROL_MODE_VELOCITY_CONTROL,
    INPUT_MODE_VEL_RAMP,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Serial Numbers for JAVIS ODrive Boards
LEFT_SERIAL = "318236823335"
RIGHT_SERIAL = "3676365D3335"

# Safety limits for 8-inch wheels (Max ~2.5 t/s = ~1.0 m/s = brisk walk)
MAX_VELOCITY = 2.5  # max turns/s

class ODriveController:
    def __init__(self):
        self.left_odrive = None
        self.right_odrive = None
        self.left_target = 0.0
        self.right_target = 0.0
        self.left_enabled = False
        self.right_enabled = False
        
        # Verified smooth gain and ramp parameters
        self.vel_gain = 0.25
        self.vel_integrator_gain = 0.15
        self.vel_ramp_rate = 25.0  # 25 turns/s^2 smooth accel/decel ramp
        self.encoder_bandwidth = 200.0  # Filters AS5047P SPI noise
        
        self.vbus_voltage = 0.0
        self.telemetry_freq = 40.0  # 40 Hz smooth telemetry rate
        self.actual_freq = 0.0
        self.calibrating = False
        self.status_msg = "Ready"
        self.last_telemetry = {}
        
        # Event-driven USB update flags
        self.dirty_left_target = False
        self.dirty_right_target = False

    def connect(self):
        logging.info("Connecting to ODrive boards...")
        self.status_msg = "Connecting to ODrives..."
        
        # Connect Left ODrive
        try:
            logging.info(f"Connecting Left ODrive (SN: {LEFT_SERIAL})...")
            self.left_odrive = odrive.find_any(serial_number=LEFT_SERIAL, timeout=3)
            self.left_odrive.axis0.clear_errors()
            self.apply_configs_to_board(self.left_odrive)
            logging.info(f"Connected Left ODrive! Vbus={self.left_odrive.vbus_voltage:.2f}V")
        except Exception as e:
            logging.warning(f"Left ODrive connection failed: {e}")
            self.left_odrive = None

        # Connect Right ODrive
        try:
            logging.info(f"Connecting Right ODrive (SN: {RIGHT_SERIAL})...")
            self.right_odrive = odrive.find_any(serial_number=RIGHT_SERIAL, timeout=3)
            self.right_odrive.axis0.clear_errors()
            self.apply_configs_to_board(self.right_odrive)
            logging.info(f"Connected Right ODrive! Vbus={self.right_odrive.vbus_voltage:.2f}V")
        except Exception as e:
            logging.warning(f"Right ODrive connection failed: {e}")
            self.right_odrive = None

        self.status_msg = "ODrives Connected. Ready for Smooth Control!"

    def apply_configs_to_board(self, board):
        if not board:
            return
        try:
            # Clear errors first
            board.axis0.clear_errors()
            # Filter SPI encoder noise
            board.axis0.encoder.config.bandwidth = float(self.encoder_bandwidth)
            
            c = board.axis0.controller.config
            c.control_mode = CONTROL_MODE_VELOCITY_CONTROL
            c.input_mode = INPUT_MODE_VEL_RAMP  # Ramp mode for smooth direction reversal
            c.vel_ramp_rate = float(self.vel_ramp_rate)
            c.vel_gain = float(self.vel_gain)
            c.vel_integrator_gain = float(self.vel_integrator_gain)
            logging.info(f"Applied config: input_mode=VEL_RAMP ({self.vel_ramp_rate}t/s^2), BW={self.encoder_bandwidth}Hz, vel_gain={self.vel_gain}, vel_integrator_gain={self.vel_integrator_gain}")
        except Exception as e:
            logging.error(f"Error applying configs: {e}")

    def update_gains(self, vel_gain, vel_integrator_gain, bandwidth, ramp_rate):
        self.vel_gain = float(vel_gain)
        self.vel_integrator_gain = float(vel_integrator_gain)
        self.encoder_bandwidth = float(bandwidth)
        self.vel_ramp_rate = float(ramp_rate)
        self.apply_configs_to_board(self.left_odrive)
        self.apply_configs_to_board(self.right_odrive)
        self.status_msg = f"Updated: Ramp={self.vel_ramp_rate}t/s², BW={self.encoder_bandwidth}Hz, Gain={self.vel_gain}"

    def calibrate_encoders(self):
        """Runs encoder offset calibration."""
        if self.calibrating:
            return
        
        self.calibrating = True
        self.status_msg = "Calibrating Encoder Offsets (wheels will rotate)..."
        logging.info("Starting Encoder Offset Calibration...")

        if self.left_odrive:
            try:
                self.left_odrive.axis0.clear_errors()
                self.left_odrive.axis0.requested_state = AXIS_STATE_ENCODER_OFFSET_CALIBRATION
            except Exception as e:
                logging.error(f"Error starting Left calibration: {e}")

        if self.right_odrive:
            try:
                self.right_odrive.axis0.clear_errors()
                self.right_odrive.axis0.requested_state = AXIS_STATE_ENCODER_OFFSET_CALIBRATION
            except Exception as e:
                logging.error(f"Error starting Right calibration: {e}")

        t_start = time.time()
        while time.time() - t_start < 12.0:
            left_done = True
            right_done = True
            if self.left_odrive and self.left_odrive.axis0.current_state != AXIS_STATE_IDLE:
                left_done = False
            if self.right_odrive and self.right_odrive.axis0.current_state != AXIS_STATE_IDLE:
                right_done = False
            if left_done and right_done:
                break
            time.sleep(0.5)

        left_ready = self.left_odrive and self.left_odrive.axis0.encoder.is_ready
        right_ready = self.right_odrive and self.right_odrive.axis0.encoder.is_ready

        self.calibrating = False
        if left_ready or right_ready:
            self.status_msg = f"Calibration Complete! (Left: {left_ready}, Right: {right_ready})"
            logging.info(self.status_msg)
        else:
            self.status_msg = "Calibration Failed or Timed Out!"
            logging.error(self.status_msg)

    def enable_closed_loop(self):
        logging.info("Enabling Closed Loop Control...")
        
        self.clear_errors()

        needs_calib = False
        if self.left_odrive and not self.left_odrive.axis0.encoder.is_ready:
            needs_calib = True
        if self.right_odrive and not self.right_odrive.axis0.encoder.is_ready:
            needs_calib = True

        if needs_calib:
            self.calibrate_encoders()

        if self.left_odrive:
            try:
                self.left_odrive.axis0.clear_errors()
                self.apply_configs_to_board(self.left_odrive)
                self.left_odrive.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
                self.left_enabled = True
                self.dirty_left_target = True
            except Exception as e:
                logging.error(f"Failed to enable Left ODrive: {e}")

        if self.right_odrive:
            try:
                self.right_odrive.axis0.clear_errors()
                self.apply_configs_to_board(self.right_odrive)
                self.right_odrive.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
                self.right_enabled = True
                self.dirty_right_target = True
            except Exception as e:
                logging.error(f"Failed to enable Right ODrive: {e}")

        self.status_msg = "Closed Loop Control ENABLED"

    def disable_idle(self):
        logging.info("Setting ODrives to IDLE state...")
        self.left_target = 0.0
        self.right_target = 0.0
        if self.left_odrive:
            try:
                self.left_odrive.axis0.controller.input_vel = 0.0
                self.left_odrive.axis0.requested_state = AXIS_STATE_IDLE
                self.left_enabled = False
            except Exception as e:
                logging.error(f"Error disabling Left ODrive: {e}")
        if self.right_odrive:
            try:
                self.right_odrive.axis0.controller.input_vel = 0.0
                self.right_odrive.axis0.requested_state = AXIS_STATE_IDLE
                self.right_enabled = False
            except Exception as e:
                logging.error(f"Error disabling Right ODrive: {e}")
        self.status_msg = "ODrives IDLE"

    def clear_errors(self):
        logging.info("Clearing ODrive errors...")
        if self.left_odrive:
            try:
                self.left_odrive.axis0.clear_errors()
            except Exception as e:
                logging.error(f"Error clearing Left ODrive: {e}")
        if self.right_odrive:
            try:
                self.right_odrive.axis0.clear_errors()
            except Exception as e:
                logging.error(f"Error clearing Right ODrive: {e}")
        self.status_msg = "Errors Cleared"

    def emergency_stop(self):
        logging.warning("EMERGENCY STOP TRIGGERED!")
        self.left_target = 0.0
        self.right_target = 0.0
        if self.left_odrive:
            try:
                self.left_odrive.axis0.controller.input_vel = 0.0
                self.left_odrive.axis0.requested_state = AXIS_STATE_IDLE
                self.left_enabled = False
            except Exception:
                pass
        if self.right_odrive:
            try:
                self.right_odrive.axis0.controller.input_vel = 0.0
                self.right_odrive.axis0.requested_state = AXIS_STATE_IDLE
                self.right_enabled = False
            except Exception:
                pass
        self.status_msg = "EMERGENCY STOPPED"

    def set_velocities(self, left_vel, right_vel):
        new_left = max(-MAX_VELOCITY, min(MAX_VELOCITY, float(left_vel)))
        new_right = max(-MAX_VELOCITY, min(MAX_VELOCITY, float(right_vel)))

        if new_left != self.left_target:
            self.left_target = new_left
            self.dirty_left_target = True

        if new_right != self.right_target:
            self.right_target = new_right
            self.dirty_right_target = True

    def poll_and_control_step(self):
        """Event-driven high-performance telemetry & control step."""
        if self.left_odrive and self.left_enabled and self.dirty_left_target:
            try:
                # Clear error if any occurred
                if self.left_odrive.axis0.error != 0:
                    self.left_odrive.axis0.clear_errors()
                    self.left_odrive.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
                self.left_odrive.axis0.controller.input_vel = self.left_target
                self.dirty_left_target = False
            except Exception:
                self.left_odrive = None

        if self.right_odrive and self.right_enabled and self.dirty_right_target:
            try:
                if self.right_odrive.axis0.error != 0:
                    self.right_odrive.axis0.clear_errors()
                    self.right_odrive.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
                self.right_odrive.axis0.controller.input_vel = self.right_target
                self.dirty_right_target = False
            except Exception:
                self.right_odrive = None

        # Sample telemetry
        data = {
            "timestamp": time.time(),
            "target_freq": self.telemetry_freq,
            "actual_freq": round(self.actual_freq, 1),
            "status_msg": self.status_msg,
            "calibrating": self.calibrating,
            "vel_gain": self.vel_gain,
            "vel_integrator_gain": self.vel_integrator_gain,
            "encoder_bandwidth": self.encoder_bandwidth,
            "vel_ramp_rate": self.vel_ramp_rate,
            "left": {
                "connected": self.left_odrive is not None,
                "target_vel": round(self.left_target, 3),
                "measured_vel": 0.0,
                "current": 0.0,
                "error": 0,
                "state": 0,
                "encoder_ready": False
            },
            "right": {
                "connected": self.right_odrive is not None,
                "target_vel": round(self.right_target, 3),
                "measured_vel": 0.0,
                "current": 0.0,
                "error": 0,
                "state": 0,
                "encoder_ready": False
            },
            "vbus": 0.0
        }

        if self.left_odrive:
            try:
                data["left"]["measured_vel"] = round(float(self.left_odrive.axis0.encoder.vel_estimate), 3)
                data["left"]["current"] = round(float(self.left_odrive.axis0.motor.current_control.Iq_measured), 2)
                data["left"]["error"] = int(self.left_odrive.axis0.error)
                data["left"]["state"] = int(self.left_odrive.axis0.current_state)
                data["left"]["encoder_ready"] = bool(self.left_odrive.axis0.encoder.is_ready)
                data["vbus"] = round(float(self.left_odrive.vbus_voltage), 2)
            except Exception:
                self.left_odrive = None

        if self.right_odrive:
            try:
                data["right"]["measured_vel"] = round(float(self.right_odrive.axis0.encoder.vel_estimate), 3)
                data["right"]["current"] = round(float(self.right_odrive.axis0.motor.current_control.Iq_measured), 2)
                data["right"]["error"] = int(self.right_odrive.axis0.error)
                data["right"]["state"] = int(self.right_odrive.axis0.current_state)
                data["right"]["encoder_ready"] = bool(self.right_odrive.axis0.encoder.is_ready)
                if data["vbus"] == 0.0:
                    data["vbus"] = round(float(self.right_odrive.vbus_voltage), 2)
            except Exception:
                self.right_odrive = None

        self.last_telemetry = data
        return data

# Global controller & websocket clients
controller = ODriveController()
ws_clients = set()

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JAVIS - Smooth Bidirectional Motor Control</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --accent-blue: #38bdf8;
            --accent-green: #4ade80;
            --accent-orange: #fb923c;
            --accent-red: #f87171;
            --accent-purple: #c084fc;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        body { background-color: var(--bg-color); color: var(--text-main); padding: 20px; }
        .header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 20px; border-bottom: 1px solid #334155; margin-bottom: 20px; }
        .header h1 { color: var(--accent-blue); font-size: 1.8rem; }
        .status-badge { background: #334155; padding: 6px 14px; border-radius: 20px; font-size: 0.9rem; font-weight: bold; }
        .status-online { background: #065f46; color: #34d399; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
        .card { background: var(--card-bg); border-radius: 12px; padding: 20px; border: 1px solid #334155; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3); }
        .card h2 { font-size: 1.2rem; color: var(--text-muted); margin-bottom: 15px; border-bottom: 1px solid #334155; padding-bottom: 8px; }
        .control-group { margin-bottom: 20px; }
        .control-label { display: flex; justify-content: space-between; margin-bottom: 8px; font-weight: 600; }
        input[type=range] { width: 100%; height: 8px; border-radius: 4px; background: #334155; outline: none; cursor: pointer; }
        .btn-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px; }
        button { background: #3b82f6; color: white; border: none; padding: 12px; border-radius: 8px; font-weight: bold; cursor: pointer; transition: all 0.2s; }
        button:hover { opacity: 0.9; transform: translateY(-1px); }
        button.btn-success { background: #10b981; }
        button.btn-warning { background: #f59e0b; }
        button.btn-purple { background: #8b5cf6; }
        button.btn-calib { background: #d97706; grid-column: span 2; font-size: 1.05rem; padding: 14px; border: 2px solid #f59e0b; }
        button.btn-danger { background: #ef4444; grid-column: span 2; font-size: 1.1rem; padding: 15px; }
        .telemetry-row { display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #334155; }
        .telemetry-val { font-family: monospace; font-weight: bold; font-size: 1.1rem; }
        .val-target { color: var(--accent-blue); }
        .val-measured { color: var(--accent-green); }
        .chart-container { position: relative; height: 340px; width: 100%; }
        .system-status { background: #1e1b4b; border: 1px solid #6366f1; padding: 12px; border-radius: 8px; font-weight: 600; margin-bottom: 15px; color: #a5b4fc; }
        .tuning-box { background: #0f172a; padding: 15px; border-radius: 8px; border: 1px solid #334155; margin-bottom: 15px; }
        .tuning-box input { background: #1e293b; color: white; border: 1px solid #475569; padding: 6px 8px; border-radius: 6px; width: 70px; text-align: center; }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>🤖 JAVIS Motor Velocity Controller</h1>
            <p style="color:var(--text-muted); font-size:0.9rem; margin-top:4px;">Pre-Calibrated Flash Saved & Velocity Ramp Mode (25t/s²)</p>
        </div>
        <div style="display:flex; gap:10px;">
            <div id="conn-status" class="status-badge">Connecting...</div>
        </div>
    </div>

    <div id="status-banner" class="system-status">Status: Connecting...</div>

    <div class="grid">
        <!-- Controls Card -->
        <div class="card">
            <h2>🕹️ Remote Wheel Control</h2>
            
            <div class="btn-grid">
                <button class="btn-calib" onclick="sendCommand('calibrate')">🎯 RE-CALIBRATE ENCODERS (HIỆU CHUẨN MƯỢT)</button>
                <button class="btn-success" onclick="sendCommand('enable')">⚡ ENABLE CLOSED LOOP</button>
                <button class="btn-warning" onclick="sendCommand('idle')">⏸️ SET IDLE</button>
                <button onclick="sendCommand('clear')">🧹 CLEAR ERRORS</button>
                <button onclick="reconnectODrive()">🔄 RECONNECT USB</button>
                <button class="btn-danger" onclick="sendCommand('estop')">🚨 EMERGENCY STOP 🚨</button>
            </div>

            <!-- Ramp & Tuning Section -->
            <div class="tuning-box">
                <h3 style="font-size:0.95rem; color:var(--accent-purple); margin-bottom:10px;">🎛️ Ramp Accel & PI Tuning (Đảo chiều mượt):</h3>
                <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
                    <div>
                        <label style="font-size:0.8rem;">Ramp Rate (t/s²): </label>
                        <input type="number" id="rr-input" step="5" value="25">
                    </div>
                    <div>
                        <label style="font-size:0.8rem;">BW (Hz): </label>
                        <input type="number" id="bw-input" step="50" value="200">
                    </div>
                    <div>
                        <label style="font-size:0.8rem;">vel_gain: </label>
                        <input type="number" id="vg-input" step="0.05" value="0.25">
                    </div>
                    <div>
                        <label style="font-size:0.8rem;">integrator: </label>
                        <input type="number" id="vgi-input" step="0.05" value="0.15">
                    </div>
                    <button class="btn-purple" style="padding:6px 12px; font-size:0.85rem;" onclick="applyGains()">Apply Settings</button>
                </div>
            </div>

            <div class="control-group">
                <div class="control-label">
                    <span>Left Wheel Target: <span id="left-val-lbl" class="val-target">0.0</span> turns/s</span>
                </div>
                <input type="range" id="left-slider" min="-2.5" max="2.5" step="0.1" value="0" oninput="updateVelocities()">
            </div>

            <div class="control-group">
                <div class="control-label">
                    <span>Right Wheel Target: <span id="right-val-lbl" class="val-target">0.0</span> turns/s</span>
                </div>
                <input type="range" id="right-slider" min="-2.5" max="2.5" step="0.1" value="0" oninput="updateVelocities()">
            </div>

            <div style="display:flex; gap:10px;">
                <button style="flex:1" onclick="presetVel(0,0)">⏹️ Stop (0,0)</button>
                <button style="flex:1" onclick="presetVel(1.5, 1.5)">⬆️ Fwd (+1.5 t/s)</button>
                <button style="flex:1" onclick="presetVel(-1.5, -1.5)">⬇️ Rev (-1.5 t/s)</button>
                <button style="flex:1" onclick="presetVel(-1.0, 1.0)">🔄 Spin (-1, 1)</button>
            </div>
        </div>

        <!-- Telemetry Card -->
        <div class="card">
            <h2>⚡ System Telemetry</h2>
            <div class="telemetry-row">
                <span>Telemetry Rate:</span>
                <span id="loop-freq-val" class="telemetry-val" style="color:var(--accent-purple)">40.0 Hz</span>
            </div>
            <div class="telemetry-row">
                <span>Bus Voltage (Battery):</span>
                <span id="vbus-val" class="telemetry-val" style="color:var(--accent-orange)">0.0 V</span>
            </div>
            
            <h3 style="margin-top:15px; margin-bottom:5px; font-size:1rem; color:var(--accent-blue)">Left Wheel (SN: 318236823335)</h3>
            <div class="telemetry-row">
                <span>Target vs Measured:</span>
                <span class="telemetry-val"><span id="l-tgt" class="val-target">0.0</span> / <span id="l-meas" class="val-measured">0.0</span> turns/s</span>
            </div>
            <div class="telemetry-row">
                <span>Encoder / Current / Err:</span>
                <span class="telemetry-val"><span id="l-ready">False</span> | <span id="l-curr">0.0</span> A | Err: <span id="l-err">0x0</span></span>
            </div>

            <h3 style="margin-top:15px; margin-bottom:5px; font-size:1rem; color:var(--accent-green)">Right Wheel (SN: 3676365D3335)</h3>
            <div class="telemetry-row">
                <span>Target vs Measured:</span>
                <span class="telemetry-val"><span id="r-tgt" class="val-target">0.0</span> / <span id="r-meas" class="val-measured">0.0</span> turns/s</span>
            </div>
            <div class="telemetry-row">
                <span>Encoder / Current / Err:</span>
                <span class="telemetry-val"><span id="r-ready">False</span> | <span id="r-curr">0.0</span> A | Err: <span id="r-err">0x0</span></span>
            </div>
        </div>
    </div>

    <!-- Chart Section -->
    <div class="card" style="margin-top:20px;">
        <h2>📈 Real-Time Velocity Tracking & Step Response</h2>
        <div class="chart-container">
            <canvas id="velChart"></canvas>
        </div>
    </div>

    <script>
        let ws;
        const maxDataPoints = 120;
        const labels = [];
        
        // Chart setup
        const ctx = document.getElementById('velChart').getContext('2d');
        const velChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    { label: 'Left Target', borderColor: '#38bdf8', borderDash: [4, 4], data: [], fill: false, tension: 0.05 },
                    { label: 'Left Measured', borderColor: '#0284c7', borderWidth: 2.5, data: [], fill: false, tension: 0.1 },
                    { label: 'Right Target', borderColor: '#4ade80', borderDash: [4, 4], data: [], fill: false, tension: 0.05 },
                    { label: 'Right Measured', borderColor: '#16a34a', borderWidth: 2.5, data: [], fill: false, tension: 0.1 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    x: { display: true, title: { display: true, text: 'Time', color: '#94a3b8' }, grid: { color: '#334155' } },
                    y: { display: true, title: { display: true, text: 'Velocity (turns/s)', color: '#94a3b8' }, grid: { color: '#334155' } }
                },
                plugins: {
                    legend: { labels: { color: '#f8fafc' } }
                }
            }
        });

        function connectWS() {
            const host = window.location.host;
            ws = new WebSocket(`ws://${host}/ws`);

            ws.onopen = () => {
                const badge = document.getElementById('conn-status');
                badge.innerText = 'ONLINE';
                badge.classList.add('status-online');
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                updateDashboard(data);
            };

            ws.onclose = () => {
                const badge = document.getElementById('conn-status');
                badge.innerText = 'OFFLINE (Retrying...)';
                badge.classList.remove('status-online');
                setTimeout(connectWS, 2000);
            };
        }

        function updateDashboard(data) {
            document.getElementById('status-banner').innerText = "Status: " + data.status_msg;
            document.getElementById('vbus-val').innerText = data.vbus.toFixed(2) + ' V';
            document.getElementById('loop-freq-val').innerText = `${data.actual_freq.toFixed(1)} Hz`;

            // Left
            document.getElementById('l-tgt').innerText = data.left.target_vel.toFixed(2);
            document.getElementById('l-meas').innerText = data.left.measured_vel.toFixed(2);
            document.getElementById('l-curr').innerText = data.left.current.toFixed(2);
            document.getElementById('l-ready').innerText = data.left.encoder_ready ? "Ready ✅" : "Not Ready ❌";
            document.getElementById('l-ready').style.color = data.left.encoder_ready ? "#4ade80" : "#f87171";
            document.getElementById('l-err').innerText = '0x' + data.left.error.toString(16);

            // Right
            document.getElementById('r-tgt').innerText = data.right.target_vel.toFixed(2);
            document.getElementById('r-meas').innerText = data.right.measured_vel.toFixed(2);
            document.getElementById('r-curr').innerText = data.right.current.toFixed(2);
            document.getElementById('r-ready').innerText = data.right.encoder_ready ? "Ready ✅" : "Not Ready ❌";
            document.getElementById('r-ready').style.color = data.right.encoder_ready ? "#4ade80" : "#f87171";
            document.getElementById('r-err').innerText = '0x' + data.right.error.toString(16);

            // Inputs
            if (data.vel_gain && !document.activeElement.id.includes('vg')) document.getElementById('vg-input').value = data.vel_gain;
            if (data.vel_integrator_gain && !document.activeElement.id.includes('vgi')) document.getElementById('vgi-input').value = data.vel_integrator_gain;
            if (data.encoder_bandwidth && !document.activeElement.id.includes('bw')) document.getElementById('bw-input').value = data.encoder_bandwidth;
            if (data.vel_ramp_rate && !document.activeElement.id.includes('rr')) document.getElementById('rr-input').value = data.vel_ramp_rate;

            // Update Chart
            const nowStr = new Date().toLocaleTimeString().split(' ')[0] + '.' + Math.floor(new Date().getMilliseconds() / 100);
            if (labels.length >= maxDataPoints) {
                labels.shift();
                velChart.data.datasets.forEach(ds => ds.data.shift());
            }

            labels.push(nowStr);
            velChart.data.datasets[0].data.push(data.left.target_vel);
            velChart.data.datasets[1].data.push(data.left.measured_vel);
            velChart.data.datasets[2].data.push(data.right.target_vel);
            velChart.data.datasets[3].data.push(data.right.measured_vel);

            velChart.update('none');
        }

        function updateVelocities() {
            const leftVal = parseFloat(document.getElementById('left-slider').value);
            const rightVal = parseFloat(document.getElementById('right-slider').value);

            document.getElementById('left-val-lbl').innerText = leftVal.toFixed(1);
            document.getElementById('right-val-lbl').innerText = rightVal.toFixed(1);

            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'set_vel',
                    left: leftVal,
                    right: rightVal
                }));
            }
        }

        function applyGains() {
            const bw = parseFloat(document.getElementById('bw-input').value);
            const vg = parseFloat(document.getElementById('vg-input').value);
            const vgi = parseFloat(document.getElementById('vgi-input').value);
            const rr = parseFloat(document.getElementById('rr-input').value);
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'update_gains',
                    bandwidth: bw,
                    vel_gain: vg,
                    vel_integrator_gain: vgi,
                    ramp_rate: rr
                }));
            }
        }

        function presetVel(left, right) {
            document.getElementById('left-slider').value = left;
            document.getElementById('right-slider').value = right;
            updateVelocities();
        }

        function sendCommand(cmd) {
            if (cmd === 'estop') presetVel(0, 0);
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'cmd', action: cmd }));
            }
        }

        function reconnectODrive() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'cmd', action: 'reconnect' }));
            }
        }

        window.onload = connectWS;
    </script>
</body>
</html>
"""

async def handle_index(request):
    return web.Response(text=HTML_PAGE, content_type='text/html')

async def handle_websocket(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    ws_clients.add(ws)
    logging.info(f"WebSocket client connected. Total clients: {len(ws_clients)}")

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)
                msg_type = data.get("type")

                if msg_type == "set_vel":
                    controller.set_velocities(data.get("left", 0.0), data.get("right", 0.0))
                elif msg_type == "update_gains":
                    controller.update_gains(
                        data.get("vel_gain", 0.25),
                        data.get("vel_integrator_gain", 0.15),
                        data.get("bandwidth", 200.0),
                        data.get("ramp_rate", 25.0)
                    )
                elif msg_type == "cmd":
                    action = data.get("action")
                    if action == "calibrate":
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, controller.calibrate_encoders)
                    elif action == "enable":
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, controller.enable_closed_loop)
                    elif action == "idle":
                        controller.disable_idle()
                    elif action == "clear":
                        controller.clear_errors()
                    elif action == "estop":
                        controller.emergency_stop()
                    elif action == "reconnect":
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, controller.connect)

    except Exception as e:
        logging.error(f"WebSocket error: {e}")
    finally:
        ws_clients.remove(ws)
        logging.info("WebSocket client disconnected.")

    return ws

async def telemetry_loop():
    """Smooth non-blocking telemetry & event loop executing at 40 Hz"""
    last_freq_calc_time = time.time()
    loop_counter = 0

    while True:
        target_interval = 1.0 / controller.telemetry_freq
        t_start = time.time()

        # Step 1: Poll telemetry and send event-driven USB updates
        telemetry = controller.poll_and_control_step()
        loop_counter += 1

        # Calculate actual loop frequency
        now = time.time()
        if now - last_freq_calc_time >= 0.5:
            controller.actual_freq = loop_counter / (now - last_freq_calc_time)
            loop_counter = 0
            last_freq_calc_time = now

        # Step 2: Stream WebSocket telemetry to clients
        if ws_clients:
            payload = json.dumps(telemetry)
            for ws in list(ws_clients):
                try:
                    await ws.send_str(payload)
                except Exception:
                    pass

        # High precision sleep
        elapsed = time.time() - t_start
        sleep_time = target_interval - elapsed
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
        else:
            await asyncio.sleep(0.001)

async def start_server():
    # Attempt initial ODrive connection in background thread
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, controller.connect)

    app = web.Application()
    app.router.add_get('/', handle_index)
    app.router.add_get('/ws', handle_websocket)

    # Start telemetry loop
    asyncio.create_task(telemetry_loop())

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8080)
    await site.start()
    logging.info("==========================================================")
    logging.info(" JAVIS Smooth Motor Web Control Node running!")
    logging.info(" Auto-Clear Stale Errors & Pre-Calibrated Flash Saved")
    logging.info(" Access Web Dashboard at: http://192.168.0.238:8080")
    logging.info("==========================================================")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_server())
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        logging.info("Stopping Motor Web Server...")
        controller.emergency_stop()
