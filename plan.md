Plan: Raspberry Pi AV Backend + Local Web UI (DRAFT)
This rewrite will use an incremental strangler approach so the vehicle stays operable while replacing the legacy TCP server. The new core will be FastAPI with REST + WebSocket for control/state, MJPEG for browser video, and a single-controller lock policy. v1 autonomy will include line following (camera/IR) and obstacle avoidance (ultrasonic), with explicit safety controls (heartbeat timeout and forced motor stop) before enabling remote motion commands. Existing hardware drivers are reused first, then wrapped behind clear interfaces so autonomy logic and API can evolve independently. Deferred features (clamp FSM automation, object detection/tracking) are preserved as backlog endpoints/state stubs so they stay visible but out of v1 critical path.

Steps

Define target module boundaries and create an ADR-style migration note in README.md and a server rewrite spec in main.py context, mapping legacy symbols (mywindow, threading_cmd_receive, threading_car_task) to new services.
Extract hardware-facing adapters (motor, servo, ultrasonic, infrared, camera, LED, controller) from car.py, camera.py, and led.py into a HAL package with idempotent startup/shutdown and safe defaults.
Introduce a central vehicle state store + control arbitration service (single controller lock, command lease, heartbeat watchdog, emergency stop) and wire hard stop on timeout directly to motor zeroing.
Replace raw command parsing in message.py and command.py with typed API schemas while keeping a temporary compatibility adapter for legacy CMD_* messages.
Build FastAPI app with REST endpoints for configuration/state and WebSocket channels for telemetry + command ingress; expose mode switching and actuator commands currently handled in main.py.
Replace raw TCP video transport from server.py and tcp_server.py with MJPEG HTTP streaming sourced from Camera.get_frame.
Implement AV behavior engine (FSM) for v1 states: manual, line_follow, obstacle_avoid, idle/safe_stop, reusing current routines from mode_infrared/mode_ultrasonic in car.py, with deterministic loop timing and max-speed guards.
Build local web client served from the backend: live MJPEG pane, telemetry/state panel, mode/config controls, controller lock visibility, and safety controls (E-stop + heartbeat status).
Add configuration consolidation by extending params.json for PID/threshold/speed/network/safety settings and load validation at boot.
Complete migration by deprecating legacy TCP startup path in main.py, retaining optional legacy bridge for a short transition window to avoid breaking Video.py immediately.
Verification

Unit checks for state transitions, command arbitration, and watchdog behavior.
Hardware-in-loop smoke tests on Pi: boot, camera stream, teleop, heartbeat loss stop, mode transitions, shutdown cleanup.
API/UI checks: GET /state, config update round-trip, WebSocket telemetry cadence, single-controller lock enforcement, MJPEG stream stability.
Regression pass for reused hardware adapters against current behavior baselines.
Decisions

API architecture: REST + WebSocket.
Video for v1: MJPEG over HTTP.
Security for v1: LAN-only without auth (documented as non-production risk).
Control policy: single controller lock.
v1 autonomy scope: line following + ultrasonic obstacle avoidance.
Deferred but tracked: clamp/object manipulation automation, object detection/tracking.