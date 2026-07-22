# EyeLine for macOS

EyeLine is a local-only gaze-correction camera pipeline for macOS. The deadline-critical MVP reads the built-in camera, processes frames in Python, and publishes RGB frames to the OBS Virtual Camera media extension through `pyvirtualcam`.

The application never records or uploads video and contains no telemetry. If face detection or correction fails, the original camera frame is forwarded.
On macOS, EyeLine opens the physical built-in wide-angle `AVCaptureDevice` directly by immutable
unique ID. The standard launcher does not use OpenCV's dynamic numeric camera indexes and never
falls back automatically to an iPhone Continuity Camera. Startup must produce a real frame before
EyeLine claims OBS Virtual Camera.

## OBS MVP quick start

Requirements: Apple Silicon macOS, OBS Studio 32.1.2 or newer, `uv`, and Python 3.12.

1. Open OBS once, choose **Start Virtual Camera**, approve the Camera Extension in System Settings, choose **Stop Virtual Camera**, and quit OBS.
2. Double-click `run-eyeline.command` for Zoom's normal 1280×720 format. This avoids a
   resolution renegotiation when switching between the MacBook and OBS cameras. Use
   `run-eyeline-1080p.command` only when Full HD is explicitly enabled in the calling app.
3. Allow Camera access when macOS asks. Wait for `EyeLine correction ready`, then choose
   **OBS Virtual Camera** in the calling application.
4. Press Control-C to stop and release both cameras.

OBS must remain closed while EyeLine runs. The launcher creates writable caches under `~/Library/Application Support/EyeLine`, downloads the pinned Face Landmarker and gaze checkpoints when needed, verifies both SHA-256 values, and then starts the selected profile at 30 FPS.

Diagnostics do not claim hardware unless probes are explicitly requested:

```sh
uv run --python 3.12 eyeline doctor
uv run --python 3.12 eyeline doctor --probe-camera --probe-output
```

See [docs/obs-mvp.md](docs/obs-mvp.md) for troubleshooting and soak commands. `tools/browser-camera-test.html` is a local WebRTC page for checking that a browser can select OBS Virtual Camera.

## Development

```sh
uv sync --locked --extra dev
uv run ruff check src tests tools/download_models.py
uv run pytest -q
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer native/scripts/build-unsigned.sh
```

The native SwiftUI/CoreMediaIO branch builds unsigned in CI and does not block the OBS MVP. Real Camera Extension activation additionally requires an Apple Development Team and normal macOS system-extension approval; no reduced-security configuration is used.

## Development branches

- `main`: stable releases
- `integration`: feature integration
- `feature/core-engine`: landmarks, correction, and temporal naturalizer
- `feature/obs-mvp`: camera/output loop, launcher, and diagnostics
- `feature/native-camera`: SwiftUI host and CoreMediaIO camera extension
