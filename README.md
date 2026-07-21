# EyeLine for macOS

EyeLine is a local-only gaze-correction camera pipeline for macOS. The deadline-critical MVP reads the built-in camera, processes frames in Python, and publishes RGB frames to the OBS Virtual Camera media extension through `pyvirtualcam`.

The application never records or uploads video and contains no telemetry. If face detection or correction fails, the original camera frame is forwarded.

## Development branches

- `main`: stable releases
- `integration`: feature integration
- `feature/core-engine`: landmarks, correction, and temporal naturalizer
- `feature/obs-mvp`: camera/output loop, launcher, and diagnostics
- `feature/native-camera`: SwiftUI host and CoreMediaIO camera extension

Runtime setup and user instructions will be added with the OBS MVP.
