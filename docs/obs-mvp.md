# EyeLine OBS Virtual Camera MVP

EyeLine captures direct AVFoundation frames at 1280×720/30 FPS for Zoom (with an optional 1920×1080 profile), processes them locally, converts the
result to packed RGB, and writes directly to the OBS camera Media Extension through
`pyvirtualcam`. Video frames are neither recorded nor transmitted over the network.

## One-time OBS initialization

1. Install the current Apple Silicon build of OBS Studio in `/Applications/OBS.app`.
2. Open OBS and approve its Camera Media Extension when macOS asks.
3. In OBS choose **Start Virtual Camera**, then **Stop Virtual Camera**.
4. Quit OBS completely. OBS must stay closed while EyeLine owns the virtual camera.
5. Run `./scripts/verify-obs-backend.command`. This probe briefly opens and releases the output.

Do not disable SIP or enable Reduced Security. If macOS has not approved the extension, complete
the normal prompt in System Settings > Privacy & Security.

## Run

For Zoom, double-click `run-eyeline.command`, or from Terminal:

```sh
./run-eyeline.command
```

Use `./run-eyeline-1080p.command` only when the calling application is explicitly configured
for Full HD; otherwise switching devices can renegotiate its input resolution.

The launcher pins Xcode's developer directory, uses Python 3.12 through `uv`, and places model,
Matplotlib, XDG, and uv caches under `~/Library/Application Support/EyeLine`. On the first camera
open, it downloads the pinned Face Landmarker and gaze checkpoints and verifies their SHA-256
checksums. Complete, existing assets are reused. A download or checksum failure stops before the
camera is opened. Allow the invoking terminal application to use the Camera when macOS asks.
Press Control-C to stop; EyeLine releases both the physical and virtual cameras in cleanup handlers.
At startup, verify that Terminal reports `Opening exact physical built-in camera: Камера MacBook
Pro`, followed by `EyeLine video ready` and `EyeLine correction ready`. The standard profiles open that device directly through AVFoundation and do not enumerate
or automatically select an iPhone Continuity Camera.

For a transport-only check which bypasses gaze correction:

```sh
uv run --python 3.12 eyeline run --passthrough
```

## Diagnostics and soak

The default doctor is read-only and does not claim either camera:

```sh
uv run --python 3.12 eyeline doctor
uv run --python 3.12 eyeline doctor --probe-camera --probe-output
```

The opt-in probes are always skipped when `CI` is set. A synthetic 30-minute processor soak that
does not require a camera or OBS is:

```sh
uv run --python 3.12 eyeline soak --duration 1800
```

Add `--obs` to publish the moving fixture to OBS Virtual Camera. Add `--passthrough` to isolate
transport from the correction engine.

When no face is found, landmarks disappear, or correction raises an exception, the pipeline
publishes the unmodified camera frame. An invalid processed frame is handled the same way, so a
model problem cannot intentionally produce a black frame.
