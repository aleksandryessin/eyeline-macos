# EyeLine Core ML converter

This Python 3.11 environment is intentionally isolated from the Python 3.12
OBS runtime. It pins the previously verified converter combination:

- `tensorflow-macos==2.12.0`
- `coremltools==9.0`

Create the environment and inspect the command without touching the app runtime:

```sh
uv sync --directory tools/converter
uv run --directory tools/converter eyeline-convert-coreml --help
```

Export the required compat.v1 checkpoint tensors to a TensorFlow SavedModel,
then pass the exact exported input name and shape:

```sh
uv run --directory tools/converter eyeline-convert-coreml \
  build/gaze_saved_model build/GazeFlow.mlpackage \
  --input-name eye_patch --input-shape 1,64,64,3
```

## Spatial-transformer fallback

Core ML conversion is not on the OBS-MVP critical path. If `coremltools` cannot
lower the upstream spatial-transformer/warp operations, export only the model
heads that predict the optical-flow and light-map tensors. Convert those heads
to an ML Program and apply the resulting flow/light map in the native target
with a Metal/Core Image warp. Do not silently replace an unsupported warp with
a visually different operation, and do not add converter dependencies to the
runtime environment.
