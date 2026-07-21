# EyeLine native camera (secondary path)

This directory fully describes the SwiftUI host and CoreMediaIO Camera
Extension. The generated `EyeLine.xcodeproj` is disposable; regenerate it from
`project.yml` with XcodeGen.

```sh
native/scripts/doctor.sh
native/scripts/build-unsigned.sh
```

The unsigned build validates source and target wiring without an Apple account.
It cannot activate a system extension. Real activation later requires an Apple
Development identity and Team with the Camera Extension entitlement, installing
the host in `/Applications`, granting Camera access, and approving the Media
Extension in System Settings. No reduced-security or SIP changes are required.

The native pipeline is deliberately fail-open:

1. `AVCaptureDevice` supplies 1280×720 BGRA frames at a requested 30 FPS.
2. Vision finds face/eye landmarks locally.
3. A Core Image warp kernel runs through a Metal-backed `CIContext` when Metal
   is available.
4. Any Vision or processing error returns the original sample buffer.
5. `CMIOExtensionStream` publishes the result as **EyeLine Camera**.

The OBS path is the deadline-critical MVP. This native path is ready for signing
and activation work without blocking OBS.
