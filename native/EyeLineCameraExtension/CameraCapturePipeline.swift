import AVFoundation
import CoreMedia
import Foundation
import os

final class CameraCapturePipeline: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
  private let logger = Logger(
    subsystem: "com.aleksandryessin.eyeline.camera-extension", category: "capture")
  private let session = AVCaptureSession()
  private let sessionQueue = DispatchQueue(label: "com.aleksandryessin.eyeline.capture.session")
  private let frameQueue = DispatchQueue(label: "com.aleksandryessin.eyeline.capture.frames")
  private let processor = VisionEyeWarpProcessor()
  private var configured = false

  var onFrame: ((CMSampleBuffer) -> Void)?

  func start(width: Int, height: Int, fps: Int32) throws {
    guard AVCaptureDevice.authorizationStatus(for: .video) == .authorized else {
      throw EyeLineCameraError.cameraPermissionRequired
    }

    if !configured {
      try configure(width: width, height: height, fps: fps)
    }

    sessionQueue.async { [session] in
      if !session.isRunning {
        session.startRunning()
      }
    }
  }

  func stop() {
    sessionQueue.async { [session] in
      if session.isRunning {
        session.stopRunning()
      }
    }
  }

  private func configure(width: Int, height: Int, fps: Int32) throws {
    let discovery = AVCaptureDevice.DiscoverySession(
      deviceTypes: [.builtInWideAngleCamera, .continuityCamera, .external],
      mediaType: .video,
      position: .unspecified
    )
    guard let camera = discovery.devices.first else {
      throw EyeLineCameraError.cameraUnavailable
    }

    let input = try AVCaptureDeviceInput(device: camera)
    let output = AVCaptureVideoDataOutput()
    output.alwaysDiscardsLateVideoFrames = true
    output.videoSettings = [
      kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
    ]
    output.setSampleBufferDelegate(self, queue: frameQueue)

    session.beginConfiguration()
    defer { session.commitConfiguration() }
    session.sessionPreset = .hd1280x720

    guard session.canAddInput(input) else { throw EyeLineCameraError.cannotAddInput }
    session.addInput(input)
    guard session.canAddOutput(output) else { throw EyeLineCameraError.cannotAddOutput }
    session.addOutput(output)

    if let connection = output.connection(with: .video), connection.isVideoMirroringSupported {
      connection.isVideoMirrored = true
    }

    do {
      try camera.lockForConfiguration()
      defer { camera.unlockForConfiguration() }
      let duration = CMTime(value: 1, timescale: fps)
      camera.activeVideoMinFrameDuration = duration
      camera.activeVideoMaxFrameDuration = duration
    } catch {
      logger.warning(
        "Could not lock 30 FPS; using device default: \(error.localizedDescription, privacy: .public)"
      )
    }

    configured = true
  }

  func captureOutput(
    _ output: AVCaptureOutput,
    didOutput sampleBuffer: CMSampleBuffer,
    from connection: AVCaptureConnection
  ) {
    // Fail-open is deliberate: if Vision or the warp fails, the source image
    // buffer is wrapped and sent instead of dropping or blacking the frame.
    let processed = processor.process(sampleBuffer: sampleBuffer) ?? sampleBuffer
    onFrame?(processed)
  }
}
