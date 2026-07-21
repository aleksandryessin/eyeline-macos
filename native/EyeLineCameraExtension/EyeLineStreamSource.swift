import CoreMedia
import CoreMediaIO
import CoreVideo
import Foundation
import os

final class EyeLineStreamSource: NSObject, CMIOExtensionStreamSource {
  static let width: Int32 = 1280
  static let height: Int32 = 720
  static let frameDuration = CMTime(value: 1, timescale: 30)

  private let logger = Logger(
    subsystem: "com.aleksandryessin.eyeline.camera-extension", category: "stream")
  private let capturePipeline = CameraCapturePipeline()
  private(set) var device: CMIOExtensionDevice?
  private(set) var stream: CMIOExtensionStream!
  let formats: [CMIOExtensionStreamFormat]

  private var activeFormatIndex = 0
  private var clientCount = 0

  init(localizedName: String) {
    var description: CMFormatDescription?
    let status = CMVideoFormatDescriptionCreate(
      allocator: kCFAllocatorDefault,
      codecType: kCVPixelFormatType_32BGRA,
      width: Self.width,
      height: Self.height,
      extensions: nil,
      formatDescriptionOut: &description
    )
    precondition(status == noErr && description != nil, "Unable to create EyeLine video format")

    formats = [
      CMIOExtensionStreamFormat(
        formatDescription: description!,
        maxFrameDuration: Self.frameDuration,
        minFrameDuration: Self.frameDuration,
        validFrameDurations: [Self.frameDuration]
      )
    ]
    super.init()

    stream = CMIOExtensionStream(
      localizedName: localizedName,
      streamID: UUID(uuidString: "D79D0BF6-A6B9-46EA-BC21-0B455816E22E")!,
      direction: .source,
      clockType: .hostTime,
      source: self
    )

    capturePipeline.onFrame = { [weak self] sampleBuffer in
      guard let self else { return }
      self.stream.send(
        sampleBuffer,
        discontinuity: [],
        hostTimeInNanoseconds: DispatchTime.now().uptimeNanoseconds
      )
    }
  }

  func attach(to device: CMIOExtensionDevice) {
    self.device = device
  }

  var availableProperties: Set<CMIOExtensionProperty> {
    [.streamActiveFormatIndex, .streamFrameDuration]
  }

  func streamProperties(
    forProperties properties: Set<CMIOExtensionProperty>
  ) throws -> CMIOExtensionStreamProperties {
    let values = CMIOExtensionStreamProperties(dictionary: [:])
    if properties.contains(.streamActiveFormatIndex) {
      values.activeFormatIndex = activeFormatIndex
    }
    if properties.contains(.streamFrameDuration) {
      values.frameDuration = Self.frameDuration
    }
    return values
  }

  func setStreamProperties(_ streamProperties: CMIOExtensionStreamProperties) throws {
    if let requestedIndex = streamProperties.activeFormatIndex {
      guard formats.indices.contains(requestedIndex) else {
        throw EyeLineCameraError.unsupportedFormat(requestedIndex)
      }
      activeFormatIndex = requestedIndex
    }
    if let duration = streamProperties.frameDuration, duration != Self.frameDuration {
      throw EyeLineCameraError.unsupportedFrameDuration(duration)
    }
  }

  func authorizedToStartStream(for client: CMIOExtensionClient) -> Bool { true }

  func startStream() throws {
    clientCount += 1
    guard clientCount == 1 else { return }
    do {
      try capturePipeline.start(width: Int(Self.width), height: Int(Self.height), fps: 30)
    } catch {
      clientCount = 0
      logger.error("Capture failed to start: \(error.localizedDescription, privacy: .public)")
      throw error
    }
  }

  func stopStream() throws {
    clientCount = max(0, clientCount - 1)
    if clientCount == 0 {
      capturePipeline.stop()
    }
  }
}

enum EyeLineCameraError: LocalizedError {
  case cameraUnavailable
  case cameraPermissionRequired
  case cannotAddInput
  case cannotAddOutput
  case unsupportedFormat(Int)
  case unsupportedFrameDuration(CMTime)

  var errorDescription: String? {
    switch self {
    case .cameraUnavailable: "No physical camera is available."
    case .cameraPermissionRequired: "Camera permission has not been granted to EyeLine."
    case .cannotAddInput: "The physical camera input could not be added."
    case .cannotAddOutput: "The camera frame output could not be added."
    case .unsupportedFormat(let index): "Unsupported stream format index \(index)."
    case .unsupportedFrameDuration: "Only 30 FPS is supported by this MVP stream."
    }
  }
}
