import CoreMediaIO
import Foundation
import IOKit.audio

final class EyeLineDeviceSource: NSObject, CMIOExtensionDeviceSource {
  private(set) var device: CMIOExtensionDevice!
  private let streamSource: EyeLineStreamSource

  init(localizedName: String) {
    streamSource = EyeLineStreamSource(localizedName: "EyeLine Camera 720p")
    super.init()

    device = CMIOExtensionDevice(
      localizedName: localizedName,
      deviceID: UUID(uuidString: "9E099C1A-7FD1-4CB6-A2F9-A2786F8E13CF")!,
      legacyDeviceID: "com.aleksandryessin.eyeline.camera",
      source: self
    )
    streamSource.attach(to: device)
    try? device.addStream(streamSource.stream)
  }

  var availableProperties: Set<CMIOExtensionProperty> {
    [.deviceModel, .deviceTransportType]
  }

  func deviceProperties(
    forProperties properties: Set<CMIOExtensionProperty>
  ) throws -> CMIOExtensionDeviceProperties {
    let values = CMIOExtensionDeviceProperties(dictionary: [:])
    if properties.contains(.deviceModel) {
      values.model = "EyeLine Native Camera 0.1"
    }
    if properties.contains(.deviceTransportType) {
      values.transportType = Int(kIOAudioDeviceTransportTypeVirtual)
    }
    return values
  }

  func setDeviceProperties(_ deviceProperties: CMIOExtensionDeviceProperties) throws {
    // Device metadata is intentionally read-only.
  }
}
