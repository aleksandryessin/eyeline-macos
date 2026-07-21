import CoreMediaIO
import Foundation
import os

@main
final class CameraExtensionProviderSource: NSObject, CMIOExtensionProviderSource {
  private let logger = Logger(
    subsystem: "com.aleksandryessin.eyeline.camera-extension", category: "provider")
  private let deviceSource: EyeLineDeviceSource
  private(set) var provider: CMIOExtensionProvider!

  override init() {
    deviceSource = EyeLineDeviceSource(localizedName: "EyeLine Camera")
    super.init()
    provider = CMIOExtensionProvider(source: self, clientQueue: nil)
    do {
      try provider.addDevice(deviceSource.device)
    } catch {
      logger.fault(
        "Could not publish EyeLine Camera: \(error.localizedDescription, privacy: .public)")
    }
  }

  static func main() {
    let source = CameraExtensionProviderSource()
    CMIOExtensionProvider.startService(provider: source.provider)
    dispatchMain()
  }

  var availableProperties: Set<CMIOExtensionProperty> {
    [.providerName, .providerManufacturer]
  }

  func connect(to client: CMIOExtensionClient) throws {
    logger.info("Client connected: \(client.description, privacy: .public)")
  }

  func disconnect(from client: CMIOExtensionClient) {
    logger.info("Client disconnected: \(client.description, privacy: .public)")
  }

  func providerProperties(
    forProperties properties: Set<CMIOExtensionProperty>
  ) throws -> CMIOExtensionProviderProperties {
    let values = CMIOExtensionProviderProperties(dictionary: [:])
    if properties.contains(.providerName) {
      values.name = "EyeLine Camera Provider"
    }
    if properties.contains(.providerManufacturer) {
      values.manufacturer = "EyeLine"
    }
    return values
  }

  func setProviderProperties(_ providerProperties: CMIOExtensionProviderProperties) throws {
    // Provider metadata is intentionally read-only.
  }
}
