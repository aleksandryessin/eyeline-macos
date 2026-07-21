import Foundation
import SystemExtensions

@MainActor
final class CameraExtensionManager: NSObject, ObservableObject {
  enum Status: Equatable {
    case idle
    case activating
    case awaitingApproval
    case active
    case deactivating
    case failed

    var description: String {
      switch self {
      case .idle: "Not activated"
      case .activating: "Activating…"
      case .awaitingApproval: "Awaiting approval in System Settings"
      case .active: "Active"
      case .deactivating: "Deactivating…"
      case .failed: "Activation failed"
      }
    }
  }

  static let extensionIdentifier = "com.aleksandryessin.eyeline.camera-extension"

  @Published private(set) var status: Status = .idle
  @Published private(set) var detail: String?

  func activate() {
    status = .activating
    detail = nil
    let request = OSSystemExtensionRequest.activationRequest(
      forExtensionWithIdentifier: Self.extensionIdentifier,
      queue: .main
    )
    request.delegate = self
    OSSystemExtensionManager.shared.submitRequest(request)
  }

  func deactivate() {
    status = .deactivating
    detail = nil
    let request = OSSystemExtensionRequest.deactivationRequest(
      forExtensionWithIdentifier: Self.extensionIdentifier,
      queue: .main
    )
    request.delegate = self
    OSSystemExtensionManager.shared.submitRequest(request)
  }
}

extension CameraExtensionManager: OSSystemExtensionRequestDelegate {
  nonisolated func request(
    _ request: OSSystemExtensionRequest,
    actionForReplacingExtension existing: OSSystemExtensionProperties,
    withExtension ext: OSSystemExtensionProperties
  ) -> OSSystemExtensionRequest.ReplacementAction {
    ext.bundleVersion >= existing.bundleVersion ? .replace : .cancel
  }

  nonisolated func requestNeedsUserApproval(_ request: OSSystemExtensionRequest) {
    Task { @MainActor in
      status = .awaitingApproval
      detail = "Approve EyeLine Camera in System Settings → General → Login Items & Extensions."
    }
  }

  nonisolated func request(
    _ request: OSSystemExtensionRequest,
    didFinishWithResult result: OSSystemExtensionRequest.Result
  ) {
    Task { @MainActor in
      status = .active
      detail =
        result == .willCompleteAfterReboot
        ? "Activation will complete after a normal restart."
        : "Extension request completed."
    }
  }

  nonisolated func request(_ request: OSSystemExtensionRequest, didFailWithError error: Error) {
    Task { @MainActor in
      status = .failed
      detail = error.localizedDescription
    }
  }
}
