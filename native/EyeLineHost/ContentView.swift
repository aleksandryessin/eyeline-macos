import AVFoundation
import SwiftUI

struct ContentView: View {
  @EnvironmentObject private var extensionManager: CameraExtensionManager
  @State private var cameraAuthorization = AVCaptureDevice.authorizationStatus(for: .video)

  var body: some View {
    VStack(alignment: .leading, spacing: 20) {
      Text("EyeLine Camera")
        .font(.largeTitle.bold())

      Text("Native camera extension preview")
        .font(.headline)

      Text(
        "The extension captures and processes frames entirely on this Mac. The OBS virtual camera remains the primary MVP output until native signing and activation are configured."
      )
      .foregroundStyle(.secondary)

      statusRow("Camera permission", value: cameraAuthorizationLabel)
      statusRow("Camera extension", value: extensionManager.status.description)

      if let detail = extensionManager.detail {
        Text(detail)
          .font(.callout)
          .foregroundStyle(.secondary)
          .textSelection(.enabled)
      }

      HStack {
        Button("Request Camera Access") {
          AVCaptureDevice.requestAccess(for: .video) { granted in
            Task { @MainActor in
              cameraAuthorization = granted ? .authorized : .denied
            }
          }
        }
        .disabled(cameraAuthorization == .authorized)

        Button("Activate Extension") {
          extensionManager.activate()
        }

        Button("Deactivate Extension") {
          extensionManager.deactivate()
        }
      }

      Spacer()
    }
    .padding(28)
  }

  private var cameraAuthorizationLabel: String {
    switch cameraAuthorization {
    case .authorized: "Authorized"
    case .denied: "Denied"
    case .restricted: "Restricted"
    case .notDetermined: "Not requested"
    @unknown default: "Unknown"
    }
  }

  private func statusRow(_ title: String, value: String) -> some View {
    HStack {
      Text(title)
      Spacer()
      Text(value)
        .foregroundStyle(.secondary)
    }
  }
}
