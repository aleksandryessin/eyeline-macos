import SwiftUI

@main
struct EyeLineApp: App {
  @StateObject private var extensionManager = CameraExtensionManager()

  var body: some Scene {
    WindowGroup {
      ContentView()
        .environmentObject(extensionManager)
        .frame(minWidth: 560, minHeight: 360)
    }
    .windowResizability(.contentSize)
  }
}
