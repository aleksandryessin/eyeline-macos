import CoreImage
import CoreMedia
import CoreVideo
import Foundation
import Metal
import Vision

/// A local, fail-open native processing path. Vision finds eye regions and a
/// Core Image warp kernel shifts source coordinates toward the optical axis.
/// CIContext is backed by Metal when a GPU is available; CPU rendering remains
/// a safe fallback.
final class VisionEyeWarpProcessor {
  private static let metalKernelSource = """
    #include <CoreImage/CoreImage.h>

    extern "C" {
      namespace coreimage {
        float2 eyeLineShift(destination dest, float2 center, float radius, float2 offset) {
          float2 delta = dest.coord() - center;
          float distance = length(delta);
          float amount = 1.0 - smoothstep(radius * 0.45, radius, distance);
          return dest.coord() + offset * amount;
        }
      }
    }
    """

  private let request = VNDetectFaceLandmarksRequest()
  private let context: CIContext
  private let warpKernel: CIWarpKernel?
  private let colorSpace = CGColorSpaceCreateDeviceRGB()
  private var outputPool: CVPixelBufferPool?
  private var poolSize: (width: Int, height: Int)?

  init() {
    if let device = MTLCreateSystemDefaultDevice() {
      context = CIContext(mtlDevice: device, options: [.cacheIntermediates: false])
    } else {
      context = CIContext(options: [.cacheIntermediates: false])
    }
    warpKernel =
      try? CIKernel.kernels(withMetalString: Self.metalKernelSource).first as? CIWarpKernel
  }

  func process(sampleBuffer: CMSampleBuffer) -> CMSampleBuffer? {
    guard
      let sourceBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
      let outputBuffer = process(pixelBuffer: sourceBuffer)
    else {
      return nil
    }

    var formatDescription: CMVideoFormatDescription?
    guard
      CMVideoFormatDescriptionCreateForImageBuffer(
        allocator: kCFAllocatorDefault,
        imageBuffer: outputBuffer,
        formatDescriptionOut: &formatDescription
      ) == noErr, let formatDescription
    else {
      return nil
    }

    var timing = CMSampleTimingInfo(
      duration: CMSampleBufferGetDuration(sampleBuffer),
      presentationTimeStamp: CMSampleBufferGetPresentationTimeStamp(sampleBuffer),
      decodeTimeStamp: CMSampleBufferGetDecodeTimeStamp(sampleBuffer)
    )
    var result: CMSampleBuffer?
    guard
      CMSampleBufferCreateReadyWithImageBuffer(
        allocator: kCFAllocatorDefault,
        imageBuffer: outputBuffer,
        formatDescription: formatDescription,
        sampleTiming: &timing,
        sampleBufferOut: &result
      ) == noErr
    else {
      return nil
    }
    return result
  }

  private func process(pixelBuffer: CVPixelBuffer) -> CVPixelBuffer? {
    let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up)
    do {
      try handler.perform([request])
    } catch {
      return nil
    }

    guard
      let face = request.results?.max(by: { $0.confidence < $1.confidence }),
      face.confidence >= 0.45,
      let landmarks = face.landmarks,
      let leftEye = center(of: landmarks.leftEye, in: face.boundingBox),
      let rightEye = center(of: landmarks.rightEye, in: face.boundingBox)
    else {
      return nil
    }

    let width = CVPixelBufferGetWidth(pixelBuffer)
    let height = CVPixelBufferGetHeight(pixelBuffer)
    let sourceImage = CIImage(cvPixelBuffer: pixelBuffer)
    let eyeRadius = max(18.0, face.boundingBox.width * CGFloat(width) * 0.115)
    let correction = min(6.0, face.boundingBox.width * CGFloat(width) * 0.012)

    var image = sourceImage
    for eye in [leftEye, rightEye] {
      let center = CIVector(
        x: eye.x * CGFloat(width),
        y: eye.y * CGFloat(height)
      )
      if let warpKernel {
        guard
          let warped = warpKernel.apply(
            extent: sourceImage.extent,
            roiCallback: { _, rect in rect.insetBy(dx: -eyeRadius, dy: -eyeRadius) },
            image: image,
            arguments: [center, eyeRadius, CIVector(x: 0, y: correction)]
          )
        else {
          return nil
        }
        image = warped
      } else {
        // Some restricted/test processes cannot compile an in-memory Core Image
        // Metal library. Keep the native path functional with a built-in,
        // Metal-accelerated geometric warp rather than dropping a frame.
        guard let filter = CIFilter(name: "CIBumpDistortion") else { return nil }
        filter.setValue(image, forKey: kCIInputImageKey)
        filter.setValue(center, forKey: kCIInputCenterKey)
        filter.setValue(eyeRadius, forKey: kCIInputRadiusKey)
        filter.setValue(0.06, forKey: kCIInputScaleKey)
        guard let warped = filter.outputImage else { return nil }
        image = warped
      }
    }

    guard let output = makeOutputBuffer(width: width, height: height) else { return nil }
    context.render(image, to: output, bounds: sourceImage.extent, colorSpace: colorSpace)
    return output
  }

  private func center(
    of region: VNFaceLandmarkRegion2D?,
    in boundingBox: CGRect
  ) -> CGPoint? {
    guard let region, region.pointCount > 0 else { return nil }
    let sum = region.normalizedPoints.reduce(CGPoint.zero) { partial, point in
      CGPoint(x: partial.x + point.x, y: partial.y + point.y)
    }
    let average = CGPoint(
      x: sum.x / CGFloat(region.pointCount),
      y: sum.y / CGFloat(region.pointCount)
    )
    return CGPoint(
      x: boundingBox.minX + average.x * boundingBox.width,
      y: boundingBox.minY + average.y * boundingBox.height
    )
  }

  private func makeOutputBuffer(width: Int, height: Int) -> CVPixelBuffer? {
    if poolSize?.width != width || poolSize?.height != height {
      let attributes: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: width,
        kCVPixelBufferHeightKey as String: height,
        kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        kCVPixelBufferMetalCompatibilityKey as String: true,
      ]
      var pool: CVPixelBufferPool?
      guard
        CVPixelBufferPoolCreate(
          kCFAllocatorDefault,
          nil,
          attributes as CFDictionary,
          &pool
        ) == kCVReturnSuccess
      else {
        return nil
      }
      outputPool = pool
      poolSize = (width, height)
    }

    guard let outputPool else { return nil }
    var output: CVPixelBuffer?
    guard
      CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, outputPool, &output)
        == kCVReturnSuccess
    else {
      return nil
    }
    return output
  }
}
