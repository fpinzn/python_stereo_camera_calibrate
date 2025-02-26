import Foundation
import AVFoundation
import Vision
import CoreImage

// Fixed explicit list of landmark names (in camelCase) in the explicit order.
let fixedLandmarkList = [
    "tailTop", "rightEarTop", "leftBackElbow", "leftEarBottom", "rightBackElbow",
    "tailBottom", "leftEarTop", "rightEye", "rightFrontKnee", "rightBackKnee",
    "rightBackPaw", "leftEye", "leftFrontPaw", "rightEarBottom", "tailMiddle",
    "nose", "neck", "leftBackKnee", "rightFrontPaw", "leftEarMiddle", "rightEarMiddle",
    "leftFrontElbow", "leftFrontKnee", "leftBackPaw", "rightFrontElbow"
]

// Helper: convert a camelCase string to snake_case.
func camelCaseToSnakeCase(_ input: String) -> String {
    let pattern = "([a-z0-9])([A-Z])"
    let regex = try! NSRegularExpression(pattern: pattern, options: [])
    let range = NSRange(location: 0, length: input.utf16.count)
    let snake = regex.stringByReplacingMatches(in: input, options: [], range: range, withTemplate: "$1_$2")
    return snake.lowercased()
}

// Helper: compute an approximate bounding box from a VNRecognizedPointsObservation.
func computeBoundingBox(from observation: VNRecognizedPointsObservation) -> [String: String] {
    var minX = 1.0, minY = 1.0, maxX = 0.0, maxY = 0.0
    var found = false
    for key in observation.availableKeys {
        if let point = try? observation.recognizedPoint(forKey: key) {
            found = true
            if point.x < minX { minX = point.x }
            if point.y < minY { minY = point.y }
            if point.x > maxX { maxX = point.x }
            if point.y > maxY { maxY = point.y }
        }
    }
    return [
        "x0": found ? "\(minX)" : "",
        "y0": found ? "\(minY)" : "",
        "x1": found ? "\(maxX)" : "",
        "y1": found ? "\(maxY)" : ""
    ]
}

// MARK: - Header Mode
// If the first argument is "header", process one frame from a sample video and output a JSON header.
if CommandLine.arguments.count >= 3 && CommandLine.arguments[1] == "header" {
    // In header mode we ignore detection and simply output our fixed header.
    let header: [String: Any] = [
        "fixed": ["absolute_time", "relative_time",
                  "bounding_box_js_x0", "bounding_box_js_y0", "bounding_box_js_x1", "bounding_box_js_y1"],
        "landmark_keys": fixedLandmarkList
    ]
    if let jsonData = try? JSONSerialization.data(withJSONObject: header, options: []),
       let jsonString = String(data: jsonData, encoding: .utf8) {
        print(jsonString)
    } else {
        print("{}")
    }
    exit(0)
}

// MARK: - Data Mode
// In data mode, process every frame and output one JSON object per frame (NDJSON).
func processFrame(_ ciImage: CIImage, frameNumber: Int) -> String {
    let request = VNDetectAnimalBodyPoseRequest()
    let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
    try? handler.perform([request])
    
    var frameDict: [String: Any] = [:]
    frameDict["frame"] = frameNumber
    frameDict["absolute_time"] = "\(Date().timeIntervalSince1970)"
    frameDict["relative_time"] = ""
    
    // Get bounding box from the first observation.
    let bbox = ((request.results as? [VNRecognizedPointsObservation])?.first).map { computeBoundingBox(from: $0) } ?? [:]
    frameDict["bounding_box"] = bbox
    
    var landmarks: [String: [String: String]] = [:]
    if let obs = (request.results as? [VNRecognizedPointsObservation])?.first {
        // Get all available keys, sorted by rawValue.
        let keys = obs.availableKeys.sorted { $0.rawValue < $1.rawValue }
        for key in keys {
            // Use "neck" if the raw value is "heck".
            let landmarkName = key.rawValue == "heck" ? "neck" : key.rawValue
            if let point = try? obs.recognizedPoint(forKey: key) {
                landmarks[landmarkName] = [
                    "x": "\(point.x)",
                    "y": "\(point.y)",
                    "c": "\(point.confidence)"
                ]
            } else {
                landmarks[landmarkName] = ["x": "", "y": "", "c": ""]
            }
        }
    }
    frameDict["landmarks"] = landmarks
    
    if let jsonData = try? JSONSerialization.data(withJSONObject: frameDict, options: []),
       let jsonString = String(data: jsonData, encoding: .utf8) {
        return jsonString
    }
    return "{}"
}

// Data mode: Process video file.
guard CommandLine.arguments.count >= 2 else {
    print("Usage: pose-estimation-video <video_path>")
    exit(1)
}
let videoPath = CommandLine.arguments[1]
let asset = AVAsset(url: URL(fileURLWithPath: videoPath))
guard let track = asset.tracks(withMediaType: .video).first else { exit(0) }
let reader: AVAssetReader
do {
    reader = try AVAssetReader(asset: asset)
} catch {
    exit(1)
}
let outputSettings: [String: Any] = [ kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA) ]
let readerOutput = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
reader.add(readerOutput)
reader.startReading()
var frameIndex = 1
while let sampleBuffer = readerOutput.copyNextSampleBuffer() {
    if let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        print(processFrame(ciImage, frameNumber: frameIndex))
    }
    frameIndex += 1
}
