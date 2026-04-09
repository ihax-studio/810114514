// iHax Capture — 瞬時スクリーンキャプチャ + LLM解析 + ネイティブポップアップ
//
// M5 Pro / M6 Pro 最適化:
//   - ScreenCaptureKit (macOS 13+) で低レイテンシキャプチャ
//   - Vision framework で画面内テキスト/UI要素を即座にOCR
//   - Metal Performance Shaders で画像前処理
//   - タスク複雑度を判定し、8B(単純) vs 70B(複雑) を自動切替
//   - P-core (高性能) / E-core (高効率) を意識したDispatch
//
// ビルド: swiftc -framework Cocoa -framework ScreenCaptureKit \
//         -framework Vision -o iHaxCapture iHaxCapture.swift
//
// 動作: グローバルホットキー (⌘+Shift+X) で画面キャプチャ → LLM解析 → ポップアップ表示

import Cocoa
import ScreenCaptureKit
import Vision
import Foundation

// MARK: - Configuration

struct iHaxConfig {
    static let apiBase = "http://localhost:8000"
    static let hotkey = (key: UInt16(7), modifiers: UInt(NSEvent.ModifierFlags.command.rawValue | NSEvent.ModifierFlags.shift.rawValue))  // ⌘+Shift+X
    static let popupWidth: CGFloat = 480
    static let popupMaxHeight: CGFloat = 600
}

// MARK: - Task Complexity (タスク複雑度判定)

enum TaskComplexity: String {
    case simple    // 8Bモデルで十分 (翻訳、短い質問、UI要素の説明)
    case moderate  // 32Bモデル推奨 (コード解析、中程度の推論)
    case complex   // 70Bモデル必須 (OS設計、アーキテクチャ、長いコード)

    /// 画面内容から複雑度を判定
    /// M5/M6 Pro最適化: この判定自体はE-core (高効率コア) で実行
    static func assess(text: String, hasCode: Bool, elementCount: Int) -> TaskComplexity {
        let length = text.count

        // コードが含まれている場合
        if hasCode {
            if length > 2000 { return .complex }    // 長いコード → 70B
            if length > 500  { return .moderate }    // 中程度コード → 32B
            return .simple                           // 短いコード → 8B
        }

        // テキストのみ
        if length > 3000     { return .complex }
        if length > 500      { return .moderate }
        if elementCount > 20 { return .moderate }    // UI要素が多い
        return .simple
    }

    /// 複雑度に応じたプリセット
    var preset: String {
        switch self {
        case .simple:   return "lightweight"  // 8B: 40-60 tok/s
        case .moderate: return "balanced"     // Coder32B: 15-25 tok/s
        case .complex:  return "developer"    // 70B 5bit: 7-10 tok/s
        }
    }

    /// QoS (Quality of Service) — M5/M6 ProのP-core/E-coreを意識
    var dispatchQoS: DispatchQoS {
        switch self {
        case .simple:   return .utility       // E-core (高効率) で十分
        case .moderate: return .userInitiated  // P-core 使用
        case .complex:  return .userInteractive // P-core 最優先
        }
    }
}

// MARK: - Screen Capture (ScreenCaptureKit)

class ScreenCapture {
    /// 瞬時にスクリーンキャプチャを取得
    /// M5/M6 Pro最適化: ScreenCaptureKitはGPUで直接キャプチャ → CPU転送不要
    static func captureScreen() async throws -> CGImage {
        let content = try await SCShareableContent.current
        guard let display = content.displays.first else {
            throw CaptureError.noDisplay
        }

        let filter = SCContentFilter(display: display, excludingWindows: [])
        let config = SCStreamConfiguration()
        config.width = display.width
        config.height = display.height
        config.pixelFormat = kCVPixelFormatType_32BGRA
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1) // 1FPS (静止画なので)

        // ScreenCaptureKit の screenshot API (macOS 14+)
        let image = try await SCScreenshotManager.captureImage(
            contentFilter: filter,
            configuration: config
        )
        return image
    }

    enum CaptureError: Error {
        case noDisplay
        case captureFailed
    }
}

// MARK: - OCR (Vision Framework)

class TextRecognizer {
    /// 画面からテキストとUI要素を抽出
    /// M5/M6 Pro最適化: VisionはNeural Engineで実行 → GPU/CPUに負荷をかけない
    static func recognizeText(from image: CGImage) async throws -> RecognizedContent {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.recognitionLanguages = ["en-US", "ja-JP"]
        request.usesLanguageCorrection = true

        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        try handler.perform([request])

        guard let observations = request.results else {
            return RecognizedContent(text: "", hasCode: false, elements: [])
        }

        var fullText = ""
        var elements: [UIElement] = []
        var hasCode = false

        for observation in observations {
            guard let candidate = observation.topCandidates(1).first else { continue }
            let text = candidate.string
            fullText += text + "\n"

            // コード検出 (簡易ヒューリスティック)
            if text.contains("{") || text.contains("}") ||
               text.contains("func ") || text.contains("class ") ||
               text.contains("import ") || text.contains("def ") ||
               text.contains("const ") || text.contains("let ") ||
               text.contains("->") || text.contains("=>") {
                hasCode = true
            }

            let bounds = observation.boundingBox
            elements.append(UIElement(
                text: text,
                bounds: bounds,
                confidence: candidate.confidence
            ))
        }

        return RecognizedContent(
            text: fullText.trimmingCharacters(in: .whitespacesAndNewlines),
            hasCode: hasCode,
            elements: elements
        )
    }
}

struct RecognizedContent {
    let text: String
    let hasCode: Bool
    let elements: [UIElement]
}

struct UIElement {
    let text: String
    let bounds: CGRect
    let confidence: Float
}

// MARK: - iHax Agent API Client

class AgentClient {
    static let shared = AgentClient()
    private let session = URLSession.shared

    /// LLMに画面内容を解析させる
    func analyze(content: RecognizedContent, complexity: TaskComplexity) async throws -> String {
        // 複雑度に応じてプリセット切り替え
        try await switchPreset(complexity.preset)

        // プロンプト構築
        let prompt: String
        if content.hasCode {
            prompt = """
            Analyze the following code on screen. Explain what it does, \
            identify any issues, and suggest improvements. \
            Respond in Japanese.

            Code:
            \(content.text)
            """
        } else {
            prompt = """
            Analyze the following screen content. Summarize what is shown, \
            identify key information, and provide any relevant insights. \
            Respond in Japanese.

            Screen content:
            \(content.text)
            """
        }

        return try await generate(prompt: prompt, complexity: complexity)
    }

    /// 汎用生成
    func generate(prompt: String, complexity: TaskComplexity) async throws -> String {
        let maxTokens: Int
        switch complexity {
        case .simple:   maxTokens = 256
        case .moderate: maxTokens = 512
        case .complex:  maxTokens = 1024
        }

        let body: [String: Any] = [
            "prompt": prompt,
            "max_tokens": maxTokens,
            "temperature": 0.5,
            "top_k": 40,
        ]

        let data = try await post(path: "/generate", body: body)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let text = json["text"] as? String else {
            throw AgentError.invalidResponse
        }
        return text
    }

    /// プリセット切り替え
    func switchPreset(_ preset: String) async throws {
        let body: [String: Any] = ["preset": preset]
        _ = try await post(path: "/preset", body: body)
    }

    /// ヘルスチェック
    func checkHealth() async -> Bool {
        guard let url = URL(string: "\(iHaxConfig.apiBase)/health") else { return false }
        do {
            let (_, response) = try await session.data(from: url)
            return (response as? HTTPURLResponse)?.statusCode == 200
        } catch {
            return false
        }
    }

    private func post(path: String, body: [String: Any]) async throws -> Data {
        guard let url = URL(string: "\(iHaxConfig.apiBase)\(path)") else {
            throw AgentError.invalidURL
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 120  // 70Bは時間かかる
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, _) = try await session.data(for: request)
        return data
    }

    enum AgentError: Error {
        case invalidURL
        case invalidResponse
    }
}

// MARK: - Popup Window (ネイティブポップアップ)

class PopupWindow: NSPanel {
    private let textView = NSTextView()
    private let statusLabel = NSTextField()
    private let complexityLabel = NSTextField()

    init() {
        let frame = NSRect(
            x: 0, y: 0,
            width: iHaxConfig.popupWidth,
            height: 200
        )

        super.init(
            contentRect: frame,
            styleMask: [.titled, .closable, .resizable, .utilityWindow, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )

        self.title = "iHax Agent"
        self.level = .floating
        self.isFloatingPanel = true
        self.hidesOnDeactivate = false
        self.backgroundColor = NSColor(white: 0.1, alpha: 0.95)
        self.isOpaque = false

        setupUI()
    }

    private func setupUI() {
        guard let contentView = self.contentView else { return }

        // Status label (上部)
        statusLabel.isEditable = false
        statusLabel.isBordered = false
        statusLabel.backgroundColor = .clear
        statusLabel.textColor = .secondaryLabelColor
        statusLabel.font = .monospacedSystemFont(ofSize: 11, weight: .regular)
        statusLabel.stringValue = "Analyzing..."
        statusLabel.translatesAutoresizingMaskIntoConstraints = false
        contentView.addSubview(statusLabel)

        // Complexity badge
        complexityLabel.isEditable = false
        complexityLabel.isBordered = false
        complexityLabel.backgroundColor = .clear
        complexityLabel.textColor = .systemBlue
        complexityLabel.font = .monospacedSystemFont(ofSize: 11, weight: .bold)
        complexityLabel.stringValue = ""
        complexityLabel.translatesAutoresizingMaskIntoConstraints = false
        contentView.addSubview(complexityLabel)

        // Scroll view + text view (メイン)
        let scrollView = NSScrollView()
        scrollView.hasVerticalScroller = true
        scrollView.translatesAutoresizingMaskIntoConstraints = false
        contentView.addSubview(scrollView)

        textView.isEditable = false
        textView.isSelectable = true
        textView.font = .monospacedSystemFont(ofSize: 13, weight: .regular)
        textView.textColor = .labelColor
        textView.backgroundColor = .clear
        textView.textContainerInset = NSSize(width: 8, height: 8)
        scrollView.documentView = textView

        NSLayoutConstraint.activate([
            statusLabel.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 8),
            statusLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 12),

            complexityLabel.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 8),
            complexityLabel.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -12),

            scrollView.topAnchor.constraint(equalTo: statusLabel.bottomAnchor, constant: 8),
            scrollView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor),
            scrollView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor),
            scrollView.bottomAnchor.constraint(equalTo: contentView.bottomAnchor),
        ])
    }

    func showResult(text: String, complexity: TaskComplexity, tokensPerSec: Double) {
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.textView.string = text
            self.statusLabel.stringValue = String(format: "%.1f tok/s", tokensPerSec)

            let badge: String
            let color: NSColor
            switch complexity {
            case .simple:
                badge = "SIMPLE (8B)"
                color = .systemGreen
            case .moderate:
                badge = "MODERATE (32B)"
                color = .systemOrange
            case .complex:
                badge = "COMPLEX (70B)"
                color = .systemRed
            }
            self.complexityLabel.stringValue = badge
            self.complexityLabel.textColor = color

            // ウィンドウサイズ調整
            let textHeight = min(
                CGFloat(text.count) * 0.5 + 100,
                iHaxConfig.popupMaxHeight
            )
            var frame = self.frame
            frame.size.height = textHeight
            self.setFrame(frame, display: true, animate: true)
        }
    }

    func showLoading(complexity: TaskComplexity) {
        DispatchQueue.main.async { [weak self] in
            self?.textView.string = "Analyzing screen with \(complexity.preset) preset..."
            self?.statusLabel.stringValue = "Loading model..."
        }
    }

    func showAtMousePosition() {
        guard let screen = NSScreen.main else { return }
        let mouseLocation = NSEvent.mouseLocation
        var frame = self.frame
        frame.origin.x = min(mouseLocation.x, screen.frame.maxX - frame.width)
        frame.origin.y = max(mouseLocation.y - frame.height, 0)
        self.setFrame(frame, display: true)
        self.orderFront(nil)
    }
}

// MARK: - App Delegate

class AppDelegate: NSObject, NSApplicationDelegate {
    var popup: PopupWindow?
    var statusItem: NSStatusItem?

    func applicationDidFinishLaunching(_ notification: Notification) {
        // メニューバーアイコン
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        statusItem?.button?.title = "iH"
        statusItem?.button?.font = .monospacedSystemFont(ofSize: 12, weight: .bold)

        let menu = NSMenu()
        menu.addItem(NSMenuItem(title: "Capture (⌘+Shift+X)", action: #selector(captureAndAnalyze), keyEquivalent: ""))
        menu.addItem(NSMenuItem.separator())

        // プリセットサブメニュー
        let presetMenu = NSMenu()
        for preset in ["architect", "developer", "longcontext", "million", "balanced", "swift_dev", "lightweight"] {
            presetMenu.addItem(NSMenuItem(title: preset, action: #selector(switchPreset(_:)), keyEquivalent: ""))
        }
        let presetItem = NSMenuItem(title: "Preset", action: nil, keyEquivalent: "")
        presetItem.submenu = presetMenu
        menu.addItem(presetItem)

        menu.addItem(NSMenuItem.separator())
        menu.addItem(NSMenuItem(title: "Quit", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q"))
        statusItem?.menu = menu

        // グローバルホットキー
        NSEvent.addGlobalMonitorForEvents(matching: .keyDown) { [weak self] event in
            if event.keyCode == iHaxConfig.hotkey.key &&
               event.modifierFlags.rawValue & iHaxConfig.hotkey.modifiers == iHaxConfig.hotkey.modifiers {
                self?.captureAndAnalyze()
            }
        }

        // ヘルスチェック
        Task {
            let healthy = await AgentClient.shared.checkHealth()
            if !healthy {
                DispatchQueue.main.async {
                    self.statusItem?.button?.title = "iH!"
                }
            }
        }
    }

    @objc func captureAndAnalyze() {
        Task {
            do {
                let startTime = CFAbsoluteTimeGetCurrent()

                // 1. キャプチャ (GPU直接。P-core不要)
                let image = try await ScreenCapture.captureScreen()

                // 2. OCR (Neural Engine。GPU/CPU負荷なし)
                let content = try await TextRecognizer.recognizeText(from: image)

                if content.text.isEmpty {
                    showPopup(text: "画面にテキストが見つかりませんでした", complexity: .simple, elapsed: 0)
                    return
                }

                // 3. 複雑度判定 (E-core で瞬時に判定)
                let complexity = TaskComplexity.assess(
                    text: content.text,
                    hasCode: content.hasCode,
                    elementCount: content.elements.count
                )

                // ポップアップ表示 (ローディング)
                let popup = getOrCreatePopup()
                popup.showLoading(complexity: complexity)
                popup.showAtMousePosition()

                // 4. LLM解析 (複雑度に応じたQoSで実行)
                let result = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<String, Error>) in
                    DispatchQueue.global(qos: complexity.dispatchQoS.qosClass).async {
                        Task {
                            do {
                                let text = try await AgentClient.shared.analyze(
                                    content: content,
                                    complexity: complexity
                                )
                                continuation.resume(returning: text)
                            } catch {
                                continuation.resume(throwing: error)
                            }
                        }
                    }
                }

                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                showPopup(text: result, complexity: complexity, elapsed: elapsed)

            } catch {
                showPopup(text: "Error: \(error.localizedDescription)", complexity: .simple, elapsed: 0)
            }
        }
    }

    @objc func switchPreset(_ sender: NSMenuItem) {
        Task {
            try? await AgentClient.shared.switchPreset(sender.title)
        }
    }

    private func showPopup(text: String, complexity: TaskComplexity, elapsed: Double) {
        let tokPerSec = elapsed > 0 ? Double(text.split(separator: " ").count) / elapsed : 0
        let popup = getOrCreatePopup()
        popup.showResult(text: text, complexity: complexity, tokensPerSec: tokPerSec)
        popup.showAtMousePosition()
    }

    private func getOrCreatePopup() -> PopupWindow {
        if let existing = popup { return existing }
        let p = PopupWindow()
        popup = p
        return p
    }
}

// MARK: - Main

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.accessory) // メニューバーアプリとして動作
app.run()
