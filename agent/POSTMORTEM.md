# LLM制作爆死回顧録 → iHax Agent 設計書

## 1. 何が起きたか（爆死の要約）

2023年夏、8GB GPUでGPT-1再現を試みて失敗した。

### ぶち当たった壁と対策の全記録

| # | 問題 | 対策 | 結果 |
|---|------|------|------|
| 1 | データ13GBがメモリに載らない | HuggingFace Dataset API + numpy.memmap | 解決 |
| 2 | tokenizer（BPE）が未知 | tiktokenライブラリのGPT2Tokenizer | 解決 |
| 3 | Positional Encodingが違う | 学習済みEmbeddingに変更 | 解決 |
| 4 | CUDA OOM（モデルサイズ） | heads=12→6, layers=12→6に縮小 | 妥協 |
| 5 | Lossが収束しない | PostLN → PreLN構造に変更 | 解決 |
| 6 | 生成がカンマ連打 | Temperature + TopK サンプリング追加 | 解決 |
| 7 | まだGPUが足りない | float16 mixed precision (autocast+GradScaler) | 一時的に解決 |
| 8 | Loss が nan になる | float16 → bfloat16 に変更 | 部分的に解決 |
| 9 | 学習が遅い | torch.compile + cudnn.benchmark + tf32 | 解決 |
| 10 | まだGPU足りない | 勾配累積 + gc.collect + empty_cache | 限界 |
| 11 | **根本的にGPU不足** | **→ Apple Silicon統合メモリで解決する** | **次のステップ** |

### 根本原因

- 8GB VRAM では GPT-1（163M params）すら学習不可能
- パラメータ 0.65GB + 学習データ + 計算グラフ + 勾配 = 8GBを大幅に超過
- 系列長を十分に取れない → 文脈理解が不十分
- Label Smoothing すら CUDA OOM で実装不可

---

## 2. ハードウェア選定: M1 Max 64GB vs M2 Max 64GB(32GB)

### Apple Silicon が LLM に強い理由

```
従来GPU: CPU RAM (16GB) ←PCIe→ VRAM (8GB)  ← ボトルネック
Apple:   統合メモリ (64GB) ← CPU/GPU/Neural Engine共有 ← これが革命
```

NVIDIA 8GB GPU で爆死した全問題が、統合メモリ 64GB で根本解決される。

### 比較表

| 項目 | M1 Max 64GB | M2 Max 32GB | M2 Max 64GB |
|------|------------|------------|------------|
| CPU | 10コア | 12コア | 12コア |
| GPU | 32コア | 38コア | 38コア |
| Neural Engine | 16コア | 16コア | 16コア |
| メモリ帯域 | 400 GB/s | 400 GB/s | 400 GB/s |
| 統合メモリ | **64GB** | 32GB | **64GB** |
| MLX推論速度 | 基準 | +10-15% | +10-15% |
| 中古相場 | 18-25万円 | 20-28万円 | 30-40万円 |
| 動かせるモデル(4bit) | **~33Bパラメータ** | ~13Bパラメータ | **~33Bパラメータ** |
| 動かせるモデル(8bit) | **~20Bパラメータ** | ~8Bパラメータ | **~20Bパラメータ** |

### 結論: 60GB GPU + 400GB SSD で限界突破

```
ターゲット: M5 Pro 64GB (2026)
- メモリ帯域: ~273 GB/s (Proチップ)
- 統合メモリ: 64GB
- SSD: 400GB (モデル保管 + KV overflow)
- GPU割り当て: 60GB (sudo sysctl iogpu.wired_limit_mb=61440)
- macOS用: 4GB + SSD swap 20GB (保険)
```

### 60GB GPU メモリ予算

```
=== 75%制限を突破 ===
デフォルト: 64GB × 75% = 48GB しかGPUに使えない
sudo sysctl iogpu.wired_limit_mb=61440 で 60GB に引き上げ
macOS: 4GB物理 + 20GB仮想メモリ(SSD 15GB/s) で十分動く

=== 60GB で何が載るか ===

70B 6bit (ほぼ原品質): モデル53GB + KV 5GB = 58GB ✓
70B 5bit + 8K context:  モデル47GB + KV 8GB = 55GB ✓
70B 4bit + 16K context: モデル40GB + KV 15GB = 55GB ✓
Coder32B + 8B 同時:     20GB + 9GB = 29GB ✓ (31GB余り)
8B + 1M context:        モデル5GB + KV 16GB = 21GB ✓ (39GB余り!)

=== 1M コンテキスト (100万トークン) ===
70Bでは1MのKVcache = 128GB → 不可能
8Bなら1MのKVcache = 16GB → 可能！
KV Q4量子化なら4GBまで縮小 → モデル5GB + KV 4GB = 9GB
→ 残り51GBを記憶システム・embeddingに全振り
→ 記憶システムが知性を補完、8Bでも実質70B級の文脈理解
```

### SSD 400GB ストレージ計画

```
全モデル保管:                ~202GB
  70B 6bit:    ~60GB
  70B 5bit:    ~50GB
  70B 4bit:    ~42GB
  Coder 32B:   ~22GB
  General 32B: ~22GB
  8B:          ~6GB

KV cache overflow:           ~50GB
FineTuning用データ:          ~50GB
余り (新モデル追加):         ~98GB
──────────────────────────────────
合計:                        400GB

全モデルをSSDに保管し、用途に応じてGPUにロードする。
モデル切り替えはSSD 15GB/s → 70B 4bitを3秒でロード可能。
```

### 8つのプリセット

| プリセット | 構成 | GPU使用 | コンテキスト | 用途 |
|-----------|------|---------|-------------|------|
| **architect** | 70B 6bit | 58GB | 4K | OS設計・アーキテクチャ |
| **developer** | 70B 5bit | 55GB | 8K | 開発メイン |
| **longcontext** | 70B 4bit | 55GB | 16K | 大規模コード解析 |
| **million** | 8B 1M ctx | 21GB | **1M** | プロジェクト全体理解 |
| **hybrid_1m** | 8B(1M) ↔ 32B切替 | 9-28GB | 1M→8K | 読む(1M)→書く(32B) |
| **balanced** | Coder32B + 8B | 29GB | 8K | コード特化 + 記憶 |
| **swift_dev** | 32B + 8B | 29GB | 8K | Swift/SwiftUI開発 |
| **lightweight** | 8B | 9GB | 8K | Xcode+Figma同時OK |

### M5 Proでの推論速度予測

```
M5 Pro 帯域 273 GB/s での推定 tok/s:
- 70B 6bit: ~5-8 tok/s   (設計議論に十分)
- 70B 5bit: ~7-10 tok/s  (開発に快適)
- 70B 4bit: ~8-12 tok/s  (サクサク)
- 33B 4bit: ~15-25 tok/s (爆速)
- 8B 4bit:  ~40-60 tok/s (瞬殺)
- 8B 1M ctx: ~20-30 tok/s (KV Q4量子化時)

※ LLM推論はメモリ帯域律速。
※ 70B 6bitの品質はDeepSeek/Qwen超え (ほぼfloat16原品質)。
※ 1Mコンテキストは8Bモデルだが、記憶システムが知性を補完。
```

**8GB GPU爆死 → 60GB GPU + 400GB SSD。
70Bほぼ原品質 & 1Mコンテキスト & OS開発・ネイティブ開発全対応。
記事の全教訓を昇華した究極構成。**

---

## 3. iHax Agent アーキテクチャ設計

### コンセプト

現在のiHax（教育PWAプラットフォーム）に、ローカルLLMを活用したAIレイヤーを追加する。

```
┌──────────────────────────────────────────────────────────┐
│                    ブラウザ (PWA)                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │ iN glish │  │ 英検2級   │  │ 数学III  │               │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘               │
│       └─────────────┼─────────────┘                      │
│                     │ fetch API                           │
│                     ▼                                     │
│  ┌───────────────────────────────────────────────┐       │
│  │         iHax Agent API (localhost:8000)        │       │
│  └───────────────────────┬───────────────────────┘       │
└──────────────────────────┼────────────────────────────────┘
                           │
┌──────────────────────────┼────────────────────────────────┐
│  iHax Agent Server (FastAPI)                              │
│                          ▼                                │
│  ┌────────────────────────────────────────────────┐      │
│  │              ルーティング & キャッシュ             │      │
│  └──┬────────────┬────────────┬───────────────────┘      │
│     ▼            ▼            ▼                           │
│  ┌───────┐  ┌────────┐  ┌────────┐  ┌──────────────┐    │
│  │問題生成│  │ 解説   │  │ 汎用   │  │  翻訳 (CDN)  │    │
│  │ Quiz  │  │ Tutor  │  │Generate│  │  外部API     │    │
│  └──┬────┘  └───┬────┘  └───┬────┘  └──────────────┘    │
│     └───────────┼───────────┘                             │
│                 ▼                                         │
│  ┌──────────────────────────────────────┐                │
│  │     3層 記憶システム (Memory)         │                │
│  │  ┌────────────────────────────────┐  │                │
│  │  │ Short: 直近N回 (コンテキスト内) │  │                │
│  │  ├────────────────────────────────┤  │                │
│  │  │ Mid: SQLite + Embedding検索    │  │  ← 24GB余裕   │
│  │  ├────────────────────────────────┤  │                │
│  │  │ Long: 要約 + Profile + Facts   │  │                │
│  │  └────────────────────────────────┘  │                │
│  └──────────────────┬───────────────────┘                │
│                     ▼                                     │
│  ┌──────────────────────────────────────┐                │
│  │    2層 MLX Engine (M5 Pro 64GB)      │                │
│  │                                      │                │
│  │  [Primary]  33B/70B ── 推論・生成    │                │
│  │  [Secondary]  8B   ── 要約・分類     │                │
│  │                                      │                │
│  │  balanced:    33B+8B 同時 (~30GB)    │                │
│  │  max_quality: 70B↔8B 切替 (~40GB)   │                │
│  └──────────────────────────────────────┘                │
└───────────────────────────────────────────────────────────┘
```

### 機能一覧

| 機能 | 説明 | 使用モデル | 接続先PWA |
|------|------|-----------|----------|
| **translate** | 日英・英日翻訳 | **無料CDN** (LLMメモリ節約) | iN glish, 英検2級 |
| **generate_quiz** | 英検問題の自動生成 | Primary (33B/70B) | 英検2級 |
| **explain** | 数学・英語の解説生成 | Primary (33B/70B) | 全アプリ |
| **generate** | 汎用テキスト生成 | Primary + 記憶コンテキスト | 全アプリ |
| **summarize** | 会話要約 (記憶圧縮) | Secondary (8B) | 内部 |
| **classify** | テキスト分類 (記憶整理) | Secondary (8B) | 内部 |
| **extract_facts** | 事実抽出 (学習記録) | Secondary (8B) | 内部 |
| **memory/search** | 過去の会話検索 | Embedding model | 全アプリ |
| **memory/profile** | ユーザープロファイル | SQLite | 全アプリ |

### 翻訳の設計方針

**翻訳は無料CDNで処理。LLMの貴重なメモリは記憶・推論に全振りする。**

```
翻訳: 無料CDN (品質は十分、コスト0、メモリ0)
推論: ローカル33B/70B (記憶コンテキスト付き)
記憶: SQLite + Embedding (ディスクベース、メモリ最小限)
要約: ローカル8B (記憶の圧縮・整理専用)

→ LLMのメモリを翻訳に使わない分、
   記憶システムに24GB多く使える。これがデカい。
```

---

## 4. 記事の教訓 → iHax Agent への適用

### 爆死から学んだこと、そのまま使える知識

| 記事で学んだこと | iHax Agentでの活用 |
|-----------------|-------------------|
| memmap でデータをメモリに載せる | 大きなモデルのメモリマップドロード (MLXが自動処理) |
| BPE tokenizer | MLXモデルは学習済みtokenizer付き、実装不要 |
| PreLN構造 | 最新モデルは全てPreLN、選ぶ側になった |
| Temperature + TopK | iHax Agentの生成パラメータとして直接使う |
| bfloat16 | MLXはデフォルトでbfloat16対応 |
| torch.compile | MLXは最初からコンパイル最適化済み |
| 勾配累積 | 推論メインなので不要（FineTuning時には使う） |
| Loss Spike対策 | FineTuning時のCheckPoint保存で活用 |

### 記事でやり残したこと → iHax Agentで実現

```
1. Instruction Tuning (SFT/RLHF)
   → 既存の英検2級データ(300問+)をSFTデータとして活用可能
   → M1 Max 64GB なら LoRA/QLoRA で 13B モデルの FineTuning が可能

2. Label Smoothing
   → MLX の FineTuning で実装可能 (CUDA OOMとは無縁)

3. より長い系列長
   → 64GB なら context length 4096-8192 で推論可能
   → 記事で苦しんだ「系列長不足」が根本解決
```

---

## 5. 実装ロードマップ

### Phase 0: 今（Mac到着前）
- [x] PWAプラットフォーム構築済み (iN glish, 英検2級, 数学)
- [x] LLM制作の知識蓄積済み (この記事の全教訓)
- [ ] Agent APIのインターフェース設計 ← 今ここ
- [ ] PWA側のfetch呼び出しコード準備

### Phase 1: Mac到着直後
- [ ] Homebrew + Python環境構築
- [ ] MLXインストール + モデルダウンロード (Llama 3.1 8B 4bit)
- [ ] iHax Agent Server起動確認
- [ ] PWA → localhost API 疎通確認

### Phase 2: 基本機能
- [ ] 翻訳API (JP↔EN) 実装
- [ ] 英検問題生成API実装
- [ ] 解説生成API実装

### Phase 3: 発展
- [ ] 英検データでLoRA FineTuning
- [ ] Whisper統合 (発音チェック)
- [ ] より大きなモデル (13B/22B) への切り替え
- [ ] マルチモデル並列推論

---

## 6. 技術的な注意事項

### Apple Silicon + MLX の制約
- GPU学習はMLXかcoremltools経由のみ (PyTorchのMPS backendは不安定)
- CUDAは使えない → NVIDIA専用コードは全て書き直し
- 統合メモリは共有 → アプリ + モデル + OS で 64GB を分け合う
- 実効使用可能メモリはモデル用に約45-50GB程度

### iHax Agent 開発時の原則
1. **推論ファースト**: 学習より推論を優先。既存モデルを最大活用
2. **量子化前提**: 4bit量子化で十分。8bitは品質向上が小さい割にメモリ2倍
3. **キャッシュ活用**: 同じ問い合わせはキャッシュして再推論を避ける
4. **段階的拡張**: 8B → 13B → 22B と段階的にモデルサイズを上げる
