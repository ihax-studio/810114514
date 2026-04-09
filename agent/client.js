/**
 * iHax Agent Client
 * =================
 * PWAアプリ (iN glish, 英検2級, 数学) から
 * ローカルLLMサーバーを呼び出すためのクライアント。
 *
 * 使い方:
 *   <script type="module">
 *   import { iHaxAgent } from './agent/client.js';
 *   const agent = new iHaxAgent();
 *
 *   // 翻訳
 *   const en = await agent.translate('今日はいい天気ですね', 'ja_to_en');
 *
 *   // 問題生成
 *   const quiz = await agent.generateQuiz('eiken2', 5);
 *
 *   // 解説
 *   const explanation = await agent.explain('現在完了形の使い方', 'english');
 *   </script>
 */

export class iHaxAgent {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
    this.available = null; // null = 未確認, true/false
  }

  /**
   * サーバーが起動しているか確認
   */
  async checkHealth() {
    try {
      const res = await fetch(`${this.baseUrl}/health`, {
        signal: AbortSignal.timeout(3000),
      });
      if (res.ok) {
        this.available = true;
        return await res.json();
      }
    } catch {
      this.available = false;
    }
    return null;
  }

  /**
   * 日英・英日翻訳
   * @param {string} text - 翻訳するテキスト
   * @param {'ja_to_en'|'en_to_ja'} direction - 翻訳方向
   * @returns {Promise<string>} 翻訳結果
   */
  async translate(text, direction = 'ja_to_en') {
    const data = await this._post('/translate', {
      text,
      direction,
      temperature: 0.3,
    });
    return data?.translated ?? null;
  }

  /**
   * 英検・英語問題の自動生成
   * @param {string} topic - トピック ('vocabulary', 'grammar', 'eiken2')
   * @param {number} count - 問題数
   * @param {'easy'|'medium'|'hard'} difficulty - 難易度
   * @returns {Promise<Array>} 問題配列
   */
  async generateQuiz(topic = 'eiken2', count = 5, difficulty = 'medium') {
    const data = await this._post('/quiz', {
      topic,
      count,
      difficulty,
      temperature: 0.7,
    });
    return data?.questions ?? [];
  }

  /**
   * 解説生成
   * @param {string} question - 質問
   * @param {'english'|'math'} subject - 科目
   * @returns {Promise<string>} 解説
   */
  async explain(question, subject = 'english') {
    const data = await this._post('/explain', {
      question,
      subject,
      language: 'ja',
      temperature: 0.5,
    });
    return data?.explanation ?? null;
  }

  /**
   * 汎用テキスト生成
   * @param {string} prompt - プロンプト
   * @param {object} options - { max_tokens, temperature, top_k }
   * @returns {Promise<string>} 生成テキスト
   */
  async generate(prompt, options = {}) {
    const data = await this._post('/generate', {
      prompt,
      max_tokens: options.max_tokens ?? 512,
      temperature: options.temperature ?? 0.7,
      top_k: options.top_k ?? 40,
    });
    return data?.text ?? null;
  }

  /** @private */
  async _post(path, body) {
    try {
      const res = await fetch(`${this.baseUrl}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(30000), // LLM推論は時間かかる
      });
      if (!res.ok) return null;
      return await res.json();
    } catch {
      return null;
    }
  }
}
