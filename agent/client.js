/**
 * iHax Agent Client v0.2
 * ======================
 * PWAアプリ (iN glish, 英検2級, 数学) から
 * ローカルLLMサーバー + 記憶システムを呼び出す。
 *
 * 使い方:
 *   <script type="module">
 *   import { iHaxAgent } from './agent/client.js';
 *   const agent = new iHaxAgent();
 *
 *   // 問題生成 (33B/70Bモデル + 記憶コンテキスト)
 *   const quiz = await agent.generateQuiz('eiken2', 5);
 *
 *   // 解説 (ユーザーの苦手分野を考慮)
 *   const explanation = await agent.explain('現在完了形の使い方');
 *
 *   // 記憶検索 (過去の学習履歴)
 *   const memories = await agent.searchMemory('present perfect');
 *
 *   // プリセット切替 (33B→70B)
 *   await agent.switchPreset('max_quality');
 *   </script>
 */

export class iHaxAgent {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
    this.available = null;
  }

  // --- Health & Config ---

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

  async getPresets() {
    return await this._get('/presets');
  }

  async switchPreset(preset) {
    return await this._post('/preset', { preset });
  }

  async getModelStatus() {
    return await this._get('/model/status');
  }

  // --- Quiz & Education ---

  async generateQuiz(topic = 'eiken2', count = 5, difficulty = 'medium') {
    const data = await this._post('/quiz', {
      topic, count, difficulty, temperature: 0.7,
    });
    return data?.questions ?? [];
  }

  async explain(question, subject = 'english') {
    const data = await this._post('/explain', {
      question, subject, language: 'ja', temperature: 0.5,
    });
    return data?.explanation ?? null;
  }

  // --- Generation (記憶コンテキスト付き) ---

  async generate(prompt, options = {}) {
    const data = await this._post('/generate', {
      prompt,
      max_tokens: options.max_tokens ?? 512,
      temperature: options.temperature ?? 0.7,
      top_k: options.top_k ?? 40,
    });
    return data?.text ?? null;
  }

  // --- Memory System ---

  async searchMemory(query, topK = 5) {
    const data = await this._post('/memory/search', {
      query, top_k: topK,
    });
    return data?.results ?? [];
  }

  async getProfile() {
    return await this._get('/memory/profile');
  }

  async updateProfile(key, value, confidence = 0.7) {
    return await this._post('/memory/profile', { key, value, confidence });
  }

  async storeFact(category, fact, confidence = 0.5) {
    return await this._post('/memory/fact', { category, fact, confidence });
  }

  async getMemoryStats() {
    return await this._get('/memory/stats');
  }

  async summarizeSession() {
    return await this._post('/memory/summarize', {});
  }

  // --- Private ---

  async _get(path) {
    try {
      const res = await fetch(`${this.baseUrl}${path}`, {
        signal: AbortSignal.timeout(10000),
      });
      if (!res.ok) return null;
      return await res.json();
    } catch {
      return null;
    }
  }

  async _post(path, body) {
    try {
      const res = await fetch(`${this.baseUrl}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(60000), // 70Bは時間かかる
      });
      if (!res.ok) return null;
      return await res.json();
    } catch {
      return null;
    }
  }
}
