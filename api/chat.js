const { GoogleGenerativeAI } = require('@google/generative-ai');

// Initialize clients lazily
let genAI = null;
const apiKey = process.env.GEMINI_API_KEY;
const modelName = process.env.GEMINI_MODEL || 'gemini-1.5';
const hfKey = process.env.HUGGINGFACE_API_KEY;
const hfModel = process.env.HUGGINGFACE_MODEL || 'mistralai/mistral-small';
const openaiKey = process.env.OPENAI_API_KEY;
const openaiModel = process.env.OPENAI_MODEL || 'gpt-3.5-turbo';

if (apiKey && apiKey.trim() !== '') {
  try {
    genAI = new GoogleGenerativeAI(apiKey);
    console.log('Gemini client initialized.');
  } catch (e) {
    console.error('Failed to init Gemini client:', e.message || e);
  }
}

module.exports = async (req, res) => {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });
  const { message, history } = req.body || {};
  if (!message) return res.status(400).json({ error: 'Missing message' });

  try {
    let responseText = '';
    if (genAI) {
      try {
        const model = genAI.getGenerativeModel({ model: modelName, systemInstruction: process.env.SYSTEM_INSTRUCTION || '' });
        const chatHistory = (history || []).map(turn => ({ role: turn.role, parts: [{ text: turn.text }] }));
        const chat = model.startChat({ history: chatHistory });
        const result = await chat.sendMessage(message);
        responseText = result.response.text();
      } catch (err) {
        console.error('Gemini model error, trying Hugging Face fallback if configured:', err.message || err);
        if (!hfKey) throw err;
      }
    }

    if (!responseText && hfKey) {
      // Prefer OpenAI if configured (often more reliable DNS-wise)
      if (openaiKey) {
        try {
          const openResp = await fetchWithRetries('https://api.openai.com/v1/chat/completions', {
            method: 'POST',
            headers: {
              Authorization: `Bearer ${openaiKey}`,
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model: openaiModel, messages: [{ role: 'user', content: message }], max_tokens: 512 })
          }, 5, 700);

          if (!openResp.ok) {
            const t = await openResp.text();
            throw new Error(`OpenAI error ${openResp.status} ${t}`);
          }

          const openData = await openResp.json();
          if (openData && openData.choices && openData.choices[0] && openData.choices[0].message && openData.choices[0].message.content) {
            responseText = openData.choices[0].message.content;
          }
        } catch (openErr) {
          console.warn('OpenAI call failed, falling back to Hugging Face if available:', openErr && (openErr.code || openErr.message));
        }
      }

      const fetch = global.fetch || (await import('node-fetch')).default;

      // Helper: fetch with retries + exponential backoff
      async function fetchWithRetries(url, opts = {}, attempts = 5, backoff = 700) {
        let lastErr;
        for (let i = 0; i < attempts; i++) {
          try {
            // timeout using AbortController
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 15000);
            const response = await fetch(url, { signal: controller.signal, ...opts });
            clearTimeout(timeout);
            if (!response.ok && response.status >= 500 && i < attempts - 1) {
              // server error, retry
              lastErr = new Error(`HTTP ${response.status}`);
              console.warn(`Fetch attempt ${i + 1} failed with ${response.status}, retrying...`);
              await new Promise(r => setTimeout(r, backoff * Math.pow(2, i)));
              continue;
            }
            return response;
          } catch (err) {
            lastErr = err;
            // retry on network errors
            if (i < attempts - 1) {
              console.warn(`Network error on fetch attempt ${i + 1}:`, err && (err.code || err.message));
              await new Promise(r => setTimeout(r, backoff * Math.pow(2, i)));
            }
          }
        }
        // attach attempts info for diagnostics
        if (lastErr && typeof lastErr === 'object') lastErr.attempts = attempts;
        throw lastErr;
      }

      try {
        const hfResp = await fetchWithRetries(`https://api-inference.huggingface.co/models/${hfModel}`, {
          method: 'POST',
          headers: { Authorization: `Bearer ${hfKey}`, 'Content-Type': 'application/json' },
          body: JSON.stringify({ inputs: message, parameters: { max_new_tokens: 300 } })
        });

        if (!hfResp.ok) {
          const text = await hfResp.text();
          throw new Error(`HuggingFace inference error: ${hfResp.status} ${text}`);
        }

        const hfData = await hfResp.json();
        if (Array.isArray(hfData) && hfData[0].generated_text) responseText = hfData[0].generated_text;
        else if (hfData.generated_text) responseText = hfData.generated_text;
        else responseText = JSON.stringify(hfData);
      } catch (err) {
        // Surface DNS / network errors as 503 so client can use offline fallback
        if (err && (err.code === 'ENOTFOUND' || (err.message && err.message.includes('ENOTFOUND')))) {
          console.error('Hugging Face host resolution failed:', err.code || err.message || err);
          return res.status(503).json({ error: 'Inference provider unreachable (DNS)', provider: 'huggingface', code: err.code || null, message: 'Host resolution failed. Please try again later.' });
        }
        // For other network-type errors include a hint and attempts if available
        if (err && (err.code === 'ECONNRESET' || err.code === 'ETIMEDOUT' || (err.message && (err.message.includes('timeout') || err.message.includes('aborted'))))) {
          console.error('Hugging Face network error:', err.code || err.message || err);
          return res.status(502).json({ error: 'Inference provider network error', provider: 'huggingface', code: err.code || null, attempts: err.attempts || null });
        }
        throw err;
      }
    }

    // Final fallback
    if (!responseText) responseText = 'Keine Antwort verfügbar.';

    // Simple markdown -> html conversions (same as site expects)
    let formattedText = responseText.replace(/\n\n/g, '<br><br>').replace(/\n/g, '<br>').replace(/\*\*(.*?)\*\*/g, '<b>$1</b>').replace(/\*(.*?)\*/g, '<i>$1</i>').replace(/^- (.+)$/gm, '• $1');
    res.json({ reply: formattedText });
  } catch (error) {
    console.error('API error:', error && (error.message || error.code || error));
    const safeMessage = (error && error.message) ? String(error.message) : 'Unknown error';
    // Avoid leaking stack traces to clients; provide structured diagnostics for client fallback
    return res.status(500).json({ error: safeMessage, code: error && error.code ? error.code : null });
  }
};
