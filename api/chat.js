const { GoogleGenerativeAI } = require('@google/generative-ai');

// Initialize clients lazily
let genAI = null;
const apiKey = process.env.GEMINI_API_KEY;
const modelName = process.env.GEMINI_MODEL || 'gemini-1.5';
const hfKey = process.env.HUGGINGFACE_API_KEY;
const hfModel = process.env.HUGGINGFACE_MODEL || 'mistralai/mistral-small';

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
      const fetch = global.fetch || (await import('node-fetch')).default;
      const hfResp = await fetch(`https://api-inference.huggingface.co/models/${hfModel}`, {
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
    }

    // Final fallback
    if (!responseText) responseText = 'Keine Antwort verfügbar.';

    // Simple markdown -> html conversions (same as site expects)
    let formattedText = responseText.replace(/\n\n/g, '<br><br>').replace(/\n/g, '<br>').replace(/\*\*(.*?)\*\*/g, '<b>$1</b>').replace(/\*(.*?)\*/g, '<i>$1</i>').replace(/^- (.+)$/gm, '• $1');
    res.json({ reply: formattedText });
  } catch (error) {
    console.error('API error:', error);
    res.status(500).json({ error: (error && error.message) ? error.message : 'Unknown error' });
  }
};
