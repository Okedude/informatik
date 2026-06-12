const express = require('express');
const dotenv = require('dotenv');
const { GoogleGenerativeAI } = require('@google/generative-ai');

// Load environment variables
dotenv.config();

const app = express();
app.use(express.json());

// Serve static files from the current directory
app.use(express.static(__dirname));

const PORT = process.env.PORT || 8000;

// Setup Gemini API if key is present
let genAI = null;
const apiKey = process.env.GEMINI_API_KEY;
// Allow configuring the exact Gemini model via environment variable.
const modelName = process.env.GEMINI_MODEL || 'gemini-1.5';
// Hugging Face fallback (free-hosted models require an API key but many have free tiers)
const hfKey = process.env.HUGGINGFACE_API_KEY;
const hfModel = process.env.HUGGINGFACE_MODEL || 'mistralai/mistral-small';
const openaiKey = process.env.OPENAI_API_KEY;
const openaiModel = process.env.OPENAI_MODEL || 'gpt-3.5-turbo';

// Helper: fetch with retries + exponential backoff (used for Hugging Face calls)
async function fetchWithRetries(fetchFn, url, opts = {}, attempts = 5, backoff = 700) {
  let lastErr;
  for (let i = 0; i < attempts; i++) {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 15000);
      const response = await fetchFn(url, { signal: controller.signal, ...opts });
      clearTimeout(timeout);
      if (!response.ok && response.status >= 500 && i < attempts - 1) {
        lastErr = new Error(`HTTP ${response.status}`);
        console.warn(`Fetch attempt ${i + 1} failed with ${response.status}, retrying...`);
        await new Promise(r => setTimeout(r, backoff * Math.pow(2, i)));
        continue;
      }
      return response;
    } catch (err) {
      lastErr = err;
      if (i < attempts - 1) {
        console.warn(`Network error on fetch attempt ${i + 1}:`, err && (err.code || err.message));
        await new Promise(r => setTimeout(r, backoff * Math.pow(2, i)));
      }
    }
  }
  if (lastErr && typeof lastErr === 'object') lastErr.attempts = attempts;
  throw lastErr;
}

// Updated system instruction – allows deepfake topic knowledge + survey data
const systemInstruction = `Du bist ein intelligenter KI-Assistent für das Schulprojekt "DEEP/FAKE" mit dem Thema: "Inwiefern gefährden KI-generierte Deep-Fakes die politische Meinungsbildung?".

Du darfst folgende Fragen beantworten:
1. Alle Fragen zum Thema Deepfakes, KI-generierte Medien und deren politischer Einfluss (das ist das Kernthema des Projekts)
2. Alle Fragen zu unserer eigenen Umfrage (n = 132 Teilnehmende, April–Juni 2026)
3. Fragen zu den Projektbefunden, Quellen, dem Quiz oder der Analyse

Für Fragen, die gar nichts mit KI, Deepfakes, Politik, Medien oder dem Projekt zu tun haben (z. B. "Wie koche ich Nudeln?" oder "Was ist die Hauptstadt von Peru?"), erkläre freundlich, dass du als DEEP/FAKE-Assistent nur zu diesem Themenbereich helfen kannst.

=== UNSERE EIGENEN UMFRAGEDATEN (n=132, Microsoft Forms, April–Juni 2026) ===

DEMOGRAFIE:
- Junge Erwachsene (16–25 J.): 88 Personen (66.7 %)
- Erwachsene (26–59 J.): 38 Personen (28.8 %)
- Senioren (60+ J.): 6 Personen (4.5 %)
- Geschlecht: 77 weiblich, 49 männlich, 6 divers
- Altersgruppen: 15–17 (11), 18–20 (45), 21–25 (32), 26–35 (14), 36–49 (14), 50–59 (10), 60+ (6)

DIE 4 HAUPTBEFUNDE:
1. SELBSTÜBERSCHÄTZUNG: Junge Erwachsene trauen sich Deepfakes am ehesten zu erkennen (Mittelwert 2.93/4), erzielen im Quiz aber nur ~5/8 Punkte → Dunning-Kruger-Effekt.
2. SORGE WÄCHST MIT ALTER: Sorge vor Wahlbeeinflussung durch Deepfakes: Junge 2.44/4, Erwachsene 2.74/4, Senioren 3.00/4. Wer weniger technikaffin ist, fürchtet mehr.
3. KONSENS BEIM WASSERZEICHEN: Unsichtbare Kennzeichnungspflicht für KI-Medien: Mittelwert 3.32/4 – der höchste Wert aller abgefragten Massnahmen.
4. FEHLENDE SENSIBILISIERUNG: Nur 1.70/4 glauben, man lerne heute genug Desinformation zu erkennen. Vertrauen in Social-Media-Plattformen: nur 1.73/4.

ALLE AUSSAGEN (Skala 1–4, Mittelwert / Junge / Erwachsene / Senioren):
- Traue mir zu, Deepfake zu erkennen: 2.72 / 2.93 / 2.29 / 2.33
- Sorge vor Wahlfälschung: 2.55 / 2.44 / 2.74 / 3.00
- Plattformen tun genug: 1.73 / 1.76 / 1.71 / 1.50
- Man lernt genug: 1.70 / 1.68 / 1.79 / 1.50
- Bedrohung für den Frieden: 2.75 / 2.76 / 2.71 / 2.83
- Hinterfrage Videos öfter: 3.18 / 3.31 / 2.92 / 3.00
- Strafbarkeit (reale Personen): 3.23 / 3.27 / 3.11 / 3.50
- Wasserzeichen-Pflicht für KI: 3.32 / 3.41 / 3.11 / 3.33
- Satire rechtfertigt Deepfakes: 1.94 / 1.90 / 2.03 / 2.00
- Vertrauen klass. Medien > Social: 3.13 / 3.22 / 2.89 / 3.33
- Bald keine Unterscheidung möglich: 2.95 / 2.95 / 2.95 / 3.00

VERHALTEN BEI VERDÄCHTIGEN NACHRICHTEN:
- Quelle prüfen: 43 (32.6%), Ignorieren: 42 (31.8%), Plattform melden: 34 (25.8%), Teilen/Fragen: 13 (9.8%)

LIMITATIONEN:
- Senioren-Gruppe zu klein (n=6), nicht repräsentativ
- Self-Selection-Bias (Online-Umfrage, tech-affine Teilnehmende)
- Geschlechterverteilung weicht leicht von Schweizer Bevölkerung ab

=== ALLGEMEINES PROJEKT-WISSEN ===

Deepfakes und politische Meinungsbildung:
- Deepfakes können Vertrauen in Medien und politische Institutionen untergraben, auch wenn sie nie gesehen wurden (sog. "liar's dividend")
- Studien zeigen: Bereits das WISSEN, dass es Deepfakes gibt, führt dazu, dass Menschen echte Videos stärker anzweifeln
- Politische Akteure können Fehlinformationen verbreiten und anschliessend echte Videos als Fake abtun
- Besonders gefährlich: im Wahlkampf kurz vor Wahlen veröffentlichte Deepfakes (zu wenig Zeit zur Widerlegung)
- Kognitive Bias: Menschen glauben eher Inhalte, die ihre bestehende Weltanschauung bestätigen (Confirmation Bias)
- Social Media Algorithmen verstärken emotionale Inhalte, was Deepfakes viraler macht als Richtigstellungen

Quellen des Projekts: Reuters Institute Digital News Report 2024, Harvard Kennedy School (2022), EU AI Act (2024), SRF (2024), NCSC Bericht (2024)

Antworte immer auf Deutsch, präzise und gut strukturiert. Nutze Fettschrift (**text**) und Listen für Übersichtlichkeit.`;

if (apiKey && apiKey.trim() !== '') {
  try {
    genAI = new GoogleGenerativeAI(apiKey);
    console.log("Gemini API Client initialized successfully.");
  } catch (error) {
    console.error("Failed to initialize Gemini API:", error);
  }
} else {
  console.log("No GEMINI_API_KEY configured. Chatbot will run in Offline-Mode.");
}

// Chat endpoint
app.post('/api/chat', async (req, res) => {
  const { message, history } = req.body;

  if (!message) {
    return res.status(400).json({ error: "Missing message field." });
  }

  // If no Gemini client is initialized, signal client to use fallback
  if (!genAI && !hfKey) {
    return res.status(503).json({ error: "No GEMINI_API_KEY configured and no HUGGINGFACE_API_KEY found. Set GEMINI_API_KEY or HUGGINGFACE_API_KEY (and HUGGINGFACE_MODEL) in .env." });
  }

  try {
    let responseText = '';
    if (genAI) {
      try {
        const model = genAI.getGenerativeModel({
          model: modelName,
          systemInstruction: systemInstruction
        });

        // Build history for multi-turn conversations
        const chatHistory = (history || []).map(turn => ({
          role: turn.role,
          parts: [{ text: turn.text }]
        }));

        const chat = model.startChat({ history: chatHistory });
        const result = await chat.sendMessage(message);
        responseText = result.response.text();
      } catch (err) {
        console.error('Gemini model error, will try configured fallbacks:', err.message || err);
        // Prefer OpenAI if configured (often more reliable DNS-wise)
        if (openaiKey) {
          const fetch = global.fetch || (await import('node-fetch')).default;
          try {
            const openResp = await fetchWithRetries(fetch, 'https://api.openai.com/v1/chat/completions', {
              method: 'POST',
              headers: { 'Authorization': `Bearer ${openaiKey}`, 'Content-Type': 'application/json' },
              body: JSON.stringify({ model: openaiModel, messages: [{ role: 'user', content: message }], max_tokens: 512 })
            });

            if (!openResp.ok) {
              const t = await openResp.text();
              throw new Error(`OpenAI error ${openResp.status} ${t}`);
            }

            const openData = await openResp.json();
            if (openData && openData.choices && openData.choices[0] && openData.choices[0].message && openData.choices[0].message.content) {
              responseText = openData.choices[0].message.content;
            }
          } catch (openErr) {
            console.error('OpenAI fallback failed:', openErr && (openErr.code || openErr.message));
            // continue to HF if configured
          }
        }

        // If OpenAI didn't produce a response, try Hugging Face if configured
        if (!responseText && hfKey) {
          const fetch = global.fetch || (await import('node-fetch')).default;
          try {
            const hfResp = await fetchWithRetries(fetch, `https://api-inference.huggingface.co/models/${hfModel}`, {
              method: 'POST',
              headers: {
                'Authorization': `Bearer ${hfKey}`,
                'Content-Type': 'application/json'
              },
              body: JSON.stringify({
                inputs: message,
                parameters: { max_new_tokens: 300 }
              })
            });

            if (!hfResp.ok) {
              const text = await hfResp.text();
              throw new Error(`HuggingFace inference error: ${hfResp.status} ${text}`);
            }

            const hfData = await hfResp.json();
            if (Array.isArray(hfData) && hfData[0].generated_text) responseText = hfData[0].generated_text;
            else if (hfData.generated_text) responseText = hfData.generated_text;
            else responseText = JSON.stringify(hfData);
          } catch (hfErr) {
            if (hfErr && (hfErr.code === 'ENOTFOUND' || (hfErr.message && hfErr.message.includes('ENOTFOUND')))) {
              console.error('Hugging Face host resolution failed:', hfErr.code || hfErr.message || hfErr);
              return res.status(503).json({ error: 'Inference provider unreachable (DNS)', provider: 'huggingface', code: hfErr.code || null, message: 'Host resolution failed. Please try again later.' });
            }
            if (hfErr && (hfErr.code === 'ECONNRESET' || hfErr.code === 'ETIMEDOUT' || (hfErr.message && (hfErr.message.includes('timeout') || hfErr.message.includes('aborted'))))) {
              console.error('Hugging Face network error:', hfErr.code || hfErr.message || hfErr);
              return res.status(502).json({ error: 'Inference provider network error', provider: 'huggingface', code: hfErr.code || null, attempts: hfErr.attempts || null });
            }
            throw hfErr;
          }
        }
        // If no fallbacks applied or none produced a response, rethrow original error
        if (!responseText) throw err;
      }
    } else if (hfKey) {
      // Use Hugging Face Inference API as a fallback to a 'real' model.
      const fetch = global.fetch || (await import('node-fetch')).default;
      try {
        const hfResp = await fetchWithRetries(fetch, `https://api-inference.huggingface.co/models/${hfModel}`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${hfKey}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            inputs: message,
            parameters: { max_new_tokens: 300 }
          })
        });

        if (!hfResp.ok) {
          const text = await hfResp.text();
          throw new Error(`HuggingFace inference error: ${hfResp.status} ${text}`);
        }

        const hfData = await hfResp.json();
        if (Array.isArray(hfData) && hfData[0].generated_text) responseText = hfData[0].generated_text;
        else if (hfData.generated_text) responseText = hfData.generated_text;
        else responseText = JSON.stringify(hfData);
      } catch (hfErr) {
        if (hfErr && (hfErr.code === 'ENOTFOUND' || (hfErr.message && hfErr.message.includes('ENOTFOUND')))) {
          console.error('Hugging Face host resolution failed:', hfErr.code || hfErr.message || hfErr);
          return res.status(503).json({ error: 'Inference provider unreachable (DNS)', provider: 'huggingface', code: hfErr.code || null, message: 'Host resolution failed. Please try again later.' });
        }
        if (hfErr && (hfErr.code === 'ECONNRESET' || hfErr.code === 'ETIMEDOUT' || (hfErr.message && (hfErr.message.includes('timeout') || hfErr.message.includes('aborted'))))) {
          console.error('Hugging Face network error:', hfErr.code || hfErr.message || hfErr);
          return res.status(502).json({ error: 'Inference provider network error', provider: 'huggingface', code: hfErr.code || null, attempts: hfErr.attempts || null });
        }
        throw hfErr;
      }
    }

    // Convert Markdown to HTML for display in the chat widget
    let formattedText = responseText
      .replace(/\n\n/g, '<br><br>')
      .replace(/\n/g, '<br>')
      .replace(/\*\*(.*?)\*\*/g, '<b>$1</b>')
      .replace(/\*(.*?)\*/g, '<i>$1</i>')
      .replace(/^### (.+)$/gm, '<b>$1</b>')
      .replace(/^## (.+)$/gm, '<b>$1</b>')
      .replace(/^# (.+)$/gm, '<b>$1</b>')
      .replace(/^- (.+)$/gm, '• $1');

    res.json({ reply: formattedText });
    } catch (error) {
    console.error("Gemini API error:", error && (error.message || error.code || error));
    // Log stack if available for debugging network/fetch failures
    if (error && error.stack) console.error(error.stack);
    const msg = (error && error.message) ? String(error.message) : '';

    // If the model is unavailable, return 503 with a helpful hint to set GEMINI_MODEL or use Hugging Face
    if (msg.includes('no longer available') || msg.includes('404') || msg.includes('Not Found')) {
      return res.status(503).json({ error: `Requested model '${modelName}' unavailable. Set GEMINI_MODEL to a supported Gemini model, or set HUGGINGFACE_API_KEY and HUGGINGFACE_MODEL in .env to use a free Hugging Face model.`, provider: 'gemini' });
    }

    // For network or provider errors include additional diagnostics where available
    if (error && (error.code === 'ENOTFOUND' || (msg && msg.includes('ENOTFOUND')))) {
      return res.status(503).json({ error: 'Inference provider unreachable (DNS)', code: error.code || null });
    }

    res.status(500).json({ error: msg || 'Unknown Gemini API error', code: error && error.code ? error.code : null });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`DEEP/FAKE server running at http://localhost:${PORT}`);
});
