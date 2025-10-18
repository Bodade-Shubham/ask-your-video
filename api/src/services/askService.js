const fs = require('fs')
const path = require('path')
const OpenAI = require('openai')

class AskServiceError extends Error {
  constructor(message, statusCode = 500) {
    super(message)
    this.name = 'AskServiceError'
    this.statusCode = statusCode
  }
}

const dot = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0)
const norm = a => Math.sqrt(a.reduce((s, v) => s + v * v, 0))
const cosine = (a, b) => (norm(a) && norm(b) ? dot(a, b) / (norm(a) * norm(b)) : 0)

function fmtTime(sec) {
  const s = Number(sec) || 0
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const ss = Math.floor(s % 60)
  return [h, m, ss]
    .map((v, i) => (i === 0 ? String(v) : String(v).padStart(2, '0')))
    .join(':')
}

function createAskService({
  logger,
  openAiApiKey = process.env.OPENAI_API_KEY,
  segmentPath = path.join(__dirname, '..', 'segment.json'),
} = {}) {
  const log = logger || console

  let openai = null
  if (openAiApiKey) {
    try {
      openai = new OpenAI({ apiKey: openAiApiKey })
    } catch (err) {
      log.warn('Failed to create OpenAI client at startup: ' + (err.message || err))
      openai = null
    }
  } else {
    log.warn('OPENAI_API_KEY not set; /ask will return 500 until you set it.')
  }

  if (!fs.existsSync(segmentPath)) {
    log.warn(`segment.json not found at ${segmentPath}. /ask will 422 until you add it.`)
  }

  let segments = []
  let segmentEmbeddings = null

  function loadSegments() {
    try {
      const raw = JSON.parse(fs.readFileSync(segmentPath, 'utf8'))
      const list = Array.isArray(raw) ? raw : raw.segments || []
      segments = list
        .map(s => ({
          text: (s.text || '').trim(),
          start: s.start ?? s.seek ?? null,
          end: s.end ?? null,
          id: s.id ?? null,
        }))
        .filter(s => s.text.length > 0)
      segmentEmbeddings = null
    } catch (e) {
      segments = []
      segmentEmbeddings = null
    }
  }
  loadSegments()

  async function ensureEmbeddings() {
    if (!segments.length) throw new AskServiceError('No segments loaded', 422)
    if (segmentEmbeddings) return
    if (!openai) throw new AskServiceError('OpenAI client not initialized', 500)

    const texts = segments.map(s => s.text)
    const model = 'text-embedding-3-small'
    segmentEmbeddings = []
    const batchSize = 80

    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize)
      const res = await openai.embeddings.create({ model, input: batch })
      res.data.forEach(d => segmentEmbeddings.push(d.embedding))
    }
    if (segmentEmbeddings.length !== segments.length) {
      segmentEmbeddings = null
      throw new AskServiceError('Embedding count mismatch', 500)
    }
  }

  async function handleAsk(question) {
    if (!openAiApiKey) {
      throw new AskServiceError('OPENAI_API_KEY not set', 500)
    }
    if (!segments.length) {
      loadSegments()
      if (!segments.length) {
        throw new AskServiceError('segment.json is empty or missing', 422)
      }
    }

    await ensureEmbeddings()

    const qEmb = await openai.embeddings.create({
      model: 'text-embedding-3-small',
      input: question,
    })
    const qVec = qEmb.data[0].embedding

    const scored = segmentEmbeddings.map((vec, i) => ({
      i,
      score: cosine(qVec, vec),
    }))
    scored.sort((a, b) => b.score - a.score)

    const K = 8
    const top = scored.slice(0, K).map(s => ({ ...segments[s.i], score: s.score }))
    let context = ''
    for (const s of top) {
      const ts =
        s.start != null && s.end != null
          ? `[${fmtTime(s.start)}–${fmtTime(s.end)}] `
          : s.start != null
          ? `[${fmtTime(s.start)}] `
          : ''
      const line = `${ts}${s.text}\n`
      if ((context + line).length > 8000) break
      context += line
    }

    const system = `You answer strictly using the provided transcript excerpts.
- If the answer is not in the excerpts, say you cannot find it in the transcript.
- Always include referenced timestamps in square brackets when you use a segment.
- Be concise and factual.`

    const user = `Question: ${question}

Transcript excerpts:
${context || '(no relevant excerpts found)'}`

    const chat = await openai.chat.completions.create({
      model: 'gpt-3.5-turbo',
      max_tokens: 200,
      messages: [
        { role: 'system', content: system },
        { role: 'user', content: user },
      ],
    })

    const answer = chat.choices?.[0]?.message?.content ?? '(no answer)'

    return {
      ok: true,
      answer,
      sources: top.map(s => ({
        start: s.start,
        end: s.end,
        score: +s.score.toFixed(4),
        text: s.text.length > 140 ? s.text.slice(0, 140) + '…' : s.text,
      })),
    }
  }

  return {
    handleAsk,
    loadSegments,
  }
}

module.exports = {
  createAskService,
  AskServiceError,
}
