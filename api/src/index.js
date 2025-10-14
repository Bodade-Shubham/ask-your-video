// index.js
require('dotenv').config()
const fs = require('fs')
const path = require('path')
const fastify = require('fastify')({ logger: true })
const OpenAI = require('openai')

// --- OpenAI client ---
// Construct the client only if the API key is provided so the server can start
// without crashing when the key is not set. Endpoints will return 500 until
// the key is configured (handleAsk already checks for the env var).
let openai = null
if (process.env.OPENAI_API_KEY) {
  try {
    openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY })
  } catch (e) {
    fastify.log.warn('Failed to create OpenAI client at startup: ' + (e.message || e))
    openai = null
  }
} else {
  fastify.log.warn('OPENAI_API_KEY not set; /ask will return 500 until you set it.')
}

// --- Load segments once ---
const SEGMENTS_PATH = path.join(__dirname, 'segment.json')
if (!fs.existsSync(SEGMENTS_PATH)) {
  fastify.log.warn(`segment.json not found at ${SEGMENTS_PATH}. /ask will 422 until you add it.`)
}
let segments = []
let segmentEmbeddings = null // filled on first use

function loadSegments() {
  try {
    const raw = JSON.parse(fs.readFileSync('src/segment.json', 'utf8'))
    // Support both {segments:[...]} and raw array
    const list = Array.isArray(raw) ? raw : raw.segments || []
    segments = list
      .map(s => ({
        text: (s.text || '').trim(),
        start: s.start ?? s.seek ?? null,
        end: s.end ?? null,
        id: s.id ?? null,
      }))
      .filter(s => s.text.length > 0)
  } catch (e) {
    segments = []
  }
}
loadSegments()

// --- Small utils ---
const dot = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0)
const norm = a => Math.sqrt(a.reduce((s, v) => s + v * v, 0))
const cosine = (a, b) => (norm(a) && norm(b) ? dot(a, b) / (norm(a) * norm(b)) : 0)

async function ensureEmbeddings() {
  if (!segments.length) throw new Error('No segments loaded')
  if (segmentEmbeddings) return

  // Chunk inputs to respect token & rate limits (simple batching)
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
    throw new Error('Embedding count mismatch')
  }
}

// --- /health ---
fastify.get('/health', async () => ({ ok: true }))

// --- /ask (GET for quick tests) ---
fastify.get('/ask', async (request, reply) => {
  const q = (request.query.q || '').toString().trim()
  if (!q) return reply.code(400).send({ error: 'Missing q query param' })
  return handleAsk(q, reply)
})

// --- /ask (POST: {question}) ---
fastify.post('/ask', async (request, reply) => {
  const question = (request.body?.question || '').toString().trim()
  if (!question) return reply.code(400).send({ error: 'Body must have {question}' })
  return handleAsk(question, reply)
})

async function handleAsk(question, reply) {
  try {
    if (!process.env.OPENAI_API_KEY) {
      return reply.code(500).send({ error: 'OPENAI_API_KEY not set' })
    }
    if (!segments.length) {
      // try a hot reload if file added after boot
      loadSegments()
      if (!segments.length) return reply.code(422).send({ error: 'segment.json is empty or missing' })
    }

    await ensureEmbeddings()

    // Embed the question
    const qEmb = await openai.embeddings.create({
      model: 'text-embedding-3-small',
      input: question,
    })
    const qVec = qEmb.data[0].embedding

    // Rank segments by cosine similarity
    const scored = segmentEmbeddings.map((vec, i) => ({
      i,
      score: cosine(qVec, vec),
    }))
    scored.sort((a, b) => b.score - a.score)

    // Take top-K and cap total context tokens (naive char cap here)
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

    // Ask the model using the context
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

    return reply.send({
      ok: true,
      answer,
      sources: top.map(s => ({
        start: s.start,
        end: s.end,
        score: +s.score.toFixed(4),
        text: s.text.length > 140 ? s.text.slice(0, 140) + '…' : s.text,
      })),
    })
  } catch (err) {
    // `request` is not available in this function (we only have `question` and `reply`).
    // Use the Fastify logger instead to avoid a ReferenceError.
    fastify.log.error(err)
    return reply.code(500).send({ error: err.message || 'Failed to answer' })
  }
}

function fmtTime(sec) {
  // Whisper "start"/"end" are seconds; handle numbers or strings
  const s = Number(sec) || 0
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const ss = Math.floor(s % 60)
  return [h, m, ss]
    .map((v, i) => (i === 0 ? String(v) : String(v).padStart(2, '0')))
    .join(':')
}

// --- start server with port-retry for EADDRINUSE ---
const start = async () => {
  const basePort = Number(process.env.PORT) || 3000
  const maxTries = 11 // try basePort ... basePort + 10
  for (let attempt = 0; attempt < maxTries; attempt++) {
    const port = basePort + attempt
    try {
      await fastify.listen({ port, host: '0.0.0.0' })
      fastify.log.info(`Server running at http://localhost:${port}`)
      return
    } catch (err) {
      if (err && err.code === 'EADDRINUSE') {
        fastify.log.warn(`Port ${port} in use, trying ${port + 1}...`)
        // try next port
        continue
      }
      fastify.log.error(err)
      process.exit(1)
    }
  }
  fastify.log.error('Failed to bind to a port after multiple attempts')
  process.exit(1)
}
start()
