require('dotenv').config()
const fastify = require('fastify')({ logger: true })

const { createAskService, AskServiceError } = require('./services/askService')

const askService = createAskService({ logger: fastify.log })

fastify.get('/health', async () => ({ ok: true }))

fastify.get('/ask', async (request, reply) => {
  const q = (request.query.q || '').toString().trim()
  if (!q) return reply.code(400).send({ error: 'Missing q query param' })
  return respondWithAnswer(q, reply)
})

fastify.post('/ask', async (request, reply) => {
  const question = (request.body?.question || '').toString().trim()
  if (!question) return reply.code(400).send({ error: 'Body must have {question}' })
  return respondWithAnswer(question, reply)
})

async function respondWithAnswer(question, reply) {
  try {
    const result = await askService.handleAsk(question)
    return reply.send(result)
  } catch (err) {
    if (err instanceof AskServiceError) {
      const status = err.statusCode || 500
      if (status >= 500) {
        fastify.log.error(err)
      } else {
        fastify.log.warn(err)
      }
      return reply.code(status).send({ error: err.message })
    }
    fastify.log.error(err)
    return reply.code(500).send({ error: err.message || 'Failed to answer' })
  }
}

const start = async () => {
  const basePort = Number(process.env.PORT) || 3000
  const maxTries = 11
  for (let attempt = 0; attempt < maxTries; attempt++) {
    const port = basePort + attempt
    try {
      await fastify.listen({ port, host: '0.0.0.0' })
      fastify.log.info(`Server running at http://localhost:${port}`)
      return
    } catch (err) {
      if (err && err.code === 'EADDRINUSE') {
        fastify.log.warn(`Port ${port} in use, trying ${port + 1}...`)
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
