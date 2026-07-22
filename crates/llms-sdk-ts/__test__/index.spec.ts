import { readFileSync } from 'fs'
import { dirname, resolve } from 'path'
import { fileURLToPath } from 'url'
import test from 'ava'

import {
  Llm,
  ApiType,
  MessageRole,
  imagePart,
  audioPart,
  documentPart,
  type Message,
  type LlmRequest,
  type Tool,
  type OutputFormat,
  ToolChoice,
  ReasoningEffort,
} from '../index'

const __dirname = dirname(fileURLToPath(import.meta.url))
const FILES_DIR = resolve(__dirname, '../../llms-sdk/files')

/* ------------------------------------------------------------------ */
/*  Factory helpers                                                    */
/* ------------------------------------------------------------------ */

function textMsg(text: string): Message {
  return {
    role: MessageRole.User,
    content: [{ type: 'Text', field0: { text } }],
  }
}

function openaiRequest(model: string, messages: Message[]): LlmRequest {
  return {
    apiType: 'openai' as ApiType,
    baseUrl: 'https://api.openai.com/v1',
    apiKey: process.env.OPENAI_API_KEY ?? '',
    model,
    messages,
    maxOutputTokens: 256,
    temperature: 0.7,
    stream: false,
    parallelToolCalls: false,
  }
}

function anthropicRequest(model: string, messages: Message[]): LlmRequest {
  return {
    apiType: 'anthropic' as ApiType,
    baseUrl: 'https://api.anthropic.com/v1',
    apiKey: process.env.ANTHROPIC_API_KEY ?? '',
    model,
    messages,
    maxOutputTokens: 256,
    reasoningEffort: 'medium' as ReasoningEffort,
    stream: false,
    parallelToolCalls: false,
  }
}

function shouldRunIntegration(): boolean {
  return process.env.RUN_INTEGRATION_TESTS?.toLowerCase() === 'true'
}

/* ------------------------------------------------------------------ */
/*  Unit tests – factory methods                                       */
/* ------------------------------------------------------------------ */

test('imagePart from file path', (t) => {
  const part = imagePart(resolve(FILES_DIR, 'cat.jpeg'))
  t.is(part.isBase64, true)
  t.is(part.mimeType, 'image/jpeg')
  t.true(part.data.length > 0)
})

test('imagePart from Buffer', (t) => {
  const buf = readFileSync(resolve(FILES_DIR, 'cat.jpeg'))
  const part = imagePart(buf)
  t.is(part.isBase64, true)
  t.is(part.mimeType, 'image/jpeg')
  t.true(part.data.length > 0)
})

test('imagePart from URL string', (t) => {
  const url = 'https://example.com/cat.jpeg'
  const part = imagePart(url)
  t.is(part.isBase64, false)
  t.is(part.mimeType, undefined)
  t.is(part.data, url)
})

test('audioPart from file path', (t) => {
  const part = audioPart(resolve(FILES_DIR, 'audio.wav'))
  t.is(part.mimeType, 'audio/vnd.wave')
  t.true(part.data.length > 0)
})

test('audioPart from Buffer', (t) => {
  const buf = readFileSync(resolve(FILES_DIR, 'audio.mp3'))
  const part = audioPart(buf)
  t.is(part.mimeType, 'audio/mpeg')
  t.true(part.data.length > 0)
})

test('documentPart from PDF file path', (t) => {
  const part = documentPart(resolve(FILES_DIR, 'file.pdf'))
  t.is(part.isBase64, true)
  t.is(part.mimeType, 'application/pdf')
  t.true(part.data.length > 0)
})

test('documentPart from PDF Buffer', (t) => {
  const buf = readFileSync(resolve(FILES_DIR, 'file.pdf'))
  const part = documentPart(buf)
  t.is(part.isBase64, true)
  t.is(part.mimeType, 'application/pdf')
  t.true(part.data.length > 0)
})

test('documentPart from URL string', (t) => {
  const url = 'https://example.com/file.pdf'
  const part = documentPart(url)
  t.is(part.isBase64, false)
  t.is(part.mimeType, undefined)
  t.is(part.data, url)
})

/* ------------------------------------------------------------------ */
/*  Integration tests – OpenAI                                         */
/* ------------------------------------------------------------------ */

const OPENAI_MODEL = 'gpt-5.4-mini'

test('OpenAI – basic text completion', async (t) => {
  if (!shouldRunIntegration() || !process.env.OPENAI_API_KEY) {
    t.pass('skipped')
    return
  }
  const req = openaiRequest(OPENAI_MODEL, [textMsg("Say 'hello world' exactly.")])
  const llm = new Llm()
  const resp = await llm.respond(req)
  t.true(resp.message.content.length > 0)
})

test('OpenAI – image input', async (t) => {
  if (!shouldRunIntegration() || !process.env.OPENAI_API_KEY) {
    t.pass('skipped')
    return
  }
  const img = imagePart(resolve(FILES_DIR, 'cat.jpeg'))
  const msg: Message = {
    role: MessageRole.User,
    content: [
      { type: 'Text', field0: { text: 'Describe this image briefly.' } },
      { type: 'Image', field0: img },
    ],
  }
  const req = openaiRequest(OPENAI_MODEL, [msg])
  const llm = new Llm()
  const resp = await llm.respond(req)
  t.true(resp.message.content.length > 0)
})

test('OpenAI – audio input', async (t) => {
  if (!shouldRunIntegration() || !process.env.OPENAI_API_KEY) {
    t.pass('skipped')
    return
  }
  const aud = audioPart(resolve(FILES_DIR, 'audio.wav'))
  const msg: Message = {
    role: MessageRole.User,
    content: [
      { type: 'Text', field0: { text: 'Describe this audio briefly.' } },
      { type: 'Audio', field0: aud },
    ],
  }
  const req = openaiRequest('gpt-audio-1.5', [msg])
  const llm = new Llm()
  const resp = await llm.respond(req)
  t.true(resp.message.content.length > 0)
})

test('OpenAI – structured output', async (t) => {
  if (!shouldRunIntegration() || !process.env.OPENAI_API_KEY) {
    t.pass('skipped')
    return
  }
  const outputFormat: OutputFormat = {
    name: 'capital',
    description: 'Country capital',
    schema: {
      type: 'object',
      properties: {
        country: { type: 'string' },
        capital: { type: 'string' },
      },
      required: ['country', 'capital'],
    },
  }
  const req: LlmRequest = {
    ...openaiRequest(OPENAI_MODEL, [textMsg('What is the capital of France?')]),
    outputFormat,
  }
  const llm = new Llm()
  const resp = await llm.respond(req)
  const textPart = resp.message.content.find((c) => c.type === 'Text')
  t.truthy(textPart)
  const parsed = JSON.parse((textPart as any).field0.text)
  t.is(parsed.country.toLowerCase(), 'france')
  t.true(parsed.capital.length > 0)
})

test('OpenAI – tool use', async (t) => {
  if (!shouldRunIntegration() || !process.env.OPENAI_API_KEY) {
    t.pass('skipped')
    return
  }
  const tool: Tool = {
    name: 'get_weather',
    description: 'Return weather for a city. Only use this tool.',
    parameters: {
      type: 'object',
      properties: {
        city: { type: 'string' },
      },
      required: ['city'],
    },
  }
  const req: LlmRequest = {
    ...openaiRequest(OPENAI_MODEL, [textMsg('Call the get_weather tool to tell me what is the weather in Paris')]),
    tools: [tool],
  }
  const llm = new Llm()
  const resp = await llm.respond(req)
  const hasTool = resp.message.content.some((p) => p.type === 'ToolCall')
  t.true(hasTool)
})

test('OpenAI - streaming text', async (t) => {
  if (!shouldRunIntegration() || !process.env.OPENAI_API_KEY) {
    t.pass('skipped')
    return
  }
  const req: LlmRequest = {
    ...openaiRequest(OPENAI_MODEL, [textMsg('Count to three.')]),
    stream: true,
  }
  const llm = new Llm()
  const deltas: string[] = []
  let complete: any = null

  await new Promise<void>((resolve, reject) => {
    llm.streamResponse(req, (err, chunk) => {
      if (err) {
        reject(err)
        return
      }
      if (!chunk) return
      if (chunk.type === 'Delta') {
        if (chunk.field0.delta) deltas.push(chunk.field0.delta)
      } else if (chunk.type === 'Complete') {
        complete = chunk.field0
        resolve()
      }
    })
  })

  t.true(deltas.length > 0)
  t.truthy(complete)
  t.truthy(complete!.message)
})

/* ------------------------------------------------------------------ */
/*  Integration tests – Anthropic                                      */
/* ------------------------------------------------------------------ */

const ANTHROPIC_MODEL = 'claude-sonnet-5'

test('Anthropic – basic text completion', async (t) => {
  if (!shouldRunIntegration() || !process.env.ANTHROPIC_API_KEY) {
    t.pass('skipped')
    return
  }
  const req = anthropicRequest(ANTHROPIC_MODEL, [textMsg("Say 'hello world' exactly.")])
  const llm = new Llm()
  const resp = await llm.respond(req)
  t.true(resp.message.content.length > 0)
})

test('Anthropic – image input', async (t) => {
  if (!shouldRunIntegration() || !process.env.ANTHROPIC_API_KEY) {
    t.pass('skipped')
    return
  }
  const img = imagePart(resolve(FILES_DIR, 'cat.jpeg'))
  const msg: Message = {
    role: MessageRole.User,
    content: [
      { type: 'Text', field0: { text: 'Describe this image briefly.' } },
      { type: 'Image', field0: img },
    ],
  }
  const req = anthropicRequest(ANTHROPIC_MODEL, [msg])
  const llm = new Llm()
  const resp = await llm.respond(req)
  t.true(resp.message.content.length > 0)
})

test('Anthropic – document input', async (t) => {
  if (!shouldRunIntegration() || !process.env.ANTHROPIC_API_KEY) {
    t.pass('skipped')
    return
  }
  const doc = documentPart(resolve(FILES_DIR, 'file.pdf'))
  const msg: Message = {
    role: MessageRole.User,
    content: [
      { type: 'Text', field0: { text: 'Summarize this document briefly.' } },
      { type: 'Document', field0: doc },
    ],
  }
  const req = anthropicRequest(ANTHROPIC_MODEL, [msg])
  const llm = new Llm()
  const resp = await llm.respond(req)
  t.true(resp.message.content.length > 0)
})

test('Anthropic – structured output', async (t) => {
  if (!shouldRunIntegration() || !process.env.ANTHROPIC_API_KEY) {
    t.pass('skipped')
    return
  }
  const outputFormat: OutputFormat = {
    name: 'capital',
    description: 'Country capital',
    schema: {
      type: 'object',
      properties: {
        country: { type: 'string' },
        capital: { type: 'string' },
      },
      required: ['country', 'capital'],
    },
  }
  const req: LlmRequest = {
    ...anthropicRequest(ANTHROPIC_MODEL, [textMsg('What is the capital of France?')]),
    outputFormat,
  }
  const llm = new Llm()
  const resp = await llm.respond(req)
  const textPart = resp.message.content.find((c) => c.type === 'Text')
  t.truthy(textPart)
  const parsed = JSON.parse((textPart as any).field0.text)
  t.is(parsed.country.toLowerCase(), 'france')
  t.true(parsed.capital.length > 0)
})

test('Anthropic – tool use', async (t) => {
  if (!shouldRunIntegration() || !process.env.ANTHROPIC_API_KEY) {
    t.pass('skipped')
    return
  }
  const tool: Tool = {
    name: 'get_weather',
    description: 'Return weather for a city. Only use this tool.',
    parameters: {
      type: 'object',
      properties: {
        city: { type: 'string' },
      },
      required: ['city'],
    },
  }
  const req: LlmRequest = {
    ...anthropicRequest(ANTHROPIC_MODEL, [
      textMsg('Call the get_weather tool to tell me what is the weather in Paris'),
    ]),
    tools: [tool],
    toolChoice: 'required' as ToolChoice,
  }
  const llm = new Llm()
  const resp = await llm.respond(req)
  const hasTool = resp.message.content.some((p) => p.type === 'ToolCall')
  t.true(hasTool)
})

test('Anthropic - streaming text', async (t) => {
  if (!shouldRunIntegration() || !process.env.ANTHROPIC_API_KEY) {
    t.pass('skipped')
    return
  }
  const req: LlmRequest = {
    ...anthropicRequest(ANTHROPIC_MODEL, [textMsg('Count to three.')]),
    stream: true,
  }
  const llm = new Llm()
  const deltas: string[] = []
  let complete: any = null

  await new Promise<void>((resolve, reject) => {
    llm.streamResponse(req, (err, chunk) => {
      if (err) {
        reject(err)
        return
      }
      if (!chunk) return
      if (chunk.type === 'Delta') {
        if (chunk.field0.delta) deltas.push(chunk.field0.delta)
      } else if (chunk.type === 'Complete') {
        complete = chunk.field0
        resolve()
      }
    })
  })

  t.true(deltas.length > 0)
  t.truthy(complete)
  t.truthy(complete!.message)
})
