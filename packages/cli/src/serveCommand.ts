// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import crypto from 'crypto';
import { writeSync } from 'fs';
import express from 'express';
import type { Request, Response } from 'express';
import { Config, LlmRole } from '@google/gemini-cli-core';
import { HarmCategory, HarmBlockThreshold } from '@google/genai';
import { convertMessages, type OpenAIContent } from './serveConverter.js';

// Must match the default in packages/cli/src/config/config.ts .option('port')
const DEFAULT_PORT = 8888;
const DEFAULT_MODEL = 'gemini-3.1-pro-preview';
const MAX_RETRIES = 5;
const RETRYABLE_CODES = [429, 500, 503];

// ─── Model ID parsing ─────────────────────────────────────────────────────────
//
// SillyTavern's effort selector does not map to any field in its OpenAI-compatible
// request body, so thinking level is encoded directly in the model name instead.
// The colon is the separator: 'gemini-2.5-pro:thinking-high'
//   → baseModel = 'gemini-2.5-pro'   (sent to the Gemini API)
//   → thinkingOverride = 'high'       (controls thinkingLevel / thinkingBudget)
//
// Valid suffixes: :thinking-minimum | :thinking-low | :thinking-medium |
//                 :thinking-high    | :thinking-maximum
//
// The base model name (no suffix) uses the model's default dynamic thinking.
// The override takes priority over any reasoning_effort field in the request.

const THINKING_SUFFIX_RE = /^(.+):thinking-(minimum|low|medium|high|maximum)$/;

function parseModelId(id: string): { baseModel: string; thinkingOverride: string | undefined } {
  const match = id.match(THINKING_SUFFIX_RE);
  if (match) return { baseModel: match[1], thinkingOverride: match[2] };
  return { baseModel: id, thinkingOverride: undefined };
}

// ─── Types ────────────────────────────────────────────────────────────────────

export interface OpenAIMessage {
  role: 'system' | 'user' | 'assistant';
  content: OpenAIContent;
}

interface OpenAIRequest {
  messages: OpenAIMessage[];
  model?: string;
  stream?: boolean;
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  stop?: string | string[];
  // Standard OpenAI reasoning effort field: 'auto'|'minimum'|'low'|'medium'|'high'|'maximum'|'none'|'off'
  reasoning_effort?: string;
  // SillyTavern's own reasoning flag
  reasoning?: { enabled?: boolean };
}

// ─── Safety settings ─────────────────────────────────────────────────────────

const SAFETY_SETTINGS = [
  { category: HarmCategory.HARM_CATEGORY_HARASSMENT,        threshold: HarmBlockThreshold.BLOCK_NONE },
  { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,       threshold: HarmBlockThreshold.BLOCK_NONE },
  { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
  { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE },
  { category: HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,   threshold: HarmBlockThreshold.BLOCK_NONE },
];

// ─── Logging ──────────────────────────────────────────────────────────────────

function log(msg: string): void {
  writeSync(2, `[serve] ${msg}\n`);
}

// ─── Error handling ───────────────────────────────────────────────────────────

function extractHttpCode(err: unknown): number | null {
  const match = String(err).match(/"code":\s*(\d+)/);
  return match ? parseInt(match[1], 10) : null;
}

function simplifyError(err: unknown): string {
  const msg = String(err);
  try {
    const jsonMatch = msg.match(/\[\s*(\{[\s\S]*\})\s*\]/);
    if (jsonMatch) {
      const parsed = JSON.parse(jsonMatch[1]) as {
        error?: { code?: number; status?: string; message?: string };
      };
      const e = parsed?.error;
      if (e) {
        return `${e.code ?? '?'} ${e.status ?? ''}: ${e.message ?? 'unknown error'}`.trim();
      }
    }
  } catch {
    // fall through
  }
  return msg.length > 300 ? msg.slice(0, 300) + '…' : msg;
}

function isRetryable(err: unknown): boolean {
  const code = extractHttpCode(err);
  if (code !== null && RETRYABLE_CODES.includes(code)) return true;
  const msg = String(err);
  return (
    msg.includes('RESOURCE_EXHAUSTED') ||
    msg.includes('UNAVAILABLE') ||
    msg.includes('INTERNAL')
  );
}

// Flat random delay between 100ms and 1000ms — no growth between attempts.
function retryDelay(): Promise<void> {
  const ms = Math.round(100 + Math.random() * 900);
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ─── Response helpers ─────────────────────────────────────────────────────────

function mapFinishReason(geminiReason?: string): string {
  switch (geminiReason) {
    case 'STOP':       return 'stop';
    case 'MAX_TOKENS': return 'length';
    case 'SAFETY':
    case 'RECITATION': return 'content_filter';
    default:           return 'stop';
  }
}

function isSafetyBlock(reason?: string): boolean {
  return reason === 'SAFETY' || reason === 'RECITATION';
}

// Extracts text and thought text separately from a response chunk.
// Parts with part.thought === true are reasoning output; all others are answer content.
function extractParts(chunk: any): { text: string; thoughtText: string } {
  const parts: any[] = chunk.candidates?.[0]?.content?.parts ?? [];
  let text = '';
  let thoughtText = '';
  for (const p of parts) {
    if (p.text == null) continue;
    if (p.thought) thoughtText += p.text;
    else text += p.text;
  }
  return { text, thoughtText };
}

// Attempts generateContentStream with retries on retryable errors.
// Throws the last error if all attempts are exhausted.
async function generateWithRetry(
  contentGenerator: any,
  request: any,
  userPromptId: string,
): Promise<AsyncGenerator<any>> {
  let lastErr: unknown;
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      return await contentGenerator.generateContentStream(
        request,
        userPromptId,
        LlmRole.MAIN,
      );
    } catch (err) {
      lastErr = err;
      if (!isRetryable(err) || attempt === MAX_RETRIES) break;
      log(`retry ${attempt + 1}/${MAX_RETRIES} (${extractHttpCode(err) ?? 'err'}) waiting...`);
      await retryDelay();
    }
  }
  throw lastErr;
}

// ─── Entry point ──────────────────────────────────────────────────────────────

export async function runServeCommand(
  config: Config,
  port: number = DEFAULT_PORT,
): Promise<void> {
  const contentGenerator = config.getContentGenerator();

  const app = express();
  app.use(express.json({ limit: '200mb' }));

  // ─── POST /v1/chat/completions ────────────────────────────────────────────

  app.post('/v1/chat/completions', async (req: Request, res: Response) => {
    const {
      messages,
      model,
      stream = true,
      temperature,
      max_tokens,
      top_p,
      stop,
      reasoning_effort,
      reasoning,
    } = req.body as OpenAIRequest;

    // targetModel is echoed back to SillyTavern in responses unchanged.
    // baseModel is the name sent to the Gemini API (suffix stripped).
    // thinkingOverride comes from the model suffix and takes priority over
    // reasoning_effort, which SillyTavern does not actually send.
    const targetModel = model ?? DEFAULT_MODEL;
    const { baseModel, thinkingOverride } = parseModelId(targetModel);
    const responseId  = `chatcmpl-${crypto.randomUUID()}`;
    const created     = Math.floor(Date.now() / 1000);

    // Reasoning is active when:
    //   - the model name carries a :thinking-<level> suffix, OR
    //   - SillyTavern's reasoning.enabled flag is set, OR
    //   - reasoning_effort is present and not an explicit disable value.
    const wantsReasoning =
      thinkingOverride !== undefined ||
      reasoning?.enabled === true ||
      (reasoning_effort !== undefined &&
        reasoning_effort !== 'none' &&
        reasoning_effort !== 'off');

    // Model suffix takes priority; fall back to reasoning_effort; then 'auto'.
    const thinkingLevel = wantsReasoning
      ? (thinkingOverride ?? reasoning_effort ?? 'auto')
      : undefined;

    log(
      `request  model=${baseModel} stream=${stream} messages=${messages.length}` +
      (wantsReasoning ? ` thinking=${thinkingLevel}` : ''),
    );

    const stopSequences = stop
      ? Array.isArray(stop) ? stop : [stop]
      : undefined;

    const request = convertMessages(messages, baseModel, {
      safetySettings:  SAFETY_SETTINGS,
      temperature,
      maxOutputTokens: max_tokens,
      topP:            top_p,
      stopSequences,
      thinkingLevel,
      includeThoughts: wantsReasoning,
    });

    // isReasoning=true emits { reasoning_content } delta instead of { content }.
    // When finishReason is not null, delta is always {} regardless of isReasoning.
    const makeStreamChunk = (
      content: string,
      finishReason: string | null = null,
      isReasoning = false,
    ) => ({
      id:      responseId,
      object:  'chat.completion.chunk',
      created,
      model:   targetModel,
      choices: [{
        index:         0,
        delta:         finishReason !== null
                         ? {}
                         : isReasoning
                           ? { reasoning_content: content }
                           : { content },
        finish_reason: finishReason,
      }],
    });

    // ── Streaming path ──────────────────────────────────────────────────────
    // All retries happen before SSE headers are sent. Once headers are
    // committed, mid-stream errors terminate the stream with an error chunk.

    if (stream) {
      let generator: AsyncGenerator<any>;
      try {
        generator = await generateWithRetry(
          contentGenerator,
          request,
          crypto.randomUUID(),
        );
      } catch (err) {
        const simple = simplifyError(err);
        log(`error ${simple}`);
        res.status(500).json({ error: { message: simple, type: 'api_error', code: 500 } });
        return;
      }

      res.setHeader('Content-Type',  'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection',    'keep-alive');

      try {
        let lastFinishReason: string | undefined;

        for await (const chunk of generator) {
          const candidate    = chunk.candidates?.[0];
          const finishReason = candidate?.finishReason as string | undefined;

          if (isSafetyBlock(finishReason)) {
            log(`safety block (${finishReason})`);
            res.write(`data: ${JSON.stringify(makeStreamChunk('', mapFinishReason(finishReason)))}\n\n`);
            res.write('data: [DONE]\n\n');
            res.end();
            return;
          }

          const { text, thoughtText } = extractParts(chunk);
          if (thoughtText) {
            res.write(`data: ${JSON.stringify(makeStreamChunk(thoughtText, null, true))}\n\n`);
          }
          if (text) {
            res.write(`data: ${JSON.stringify(makeStreamChunk(text))}\n\n`);
          }

          if (finishReason) lastFinishReason = finishReason;
        }

        log(`response finish_reason=${mapFinishReason(lastFinishReason)}`);
        res.write(`data: ${JSON.stringify(makeStreamChunk('', mapFinishReason(lastFinishReason)))}\n\n`);
        res.write('data: [DONE]\n\n');
        res.end();
      } catch (err) {
        const simple = simplifyError(err);
        log(`stream error: ${simple}`);
        res.write(`data: ${JSON.stringify({ error: { message: simple } })}\n\n`);
        res.write('data: [DONE]\n\n');
        res.end();
      }

    // ── Non-streaming path ──────────────────────────────────────────────────

    } else {
      try {
        const generator = await generateWithRetry(
          contentGenerator,
          request,
          crypto.randomUUID(),
        );

        let fullText         = '';
        let fullThoughtText  = '';
        let lastFinishReason: string | undefined;

        for await (const chunk of generator) {
          const candidate    = chunk.candidates?.[0];
          const finishReason = candidate?.finishReason as string | undefined;

          if (isSafetyBlock(finishReason)) {
            log(`safety block (${finishReason})`);
            res.json({
              id: responseId, object: 'chat.completion', created, model: targetModel,
              choices: [{ index: 0, message: { role: 'assistant', content: '' }, finish_reason: mapFinishReason(finishReason) }],
            });
            return;
          }

          const { text, thoughtText } = extractParts(chunk);
          fullText        += text;
          fullThoughtText += thoughtText;
          if (finishReason) lastFinishReason = finishReason;
        }

        log(`response finish_reason=${mapFinishReason(lastFinishReason)} chars=${fullText.length}`);
        res.json({
          id:      responseId,
          object:  'chat.completion',
          created,
          model:   targetModel,
          choices: [{
            index:         0,
            message: {
              role:    'assistant',
              content: fullText,
              // Only include reasoning_content when there is actual thought output.
              ...(fullThoughtText && { reasoning_content: fullThoughtText }),
            },
            finish_reason: mapFinishReason(lastFinishReason),
          }],
        });
      } catch (err) {
        const simple = simplifyError(err);
        log(`error ${simple}`);
        res.status(500).json({ error: { message: simple, type: 'api_error', code: 500 } });
      }
    }
  });

  // ─── GET /v1/models ──────────────────────────────────────────────────────────
  // Base model entries use the model's default dynamic thinking.
  // Suffixed entries encode the thinking level directly in the model ID so
  // SillyTavern users can control reasoning depth via the model selector.

  app.get('/v1/models', (_req: Request, res: Response) => {
    const ts = 1677610602;
    const makeEntry = (id: string) => ({ id, object: 'model', created: ts, owned_by: 'google' });
    const withLevels = (base: string) => [
      makeEntry(base),
// Removed thinking levels from drop down menu. Add back in if necessary.
//      makeEntry(`${base}:thinking-minimum`),
//      makeEntry(`${base}:thinking-low`),
//      makeEntry(`${base}:thinking-medium`),
//      makeEntry(`${base}:thinking-high`),
      makeEntry(`${base}:thinking-maximum`),
    ];
    res.json({
      object: 'list',
      data: [
        ...withLevels('gemini-2.5-flash'),
        ...withLevels('gemini-2.5-pro'),
        ...withLevels('gemini-3-flash-preview'),
        ...withLevels('gemini-3.1-pro-preview'),
      ],
    });
  });

  // ─── Start ────────────────────────────────────────────────────────────────

  await new Promise<void>((resolve) => {
    app.listen(port, () => {
      writeSync(2, `Serve mode active\n`);
      writeSync(2, `Connect SillyTavern to: http://127.0.0.1:${port}/v1\n`);
      resolve();
    });
  });

  await new Promise(() => {});
}