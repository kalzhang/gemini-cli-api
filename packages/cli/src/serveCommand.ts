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
const MAX_RETRIES = 3;
const RETRYABLE_CODES = [429, 500, 503];

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

function extractText(chunk: any): string {
  return (
    chunk.candidates?.[0]?.content?.parts
      ?.filter((p: any) => p.text != null)
      .map((p: any) => p.text)
      .join('') ?? ''
  );
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
    } = req.body as OpenAIRequest;

    const targetModel = model ?? DEFAULT_MODEL;
    const responseId  = `chatcmpl-${crypto.randomUUID()}`;
    const created     = Math.floor(Date.now() / 1000);

    log(`request  model=${targetModel} stream=${stream} messages=${messages.length}`);

    const stopSequences = stop
      ? Array.isArray(stop) ? stop : [stop]
      : undefined;

    const request = convertMessages(messages, targetModel, {
      safetySettings:  SAFETY_SETTINGS,
      temperature,
      maxOutputTokens: max_tokens,
      topP:            top_p,
      stopSequences,
    });

    const makeStreamChunk = (content: string, finishReason: string | null = null) => ({
      id:      responseId,
      object:  'chat.completion.chunk',
      created,
      model:   targetModel,
      choices: [{
        index:         0,
        delta:         finishReason !== null ? {} : { content },
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

          const text = extractText(chunk);
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

          fullText += extractText(chunk);
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
            message:       { role: 'assistant', content: fullText },
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

  app.get('/v1/models', (_req: Request, res: Response) => {
    res.json({
      object: 'list',
      data: [
        { id: 'gemini-2.5-flash',   object: 'model', created: 1677610602, owned_by: 'google' },
        { id: 'gemini-2.5-pro', object: 'model', created: 1677610602, owned_by: 'google' },
        { id: 'gemini-3-flash-preview', object: 'model', created: 1677610602, owned_by: 'google' },
        { id: 'gemini-3.1-pro-preview', object: 'model', created: 1677610602, owned_by: 'google' },
      ],
    });
  });


  // ─── Start ────────────────────────────────────────────────────────────────

  await new Promise<void>((resolve) => {
    app.listen(port, () => {
      writeSync(2, `Serve mode active -> listening on http://localhost:${port}\n`);
      writeSync(2, `Connect SillyTavern to: http://localhost:${port}/v1\n`);
      resolve();
    });
  });

  await new Promise(() => {});
}