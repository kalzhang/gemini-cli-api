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

import type {
  GenerateContentParameters,
  GenerateContentConfig,
  Content,
  SafetySetting,
} from '@google/genai';
import type { OpenAIMessage } from './serveCommand.js';

// ---------- Types ----------

// Exported so serveCommand.ts can use it in OpenAIMessage and OpenAIRequest,
// keeping the type definition in one place.
export type OpenAIContent =
  | string
  | Array<{ type: string; text?: string; [key: string]: unknown }>;

export interface ConvertConfig {
  safetySettings?: SafetySetting[];
  temperature?: number;
  maxOutputTokens?: number;
  topP?: number;
  stopSequences?: string[];
}

// ---------- Internal helpers ----------

// Normalises SillyTavern's content field to a plain string.
// SillyTavern can send content as a string or as an array of typed parts even
// when multimodal support is not configured. Array content is flattened to
// text parts only — non-text parts are dropped until multimodal is added.
// This prevents [object Object] reaching the model if the type guard is ever
// bypassed at the call site.
function contentToString(content: OpenAIContent): string {
  if (typeof content === 'string') return content;
  return content
    .filter(
      (p): p is { type: 'text'; text: string } =>
        p.type === 'text' && typeof p.text === 'string',
    )
    .map((p) => p.text)
    .join('');
}

function toGeminiRole(role: 'user' | 'assistant'): 'user' | 'model' {
  return role === 'assistant' ? 'model' : 'user';
}

// Gemini requires strictly alternating user/model turns.
// Consecutive turns of the same role get a single-space placeholder inserted
// between them. A bare empty string risks backend rejection on zero-length
// part validation; a single space is safe and invisible to the model.
function normalizeTurns(turns: Content[]): Content[] {
  if (turns.length === 0) return turns;

  const normalized: Content[] = [];

  for (const turn of turns) {
    const prev = normalized[normalized.length - 1];
    if (prev && prev.role === turn.role) {
      const placeholderRole = turn.role === 'user' ? 'model' : 'user';
      normalized.push({ role: placeholderRole, parts: [{ text: ' ' }] });
    }
    normalized.push(turn);
  }

  // Gemini rejects conversations that open with a 'model' turn.
  // Shouldn't happen with SillyTavern but guard defensively.
  if (normalized[0]?.role === 'model') {
    normalized.unshift({ role: 'user', parts: [{ text: ' ' }] });
  }

  return normalized;
}

// ---------- Main export ----------

export function convertMessages(
  messages: OpenAIMessage[],
  model: string,
  overrides: ConvertConfig = {},
): GenerateContentParameters {
  const systemParts: string[] = [];
  const conversationTurns: Content[] = [];

  for (const msg of messages) {
    const text = contentToString(msg.content);

    if (msg.role === 'system') {
      // Trim to prevent whitespace accumulating between the multiple system
      // blocks SillyTavern emits (persona, scenario, main prompt, etc.).
      const trimmed = text.trim();
      if (trimmed) systemParts.push(trimmed);
      continue;
    }

    // Skip turns with empty or whitespace-only content. An empty text part
    // risks backend rejection. Structural placeholders are handled by
    // normalizeTurns; we don't need genuinely empty content turns.
    const trimmed = text.trim();
    if (!trimmed) continue;

    conversationTurns.push({
      role: toGeminiRole(msg.role),
      parts: [{ text: trimmed }],
    });
  }

  const contents = normalizeTurns(conversationTurns);

  // Edge case: all messages were system messages, or all conversation turns
  // were empty. Insert a minimal user turn so the request is structurally
  // valid — the model will respond to systemInstruction alone.
  if (contents.length === 0) {
    contents.push({ role: 'user', parts: [{ text: ' ' }] });
  }

  // systemInstruction is passed as a pre-formed Content object with an
  // explicit role so toContent() in the CLI's converter.ts preserves it
  // rather than defaulting to 'user'. The Firebase AI Logic SDK confirms
  // role: 'system' is accepted. The backend identifies systemInstruction by
  // its position in the request body, not the role value.
  const systemInstruction: Content | undefined =
    systemParts.length > 0
      ? { role: 'system', parts: [{ text: systemParts.join('\n\n') }] }
      : undefined;

  // Build config as GenerateContentConfig directly so TypeScript enforces
  // field names — a typo like maxOutputToken (missing 's') won't compile.
  // Conditional spread omits undefined fields without losing the type.
  const config: GenerateContentConfig = {
    ...(systemInstruction !== undefined   && { systemInstruction }),
    ...(overrides.safetySettings          && { safetySettings:  overrides.safetySettings }),
    ...(overrides.temperature !== undefined && { temperature:     overrides.temperature }),
    ...(overrides.maxOutputTokens !== undefined && { maxOutputTokens: overrides.maxOutputTokens }),
    ...(overrides.topP !== undefined      && { topP:            overrides.topP }),
    ...(overrides.stopSequences           && { stopSequences:   overrides.stopSequences }),
  };

  return { model, contents, config };
}