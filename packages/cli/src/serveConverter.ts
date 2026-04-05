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
import { writeSync } from 'fs';
import type {
  GenerateContentParameters,
  GenerateContentConfig,
  Content,
  Part,
  SafetySetting,
} from '@google/genai';
import type { OpenAIMessage } from './serveCommand.js';

// ---------- Types ----------

// Exported so serveCommand.ts can use it in OpenAIMessage and OpenAIRequest,
// keeping the type definition in one place.
export type OpenAIContentPart =
  | { type: 'text'; text: string; [key: string]: unknown }
  | { type: 'image_url'; image_url: { url: string }; [key: string]: unknown }
  | { type: string; [key: string]: unknown };

export type OpenAIContent = string | OpenAIContentPart[];

export interface ConvertConfig {
  safetySettings?: SafetySetting[];
  temperature?: number;
  maxOutputTokens?: number;
  topP?: number;
  stopSequences?: string[];
  // Thinking level string as received from the client:
  // 'auto' | 'minimum' | 'low' | 'medium' | 'high' | 'maximum'
  // Mapped to thinkingLevel (Gemini 3) or thinkingBudget (Gemini 2.5) by buildThinkingConfig.
  thinkingLevel?: string;
  // When true, thought summaries are included in response parts (part.thought === true).
  includeThoughts?: boolean;
}

// ---------- Internal helpers ----------

// Parses a data URL into its mime type and raw base64 payload.
// Returns null if the URL is not a valid base64 data URL.
function parseDataUrl(url: string): { mimeType: string; data: string } | null {
  const match = url.match(/^data:([^;]+);base64,(.+)$/s);
  if (!match) return null;
  return { mimeType: match[1], data: match[2] };
}

// Converts an OpenAI content field to a Gemini Part array.
//
// Text parts are passed through as-is (trimming is the caller's responsibility
// so that system message accumulation and turn skipping logic can decide).
//
// image_url parts with base64 data URLs are converted to inlineData parts.
// image_url parts with HTTP URLs are not supported and are skipped with a log
// line — fetching arbitrary URLs at request time adds network dependency and
// failure modes that are out of scope.
//
// All other part types are silently dropped (future multimodal types).
function contentToParts(content: OpenAIContent): Part[] {
  if (typeof content === 'string') {
    return content ? [{ text: content }] : [];
  }

  const parts: Part[] = [];
  for (const item of content) {
    if (item.type === 'text') {
      const textItem = item as { type: 'text'; text?: string };
      if (typeof textItem.text === 'string' && textItem.text) {
        parts.push({ text: textItem.text });
      }
    } else if (item.type === 'image_url') {
      const imageItem = item as { type: 'image_url'; image_url?: { url?: string } };
      const url = imageItem.image_url?.url;
      if (!url) {
        writeSync(2, '[serve] skipping image_url part with missing url\n');
        continue;
      }
      if (!url.startsWith('data:')) {
        writeSync(
          2,
          `[serve] skipping HTTP image URL (not supported): ${url.slice(0, 80)}\n`,
        );
        continue;
      }
      const parsed = parseDataUrl(url);
      if (!parsed) {
        writeSync(2, '[serve] skipping malformed base64 data URL\n');
        continue;
      }
      parts.push({ inlineData: { mimeType: parsed.mimeType, data: parsed.data } });
    }
    // All other types are silently dropped.
  }
  return parts;
}

// Returns true if a parts array has any content worth sending to the model.
// A text-only part whose text is blank after trimming does not count.
// An inlineData part always counts regardless of size.
function hasContent(parts: Part[]): boolean {
  return parts.some((p) => {
    if ('inlineData' in p && p.inlineData) return true;
    if ('text' in p && typeof p.text === 'string' && p.text.trim()) return true;
    return false;
  });
}

// Trims whitespace from all text parts in place, then removes any text parts
// that are blank after trimming. inlineData parts are left untouched.
function trimTextParts(parts: Part[]): Part[] {
  return parts
    .map((p) =>
      'text' in p && typeof p.text === 'string'
        ? { text: p.text.trim() }
        : p,
    )
    .filter((p) => !('text' in p) || Boolean(p.text));
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

// Builds a thinkingConfig object for GenerateContentConfig.
//
// Gemini 3 models use thinkingLevel (string enum).
// Gemini 2.5 models use thinkingBudget (integer token count).
//
// Model family is detected from the model string:
//   /^gemini-3/i  → Gemini 3  (thinkingLevel)
//   otherwise     → Gemini 2.5 (thinkingBudget)
//
// Flash detection (/flash/i) controls availability of 'minimal' (Gemini 3)
// and token budget 0 / maximum budget (Gemini 2.5).
//
// Level mapping (Gemini 3):
//   minimum → 'minimal' (Flash) | 'low' (Pro, no minimal support)
//   low     → 'low'
//   medium  → 'medium' (Flash) | 'high' (Pro, medium unsupported)
//   high    → 'high'
//   maximum → 'high'
//   auto    → omitted (model uses default dynamic thinking)
//
// Budget mapping (Gemini 2.5):
//   minimum →    0 (Flash, disables thinking) |  128 (Pro, cannot disable)
//   low     → 1024
//   medium  → 8192
//   high    → 16384
//   maximum → 24576 (Flash) | 32768 (Pro)
//   auto    →   -1 (dynamic)
//
// Returns undefined when neither thinking control nor includeThoughts is needed,
// so the caller can omit the field entirely.
//
// Uses GenerateContentConfig['thinkingConfig'] as the return type to avoid
// importing ThinkingConfig separately, which may not be exported in all SDK
// versions. The shape is validated at compile time through the config spread.
function buildThinkingConfig(
  model: string,
  level: string | undefined,
  includeThoughts: boolean,
): GenerateContentConfig['thinkingConfig'] {
  if (!level && !includeThoughts) return undefined;

  const isGemini3 = /^gemini-3/i.test(model);
  const isFlash   = /flash/i.test(model);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const config: any = {};

  if (includeThoughts) config.includeThoughts = true;

  if (level && level !== 'auto') {
    if (isGemini3) {
      const levelMap: Record<string, string> = {
        minimum: isFlash ? 'minimal' : 'low',   // Pro has no 'minimal'
        low:     'low',
        medium:  isFlash ? 'medium' : 'high',   // Pro has no 'medium'
        high:    'high',
        maximum: 'high',
      };
      config.thinkingLevel = levelMap[level] ?? 'high';
    } else {
      // Gemini 2.5 — token budgets
      const budgetMap: Record<string, number> = {
        minimum: isFlash ? 0 : 128,       // Pro cannot disable thinking (min 128)
        low:     1024,
        medium:  8192,
        high:    16384,
        maximum: isFlash ? 24576 : 32768,
      };
      config.thinkingBudget = budgetMap[level] ?? -1;
    }
  } else if (!isGemini3 && level === 'auto') {
    // For Gemini 2.5, 'auto' means dynamic thinking (-1).
    // For Gemini 3, omitting thinkingLevel already means dynamic, so no action needed.
    config.thinkingBudget = -1;
  }

  return config as GenerateContentConfig['thinkingConfig'];
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
    if (msg.role === 'system') {
      // System messages are text-only. Image parts inside a system message are
      // dropped — systemInstruction is a text field in the Gemini API.
      const parts = contentToParts(msg.content);
      const text = parts
        .filter((p): p is { text: string } => 'text' in p && typeof p.text === 'string')
        .map((p) => p.text)
        .join('');
      const trimmed = text.trim();
      if (trimmed) systemParts.push(trimmed);
      continue;
    }

    const parts = trimTextParts(contentToParts(msg.content));

    // Skip turns with no meaningful content. Empty text parts risk backend
    // rejection. Structural placeholders for alternation are handled by
    // normalizeTurns — we don't manufacture empty turns here.
    if (!hasContent(parts)) continue;

    conversationTurns.push({
      role: toGeminiRole(msg.role),
      parts,
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

  const thinkingConfig = buildThinkingConfig(
    model,
    overrides.thinkingLevel,
    overrides.includeThoughts ?? false,
  );

  // Build config as GenerateContentConfig directly so TypeScript enforces
  // field names — a typo like maxOutputToken (missing 's') won't compile.
  // Conditional spread omits undefined fields without losing the type.
  const config: GenerateContentConfig = {
    ...(systemInstruction !== undefined && { systemInstruction }),
    ...(overrides.safetySettings && { safetySettings: overrides.safetySettings }),
    ...(overrides.temperature !== undefined && { temperature: overrides.temperature }),
    ...(overrides.maxOutputTokens !== undefined && { maxOutputTokens: overrides.maxOutputTokens }),
    ...(overrides.topP !== undefined && { topP: overrides.topP }),
    ...(overrides.stopSequences && { stopSequences: overrides.stopSequences }),
    ...(thinkingConfig !== undefined && { thinkingConfig }),
  };

  return { model, contents, config };
}