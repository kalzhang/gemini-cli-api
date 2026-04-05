# gemini-cli-api

A fork of [Gemini CLI](https://github.com/google-gemini/gemini-cli) that adds a
`gemini serve` subcommand, exposing an OpenAI-compatible API endpoint on
localhost. Designed to connect any OpenAI-compatible client to
Gemini's Code Assist backend using your personal Google account OAuth — the same
auth path the CLI itself uses.

## Why

The standard Gemini API requires an API key. This fork routes requests through
the CLI instead. It reuses your existing Gemini CLI OAuth tokens. No setup or login required if you're already authenticated. Plug and play.

## Requirements

- Node.js 20+
- A Google account already authenticated with Gemini CLI (`~/.gemini/oauth_creds.json` must exist)
- OpenAI-compatible client

## Setup
```bash
git clone https://github.com/kalzhang/gemini-cli-api.git
cd gemini-cli-api
npm install
cd packages/cli
npm install express
npm install --save-dev @types/express
cd ../..
npm run build
```

## Running

From `packages/cli/`:
```bash

# Batch script
serve.bat

# Or directly
npm start -- serve
npm start -- serve --port 9090
```

## Connecting to an OpenAI endpoint

| Setting | Value |
|---|---|
| API type | Chat Completion |
| Source | Custom (OpenAI-compatible) |
| Base URL | `http://127.0.0.1:8888/v1` |
| API key | anything (not validated) |
| Model | your model here. e.g.,`gemini-3.1-pro-preview` |

## Customizable values

All in `packages/cli/src/serveCommand.ts` unless noted.

| Constant | Default | Description |
|---|---|---|
| `DEFAULT_PORT` | `8888` | Port when `--port` is not passed. Must match the yargs default in `config.ts`. |
| `DEFAULT_MODEL` | `gemini-3.1-pro-preview` | Model used when the client does not specify one. |
| `MAX_RETRIES` | `5` | Number of retry attempts on 429 / 500 / 503 errors. |
| `RETRYABLE_CODES` | `[429, 500, 503]` | HTTP status codes that trigger a retry. |
| `retryDelay()` | 100–1000ms random | Delay between retries. Edit the function to change the range. |
| `SAFETY_SETTINGS` | all `BLOCK_NONE` | Gemini safety filter thresholds. |
| `limit` in `express.json()` | `200mb` | Max request body size. |
| `/v1/models` response | 2.5-pro, 2.5-flash, 3.1-pro, 3.0-flash | Model list returned to the client. Add entries here for other models. |

## How it works

`gemini serve` initializes auth exactly as the normal CLI does, with the same OAuth
client, same token refresh path, and same User-Agent. It then starts an Express
server. Each incoming request is converted from OpenAI format to Gemini format
and sent directly to `ContentGenerator.generateContentStream()`, bypassing the
CLI's session and history machinery entirely. Responses are streamed back as
OpenAI SSE chunks.

The conversion handles:
- Multiple system messages → concatenated into `systemInstruction`
- `assistant` role → `model` role
- Consecutive same-role turns → single-space placeholder inserted
- OpenAI's array-format `content` field → flattened to text
- Multimedia formats (e.g., images) are correctly injected

## Known limitations
- The `gemini serve` command bypasses GEMINI.md loading, tool registration, and
  extension loading — the server is stateless and has no use for them.
- All logging goes to stderr via `writeSync` to bypass the CLI's console
  patcher, which intercepts normal stdout/stderr in serve mode.

## Files changed from upstream

| File | Change |
|---|---|
| `packages/cli/src/serveCommand.ts` | New file — Express server |
| `packages/cli/src/serveConverter.ts` | New file — OpenAI → Gemini message converter |
| `packages/cli/src/gemini.tsx` | Added serve dispatch and sandbox bypass |
| `packages/cli/src/config/config.ts` | Added `serve` subcommand and `--port` option |
| `packages/cli/package.json` | Added `express` and `@types/express` |