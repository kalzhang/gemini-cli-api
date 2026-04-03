# gemini-cli-api

A fork of [Gemini CLI](https://github.com/google-gemini/gemini-cli) that adds a
`gemini serve` subcommand, exposing an OpenAI-compatible API endpoint on
localhost. Designed to connect SillyTavern (or any OpenAI-compatible client) to
Gemini's Code Assist backend using your personal Google account OAuth — the same
auth path the CLI itself uses.

## Why

The standard Gemini API requires an API key. This fork routes requests through
the CLI's OAuth identity instead, giving you access to the free Code Assist
quota (60 req/min, 1000 req/day) without managing API keys.

## Requirements

- Node.js 20+
- A Google account already authenticated with Gemini CLI (`~/.gemini/oauth_creds.json` must exist)
- SillyTavern (or any OpenAI-compatible client)

## Setup
```bash
git clone https://github.com/YOUR_USERNAME/gemini-cli-api.git
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
# Build and serve
build-and-serve.bat

# Serve only (no rebuild)
serve.bat

# Or directly
npm start -- serve
npm start -- serve --port 9090
```

## Connecting SillyTavern

In SillyTavern's API settings:

| Setting | Value |
|---|---|
| API type | Chat Completion |
| Source | Custom (OpenAI-compatible) |
| Base URL | `http://127.0.0.1:8888/v1` |
| API key | anything (not validated) |
| Model | `gemini-2.5-pro` or `gemini-2.5-flash` |

## Customizable values

All in `packages/cli/src/serveCommand.ts` unless noted.

| Constant | Default | Description |
|---|---|---|
| `DEFAULT_PORT` | `8888` | Port when `--port` is not passed. Must match the yargs default in `config.ts`. |
| `DEFAULT_MODEL` | `gemini-2.5-pro` | Model used when the client does not specify one. |
| `MAX_RETRIES` | `3` | Number of retry attempts on 429 / 500 / 503 errors. |
| `RETRYABLE_CODES` | `[429, 500, 503]` | HTTP status codes that trigger a retry. |
| `retryDelay()` | 100–1000ms random | Delay between retries. Edit the function to change the range. |
| `SAFETY_SETTINGS` | all `BLOCK_NONE` | Gemini safety filter thresholds. |
| `limit` in `express.json()` | `100mb` | Max request body size. |
| `/v1/models` response | 2.5-pro, 2.5-flash | Model list returned to the client. Add entries here for other models. |

## How it works

`gemini serve` initializes auth exactly as the normal CLI does — same OAuth
client, same token refresh path, same User-Agent — then starts an Express
server. Each incoming request is converted from OpenAI format to Gemini format
and sent directly to `ContentGenerator.generateContentStream()`, bypassing the
CLI's session and history machinery entirely. Responses are streamed back as
OpenAI SSE chunks.

The conversion handles:
- Multiple system messages → concatenated into `systemInstruction`
- `assistant` role → `model` role
- Consecutive same-role turns → single-space placeholder inserted
- SillyTavern's array-format `content` field → flattened to text

## Known limitations

- Multimodal (image) input is not yet implemented. Image parts in messages are
  silently dropped.
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