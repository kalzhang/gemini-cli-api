/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import {
  type StartupWarning,
  WarningPriority,
  type Config,
  type ResumedSessionData,
  type WorktreeInfo,
  type OutputPayload,
  type ConsoleLogPayload,
  type UserFeedbackPayload,
  sessionId,
  logUserPrompt,
  AuthType,
  UserPromptEvent,
  coreEvents,
  CoreEvent,
  getOauthClient,
  patchStdio,
  writeToStdout,
  writeToStderr,
  shouldEnterAlternateScreen,
  startupProfiler,
  ExitCodes,
  SessionStartSource,
  SessionEndReason,
  ValidationCancelledError,
  ValidationRequiredError,
  type AdminControlsSettings,
  debugLogger,
  isHeadlessMode,
} from '@google/gemini-cli-core';
import { writeSync } from 'fs';
import { loadCliConfig, parseArguments } from './config/config.js';
import * as cliConfig from './config/config.js';
import { readStdin } from './utils/readStdin.js';
import { createHash } from 'node:crypto';
import v8 from 'node:v8';
import os from 'node:os';
import dns from 'node:dns';
import { start_sandbox } from './utils/sandbox.js';
import {
  loadSettings,
  SettingScope,
  type DnsResolutionOrder,
  type LoadedSettings,
} from './config/settings.js';
import {
  loadTrustedFolders,
  type TrustedFoldersError,
} from './config/trustedFolders.js';
import { getStartupWarnings } from './utils/startupWarnings.js';
import { getUserStartupWarnings } from './utils/userStartupWarnings.js';
import { ConsolePatcher } from './ui/utils/ConsolePatcher.js';
import { runNonInteractive } from './nonInteractiveCli.js';
import { runServeCommand } from './serveCommand.js';
import {
  cleanupCheckpoints,
  registerCleanup,
  registerSyncCleanup,
  runExitCleanup,
  registerTelemetryConfig,
  setupSignalHandlers,
} from './utils/cleanup.js';
import { setupWorktree } from './utils/worktreeSetup.js';
import {
  cleanupToolOutputFiles,
  cleanupExpiredSessions,
} from './utils/sessionCleanup.js';
import {
  initializeApp,
  type InitializationResult,
} from './core/initializer.js';
import { validateAuthMethod } from './config/auth.js';
import { runAcpClient } from './acp/acpClient.js';
import { validateNonInteractiveAuth } from './validateNonInterActiveAuth.js';
import { appEvents, AppEvent } from './utils/events.js';
import { SessionError, SessionSelector } from './utils/sessionUtils.js';
import {
  relaunchAppInChildProcess,
  relaunchOnExitCode,
} from './utils/relaunch.js';
import { loadSandboxConfig } from './config/sandboxConfig.js';
import { deleteSession, listSessions } from './utils/sessions.js';
import { createPolicyUpdater } from './config/policy.js';
import { isAlternateBufferEnabled } from './ui/hooks/useAlternateBuffer.js';
import { setupTerminalAndTheme } from './utils/terminalTheme.js';
import { runDeferredCommand } from './deferred.js';
import { cleanupBackgroundLogs } from './utils/logCleanup.js';
import { SlashCommandConflictHandler } from './services/SlashCommandConflictHandler.js';

export function validateDnsResolutionOrder(
  order: string | undefined,
): DnsResolutionOrder {
  const defaultValue: DnsResolutionOrder = 'ipv4first';
  if (order === undefined) {
    return defaultValue;
  }
  if (order === 'ipv4first' || order === 'verbatim') {
    return order;
  }
  debugLogger.warn(
    `Invalid value for dnsResolutionOrder in settings: "${order}". Using default "${defaultValue}".`,
  );
  return defaultValue;
}

export function getNodeMemoryArgs(isDebugMode: boolean): string[] {
  const totalMemoryMB = os.totalmem() / (1024 * 1024);
  const heapStats = v8.getHeapStatistics();
  const currentMaxOldSpaceSizeMb = Math.floor(
    heapStats.heap_size_limit / 1024 / 1024,
  );
  const targetMaxOldSpaceSizeInMB = Math.floor(totalMemoryMB * 0.5);
  if (isDebugMode) {
    debugLogger.debug(
      `Current heap size ${currentMaxOldSpaceSizeMb.toFixed(2)} MB`,
    );
  }
  if (process.env['GEMINI_CLI_NO_RELAUNCH']) {
    return [];
  }
  if (targetMaxOldSpaceSizeInMB > currentMaxOldSpaceSizeMb) {
    if (isDebugMode) {
      debugLogger.debug(
        `Need to relaunch with more memory: ${targetMaxOldSpaceSizeInMB.toFixed(2)} MB`,
      );
    }
    return [`--max-old-space-size=${targetMaxOldSpaceSizeInMB}`];
  }
  return [];
}

export function setupUnhandledRejectionHandler() {
  let unhandledRejectionOccurred = false;
  process.on('unhandledRejection', (reason, _promise) => {
    const errorMessage = `=========================================
This is an unexpected error. Please file a bug report using the /bug tool.
CRITICAL: Unhandled Promise Rejection!
=========================================
Reason: ${reason}${
      reason instanceof Error && reason.stack
        ? `
Stack trace:
${reason.stack}`
        : ''
    }`;
    debugLogger.error(errorMessage);
    if (!unhandledRejectionOccurred) {
      unhandledRejectionOccurred = true;
      appEvents.emit(AppEvent.OpenDebugConsole);
    }
  });
}

export async function startInteractiveUI(
  config: Config,
  settings: LoadedSettings,
  startupWarnings: StartupWarning[],
  workspaceRoot: string = process.cwd(),
  resumedSessionData: ResumedSessionData | undefined,
  initializationResult: InitializationResult,
) {
  const { startInteractiveUI: doStartUI } = await import('./interactiveCli.js');
  await doStartUI(
    config,
    settings,
    startupWarnings,
    workspaceRoot,
    resumedSessionData,
    initializationResult,
  );
}

export async function main() {
  const cliStartupHandle = startupProfiler.start('cli_startup');

  const adminControlsListner = setupAdminControlsListener();
  registerCleanup(adminControlsListner.cleanup);

  const cleanupStdio = patchStdio();
  registerSyncCleanup(() => {
    initializeOutputListenersAndFlush();
    cleanupStdio();
  });

  setupUnhandledRejectionHandler();
  setupSignalHandlers();

  const slashCommandConflictHandler = new SlashCommandConflictHandler();
  slashCommandConflictHandler.start();
  registerCleanup(() => slashCommandConflictHandler.stop());

  const loadSettingsHandle = startupProfiler.start('load_settings');
  const settings = loadSettings();
  loadSettingsHandle?.end();

  const requestedWorktree = cliConfig.getRequestedWorktreeName(settings);
  let worktreeInfo: WorktreeInfo | undefined;
  if (requestedWorktree !== undefined) {
    const worktreeHandle = startupProfiler.start('setup_worktree');
    worktreeInfo = await setupWorktree(requestedWorktree || undefined);
    worktreeHandle?.end();
  }

  const cleanupOpsHandle = startupProfiler.start('cleanup_ops');
  Promise.all([
    cleanupCheckpoints(),
    cleanupToolOutputFiles(settings.merged),
    cleanupBackgroundLogs(),
  ])
    .catch((e) => {
      debugLogger.error('Early cleanup failed:', e);
    })
    .finally(() => {
      cleanupOpsHandle?.end();
    });

  const parseArgsHandle = startupProfiler.start('parse_arguments');
  const argvPromise = parseArguments(settings.merged).finally(() => {
    parseArgsHandle?.end();
  });

  const rawStartupWarningsPromise = getStartupWarnings();

  settings.errors.forEach((error) => {
    coreEvents.emitFeedback('warning', error.message);
  });

  const trustedFolders = loadTrustedFolders();
  trustedFolders.errors.forEach((error: TrustedFoldersError) => {
    coreEvents.emitFeedback(
      'warning',
      `Error in ${error.path}: ${error.message}`,
    );
  });

  const argv = await argvPromise;


  if (
    (argv.allowedTools && argv.allowedTools.length > 0) ||
    (settings.merged.tools?.allowed && settings.merged.tools.allowed.length > 0)
  ) {
    coreEvents.emitFeedback(
      'warning',
      'Warning: --allowed-tools cli argument and tools.allowed in settings.json are deprecated and will be removed in 1.0: Migrate to Policy Engine: https://geminicli.com/docs/core/policy-engine/',
    );
  }

  if (
    settings.merged.tools?.exclude &&
    settings.merged.tools.exclude.length > 0
  ) {
    coreEvents.emitFeedback(
      'warning',
      'Warning: tools.exclude in settings.json is deprecated and will be removed in 1.0. Migrate to Policy Engine: https://geminicli.com/docs/core/policy-engine/',
    );
  }

  if (argv.startupMessages) {
    argv.startupMessages.forEach((msg) => {
      coreEvents.emitFeedback('info', msg);
    });
  }

  if (argv.promptInteractive && !process.stdin.isTTY) {
    writeToStderr(
      'Error: The --prompt-interactive flag cannot be used when input is piped from stdin.\n',
    );
    await runExitCleanup();
    process.exit(ExitCodes.FATAL_INPUT_ERROR);
  }

  const isDebugMode = cliConfig.isDebugMode(argv);
  const consolePatcher = new ConsolePatcher({
    stderr: true,
    interactive: isHeadlessMode() ? false : true,
    debugMode: isDebugMode,
    onNewMessage: (msg) => {
      coreEvents.emitConsoleLog(msg.type, msg.content);
    },
  });
  consolePatcher.patch();
  registerCleanup(consolePatcher.cleanup);

  dns.setDefaultResultOrder(
    validateDnsResolutionOrder(settings.merged.advanced.dnsResolutionOrder),
  );

  if (
    !settings.merged.security.auth.selectedType ||
    settings.merged.security.auth.selectedType === AuthType.LEGACY_CLOUD_SHELL
  ) {
    if (
      process.env['CLOUD_SHELL'] === 'true' ||
      process.env['GEMINI_CLI_USE_COMPUTE_ADC'] === 'true'
    ) {
      settings.setValue(
        SettingScope.User,
        'security.auth.selectedType',
        AuthType.COMPUTE_ADC,
      );
    }
  }

  const partialConfig = await loadCliConfig(settings.merged, sessionId, argv, {
    projectHooks: settings.workspace.settings.hooks,
  });
  adminControlsListner.setConfig(partialConfig);

  let initialAuthFailed = false;
  if (!settings.merged.security.auth.useExternal && !argv.isCommand) {
    try {
      if (
        partialConfig.isInteractive() &&
        settings.merged.security.auth.selectedType
      ) {
        const err = validateAuthMethod(
          settings.merged.security.auth.selectedType,
        );
        if (err) {
          throw new Error(err);
        }
        await partialConfig.refreshAuth(
          settings.merged.security.auth.selectedType,
        );
      } else if (!partialConfig.isInteractive()) {
        const authType = await validateNonInteractiveAuth(
          settings.merged.security.auth.selectedType,
          settings.merged.security.auth.useExternal,
          partialConfig,
          settings,
        );
        await partialConfig.refreshAuth(authType);
      }
    } catch (err) {
      if (err instanceof ValidationCancelledError) {
        await runExitCleanup();
        process.exit(ExitCodes.SUCCESS);
      }
      if (!(err instanceof ValidationRequiredError)) {
        debugLogger.error('Error authenticating:', err);
        initialAuthFailed = true;
      }
    }
  }

  const remoteAdminSettings = partialConfig.getRemoteAdminSettings();
  if (remoteAdminSettings) {
    settings.setRemoteAdminSettings(remoteAdminSettings);
  }

  await runDeferredCommand(settings.merged);


  if (!process.env['SANDBOX'] && !argv.isCommand && argv._?.[0] !== 'serve') {
    const memoryArgs = settings.merged.advanced.autoConfigureMemory
      ? getNodeMemoryArgs(isDebugMode)
      : [];
    const sandboxConfig = await loadSandboxConfig(settings.merged, argv);

    if (sandboxConfig) {
      if (initialAuthFailed) {
        await runExitCleanup();
        process.exit(ExitCodes.FATAL_AUTHENTICATION_ERROR);
      }
      let stdinData = '';
      if (!process.stdin.isTTY) {
        stdinData = await readStdin();
      }

      const injectStdinIntoArgs = (
        args: string[],
        stdinData?: string,
      ): string[] => {
        const finalArgs = [...args];
        if (stdinData) {
          const promptIndex = finalArgs.findIndex(
            (arg) => arg === '--prompt' || arg === '-p',
          );
          if (promptIndex > -1 && finalArgs.length > promptIndex + 1) {
            finalArgs[promptIndex + 1] =
              `${stdinData}\n\n${finalArgs[promptIndex + 1]}`;
          } else {
            finalArgs.push('--prompt', stdinData);
          }
        }
        return finalArgs;
      };

      const sandboxArgs = injectStdinIntoArgs(process.argv, stdinData);
      await relaunchOnExitCode(() =>
        start_sandbox(sandboxConfig, memoryArgs, partialConfig, sandboxArgs),
      );
      await runExitCleanup();
      process.exit(ExitCodes.SUCCESS);
    } else {
      // Relaunch app so we always have a child process that can be internally
      // restarted if needed.
      await relaunchAppInChildProcess(memoryArgs, [], remoteAdminSettings);
    }
  }

  {

    const loadConfigHandle = startupProfiler.start('load_cli_config');
    const config = await loadCliConfig(settings.merged, sessionId, argv, {
      projectHooks: settings.workspace.settings.hooks,
      worktreeSettings: worktreeInfo,
    });
    loadConfigHandle?.end();

    await config.storage.initialize();

    adminControlsListner.setConfig(config);

    if (config.isInteractive() && settings.merged.general.devtools) {
      const { setupInitialActivityLogger } = await import(
        './utils/devtoolsService.js'
      );
      await setupInitialActivityLogger(config);
    }

    registerTelemetryConfig(config);

    const policyEngine = config.getPolicyEngine();
    const messageBus = config.getMessageBus();
    createPolicyUpdater(policyEngine, messageBus, config.storage);

    registerCleanup(async () => {
      await config.getHookSystem()?.fireSessionEndEvent(SessionEndReason.Exit);
    });

    cleanupExpiredSessions(config, settings.merged).catch((e) => {
      debugLogger.error('Failed to cleanup expired sessions:', e);
    });

    if (config.getListExtensions()) {
      debugLogger.log('Installed extensions:');
      for (const extension of config.getExtensions()) {
        debugLogger.log(`- ${extension.name}`);
      }
      await runExitCleanup();
      process.exit(ExitCodes.SUCCESS);
    }

    if (config.getListSessions()) {
      const authType = settings.merged.security.auth.selectedType;
      if (authType) {
        try {
          await config.refreshAuth(authType);
        } catch (e) {
          debugLogger.debug(
            'Auth failed for --list-sessions, summaries may not be generated:',
            e,
          );
        }
      }
      await listSessions(config);
      await runExitCleanup();
      process.exit(ExitCodes.SUCCESS);
    }

    const sessionToDelete = config.getDeleteSession();
    if (sessionToDelete) {
      await deleteSession(config, sessionToDelete);
      await runExitCleanup();
      process.exit(ExitCodes.SUCCESS);
    }

    if (argv._?.[0] === 'serve') {
  try {
    const selectedType = settings.merged.security.auth.selectedType;
    if (selectedType) {
      await config.refreshAuth(selectedType);
    }

    // Install a filtered output listener before initializeOutputListenersAndFlush
    // so its listenerCount guard skips adding an unfiltered one. drainBacklogs()
    // still fires inside initializeOutputListenersAndFlush, routed through this filter.
    const SERVE_SUPPRESS = [
      'Keychain initialization encountered an error',
      'Using FileKeychain fallback',
    ];
    const matchesSuppressed = (chunk: unknown): boolean => {
      const text = typeof chunk === 'string' ? chunk : Buffer.isBuffer(chunk) ? chunk.toString() : String(chunk);
      return SERVE_SUPPRESS.some((s) => text.includes(s));
    };

    coreEvents.on(CoreEvent.Output, (payload: OutputPayload) => {
      if (matchesSuppressed(payload.chunk)) return;
      if (payload.isStderr) writeToStderr(payload.chunk, payload.encoding);
      else writeToStdout(payload.chunk, payload.encoding);
    });

    coreEvents.on(CoreEvent.ConsoleLog, (payload: ConsoleLogPayload) => {
      if (matchesSuppressed(payload.content)) return;
      if (payload.type === 'error' || payload.type === 'warn') writeToStderr(payload.content);
      else writeToStdout(payload.content);
    });

    initializeOutputListenersAndFlush();
    await runServeCommand(config, argv.port);
  } catch (err) {
    writeSync(2, `Serve mode error: ${err}\n`);
  }
  await runExitCleanup();
  return;
}

    const wasRaw = process.stdin.isRaw;
    if (config.isInteractive() && !wasRaw && process.stdin.isTTY) {
      process.stdin.setRawMode(true);
      registerSyncCleanup(() => {
        process.stdin.setRawMode(wasRaw);
      });
    }

    const terminalHandle = startupProfiler.start('setup_terminal');
    await setupTerminalAndTheme(config, settings);
    terminalHandle?.end();

    const initAppHandle = startupProfiler.start('initialize_app');
    const initializationResult = await initializeApp(config, settings);
    initAppHandle?.end();

    if (
      settings.merged.security.auth.selectedType ===
        AuthType.LOGIN_WITH_GOOGLE &&
      config.isBrowserLaunchSuppressed()
    ) {
      await getOauthClient(settings.merged.security.auth.selectedType, config);
    }

    if (config.getAcpMode()) {
      return runAcpClient(config, settings, argv);
    }

    let input = config.getQuestion();
    const useAlternateBuffer = shouldEnterAlternateScreen(
      isAlternateBufferEnabled(config),
      config.getScreenReader(),
    );
    const rawStartupWarnings = await rawStartupWarningsPromise;
    const startupWarnings: StartupWarning[] = [
      ...rawStartupWarnings.map((message) => ({
        id: `startup-${createHash('sha256').update(message).digest('hex').substring(0, 16)}`,
        message,
        priority: WarningPriority.High,
      })),
      ...(await getUserStartupWarnings(settings.merged, undefined, {
        isAlternateBuffer: useAlternateBuffer,
      })),
    ];

    let resumedSessionData: ResumedSessionData | undefined = undefined;
    if (argv.resume) {
      const sessionSelector = new SessionSelector(config);
      try {
        const result = await sessionSelector.resolveSession(argv.resume);
        resumedSessionData = {
          conversation: result.sessionData,
          filePath: result.sessionPath,
        };
        config.setSessionId(resumedSessionData.conversation.sessionId);
      } catch (error) {
        if (
          error instanceof SessionError &&
          error.code === 'NO_SESSIONS_FOUND'
        ) {
          startupWarnings.push({
            id: 'resume-no-sessions',
            message: error.message,
            priority: WarningPriority.High,
          });
        } else {
          coreEvents.emitFeedback(
            'error',
            `Error resuming session: ${error instanceof Error ? error.message : 'Unknown error'}`,
          );
          await runExitCleanup();
          process.exit(ExitCodes.FATAL_INPUT_ERROR);
        }
      }
    }

    cliStartupHandle?.end();

    if (config.isInteractive()) {
      if (process.stdin.isTTY) {
        process.stdin.resume();
      }
      await startInteractiveUI(
        config,
        settings,
        startupWarnings,
        process.cwd(),
        resumedSessionData,
        initializationResult,
      );
      return;
    }

    await config.initialize();
    startupProfiler.flush(config);

    let stdinData: string | undefined = undefined;
    if (!process.stdin.isTTY) {
      stdinData = await readStdin();
      if (stdinData) {
        input = input ? `${stdinData}\n\n${input}` : stdinData;
      }
    }

    const sessionStartSource = resumedSessionData
      ? SessionStartSource.Resume
      : SessionStartSource.Startup;

    const hookSystem = config?.getHookSystem();
    if (hookSystem) {
      const result = await hookSystem.fireSessionStartEvent(sessionStartSource);
      if (result) {
        if (result.systemMessage) {
          writeToStderr(result.systemMessage + '\n');
        }
        const additionalContext = result.getAdditionalContext();
        if (additionalContext) {
          const wrappedContext = `<hook_context>${additionalContext}</hook_context>`;
          input = input ? `${wrappedContext}\n\n${input}` : wrappedContext;
        }
      }
    }

    if (!input) {
      debugLogger.error(
        `No input provided via stdin. Input can be provided by piping data into gemini or using the --prompt option.`,
      );
      await runExitCleanup();
      process.exit(ExitCodes.FATAL_INPUT_ERROR);
    }

    const prompt_id = sessionId;
    logUserPrompt(
      config,
      new UserPromptEvent(
        input.length,
        prompt_id,
        config.getContentGeneratorConfig()?.authType,
        input,
      ),
    );

    const authType = await validateNonInteractiveAuth(
      settings.merged.security.auth.selectedType,
      settings.merged.security.auth.useExternal,
      config,
      settings,
    );
    await config.refreshAuth(authType);

    if (config.getDebugMode()) {
      debugLogger.log('Session ID: %s', sessionId);
    }

    initializeOutputListenersAndFlush();

    await runNonInteractive({
      config,
      settings,
      input,
      prompt_id,
      resumedSessionData,
    });
    await runExitCleanup();
    process.exit(ExitCodes.SUCCESS);
  }
}

export function initializeOutputListenersAndFlush() {
  if (coreEvents.listenerCount(CoreEvent.Output) === 0) {
    coreEvents.on(CoreEvent.Output, (payload: OutputPayload) => {
      if (payload.isStderr) {
        writeToStderr(payload.chunk, payload.encoding);
      } else {
        writeToStdout(payload.chunk, payload.encoding);
      }
    });

    if (coreEvents.listenerCount(CoreEvent.ConsoleLog) === 0) {
      coreEvents.on(CoreEvent.ConsoleLog, (payload: ConsoleLogPayload) => {
        if (payload.type === 'error' || payload.type === 'warn') {
          writeToStderr(payload.content);
        } else {
          writeToStdout(payload.content);
        }
      });
    }

    if (coreEvents.listenerCount(CoreEvent.UserFeedback) === 0) {
      coreEvents.on(CoreEvent.UserFeedback, (payload: UserFeedbackPayload) => {
        if (payload.severity === 'error' || payload.severity === 'warning') {
          writeToStderr(payload.message);
        } else {
          writeToStdout(payload.message);
        }
      });
    }
  }
  coreEvents.drainBacklogs();
}

function setupAdminControlsListener() {
  let pendingSettings: AdminControlsSettings | undefined;
  let config: Config | undefined;

  const messageHandler = (msg: unknown) => {
    // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
    const message = msg as {
      type?: string;
      settings?: AdminControlsSettings;
    };
    if (message?.type === 'admin-settings' && message.settings) {
      if (config) {
        config.setRemoteAdminSettings(message.settings);
      } else {
        pendingSettings = message.settings;
      }
    }
  };

  process.on('message', messageHandler);

  return {
    setConfig: (newConfig: Config) => {
      config = newConfig;
      if (pendingSettings) {
        config.setRemoteAdminSettings(pendingSettings);
      }
    },
    cleanup: () => {
      process.off('message', messageHandler);
    },
  };
}