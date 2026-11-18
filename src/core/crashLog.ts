const CRASH_STORAGE_KEY = 'gamelens_crash_log';
const MAX_CRASH_BYTES = 20_000;

export function readCrashLog(): string {
  try {
    return localStorage.getItem(CRASH_STORAGE_KEY) ?? '';
  } catch {
    return '';
  }
}

export function clearCrashLog(): void {
  try {
    localStorage.removeItem(CRASH_STORAGE_KEY);
  } catch {
    // ignore storage failures
  }
}

export function recordCrash(where: string, error: unknown): string {
  const stamp = new Date().toISOString().replace('T', ' ').slice(0, 19);
  const message =
    error instanceof Error
      ? `${error.name}: ${error.message}\n${error.stack ?? ''}`
      : String(error);
  const entry = `[${stamp}] ${where}\n${message}\n\n`;
  try {
    const prev = readCrashLog();
    const combined = (prev + entry).slice(-MAX_CRASH_BYTES);
    localStorage.setItem(CRASH_STORAGE_KEY, combined);
    return combined;
  } catch {
    return entry;
  }
}
