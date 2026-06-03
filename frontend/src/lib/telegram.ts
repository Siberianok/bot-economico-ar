declare global {
  interface Window {
    Telegram?: {
      WebApp?: {
        initData?: string;
        ready?: () => void;
        expand?: () => void;
      };
    };
  }
}

export function initTelegram(): void {
  const webApp = window.Telegram?.WebApp;
  if (!webApp) {
    return;
  }
  webApp.ready?.();
  webApp.expand?.();
}

export function getTelegramInitData(): string {
  return window.Telegram?.WebApp?.initData || "";
}

export function getTelegramStatus(): { detected: boolean; initDataAvailable: boolean; userId: string } {
  const initData = getTelegramInitData();
  return {
    detected: Boolean(window.Telegram?.WebApp),
    initDataAvailable: Boolean(initData),
    userId: import.meta.env.VITE_DEV_TELEGRAM_USER_ID || "local-dev",
  };
}

