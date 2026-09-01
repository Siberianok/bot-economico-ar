declare global {
  interface Window {
    Telegram?: {
      WebApp?: {
        initData?: string;
        initDataUnsafe?: {
          user?: {
            id?: number;
          };
        };
        platform?: string;
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

export function getTelegramStatus(): {
  sdkLoaded: boolean;
  detected: boolean;
  initDataAvailable: boolean;
  userId: string;
} {
  const webApp = window.Telegram?.WebApp;
  const initData = getTelegramInitData();
  const telegramUserId = webApp?.initDataUnsafe?.user?.id;
  const detected = Boolean(
    initData || telegramUserId || (webApp?.platform && webApp.platform !== "unknown"),
  );
  const developmentUserId = import.meta.env.DEV
    ? import.meta.env.VITE_DEV_TELEGRAM_USER_ID || "local-dev"
    : "no disponible";
  return {
    sdkLoaded: Boolean(webApp),
    detected,
    initDataAvailable: Boolean(initData),
    userId: telegramUserId ? String(telegramUserId) : developmentUserId,
  };
}
