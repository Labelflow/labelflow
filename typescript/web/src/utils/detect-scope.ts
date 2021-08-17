import {
  detect,
  BrowserInfo,
  SearchBotDeviceInfo,
  BotInfo,
  NodeInfo,
  ReactNativeInfo,
} from "detect-browser";

declare let self: ServiceWorkerGlobalScope;

// eslint-disable-next-line import/no-mutable-exports
export let isInWindowScope: boolean | null = null;

// eslint-disable-next-line import/no-mutable-exports
export let isInServiceWorkerScope: boolean | null = null;

// eslint-disable-next-line import/no-mutable-exports
export let browser:
  | BrowserInfo
  | SearchBotDeviceInfo
  | BotInfo
  | NodeInfo
  | ReactNativeInfo
  | null = null;

// Robust way to detect if window exists, only once
const detectScope = () => {
  if (isInWindowScope === null) {
    try {
      if (window) {
        isInWindowScope = true;

        browser = detect();
      } else {
        isInWindowScope = false;
      }
    } catch (e) {
      isInWindowScope = false;
    }
  }

  if (isInServiceWorkerScope === null) {
    try {
      if (self) {
        isInServiceWorkerScope = true;
      } else {
        isInServiceWorkerScope = false;
      }
    } catch (e) {
      isInServiceWorkerScope = false;
    }
  }
};
detectScope();