import { browser } from "$app/environment";
import { get, writable } from "svelte/store";

export type ThemePreference = "light" | "dark" | "system";
export type ResolvedTheme = "light" | "dark";

const STORAGE_KEY = "theme-preference";

function isThemePreference(value: string | null): value is ThemePreference {
  return value === "light" || value === "dark" || value === "system";
}

function getStoredPreference(): ThemePreference {
  if (!browser) {
    return "system";
  }

  const stored = window.localStorage.getItem(STORAGE_KEY);
  return isThemePreference(stored) ? stored : "system";
}

function getSystemTheme(): ResolvedTheme {
  if (!browser) {
    return "light";
  }

  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

function resolveTheme(preference: ThemePreference): ResolvedTheme {
  return preference === "system" ? getSystemTheme() : preference;
}

function applyTheme(preference: ThemePreference) {
  if (!browser) {
    return;
  }

  const resolved = resolveTheme(preference);
  document.documentElement.dataset.theme = resolved;
  document.documentElement.dataset.themePreference = preference;
  document.documentElement.style.colorScheme = resolved;
  resolvedTheme.set(resolved);
}

export const themePreference = writable<ThemePreference>("system");
export const resolvedTheme = writable<ResolvedTheme>("light");

export function setThemePreference(preference: ThemePreference) {
  themePreference.set(preference);
}

export function initializeTheme() {
  if (!browser) {
    return () => {};
  }

  const initialPreference = getStoredPreference();
  themePreference.set(initialPreference);
  applyTheme(initialPreference);

  const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
  const handleSystemThemeChange = () => {
    if (get(themePreference) === "system") {
      applyTheme("system");
    }
  };

  const unsubscribe = themePreference.subscribe((preference) => {
    window.localStorage.setItem(STORAGE_KEY, preference);
    applyTheme(preference);
  });

  mediaQuery.addEventListener("change", handleSystemThemeChange);

  return () => {
    unsubscribe();
    mediaQuery.removeEventListener("change", handleSystemThemeChange);
  };
}
