'use strict';

// Runs before the stylesheet so the saved preference is applied before first paint.
(() => {
  const storageKey = 'halobridge_theme';
  const systemTheme = window.matchMedia('(prefers-color-scheme: dark)');
  const normalize = value => ['system', 'light', 'dark'].includes(value) ? value : 'system';
  let preference = 'system';
  try { preference = normalize(localStorage.getItem(storageKey)); } catch {}

  function applyTheme() {
    const dark = preference === 'dark' || (preference === 'system' && systemTheme.matches);
    document.documentElement.dataset.theme = dark ? 'dark' : 'light';
    document.querySelector('meta[name="theme-color"]').content = dark ? '#111820' : '#f5f6f8';
    const select = document.getElementById('themeSelect');
    if (select) select.value = preference;
  }

  applyTheme();
  systemTheme.addEventListener('change', applyTheme);
  window.addEventListener('storage', event => {
    if (event.storageArea !== localStorage || (event.key !== storageKey && event.key !== null)) return;
    preference = normalize(event.newValue);
    applyTheme();
  });
  document.addEventListener('DOMContentLoaded', () => {
    const select = document.getElementById('themeSelect');
    select.value = preference;
    select.addEventListener('change', () => {
      preference = normalize(select.value);
      applyTheme();
      try { localStorage.setItem(storageKey, preference); } catch {}
    });
  });
})();
