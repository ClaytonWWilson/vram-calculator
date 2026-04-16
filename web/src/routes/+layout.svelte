<script>
  import { onMount } from "svelte";
  import { initializeTheme } from "$lib/theme";
  import '../app.css';
  import 'katex/dist/katex.min.css';

  onMount(() => initializeTheme());
</script>

<div class="site-layout">
  <slot />

  <footer class="site-footer">
    <p>
      Made with
      <span class="heart-mark" aria-label="love">
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path
            d="M12 21s-6.7-4.4-9.4-8.2C.7 10.1 1 6.3 3.8 4.5c2.2-1.5 5.1-.9 6.9 1.2L12 7.1l1.3-1.4c1.8-2.1 4.7-2.7 6.9-1.2 2.8 1.8 3.1 5.6 1.2 8.3C18.7 16.6 12 21 12 21Z"
          />
        </svg>
      </span>
      by
      <a
        class="github-link"
        href="https://github.com/ClaytonWWilson"
        target="_blank"
        rel="noreferrer"
      >
        <svg class="github-mark" viewBox="0 0 24 24" aria-hidden="true">
          <path
            d="M12 2C6.5 2 2 6.6 2 12.2c0 4.5 2.9 8.3 6.8 9.7.5.1.7-.2.7-.5v-1.9c-2.8.6-3.4-1.2-3.4-1.2-.5-1.2-1.1-1.5-1.1-1.5-.9-.6.1-.6.1-.6 1 .1 1.6 1.1 1.6 1.1.9 1.6 2.4 1.1 3 .8.1-.7.4-1.1.7-1.4-2.2-.3-4.6-1.1-4.6-5 0-1.1.4-2 1.1-2.8-.1-.3-.5-1.3.1-2.7 0 0 .9-.3 2.9 1.1a9.7 9.7 0 0 1 5.2 0c2-1.4 2.9-1.1 2.9-1.1.6 1.4.2 2.4.1 2.7.7.8 1.1 1.7 1.1 2.8 0 3.9-2.4 4.7-4.6 5 .4.3.7 1 .7 2v3c0 .3.2.6.7.5 3.9-1.4 6.8-5.2 6.8-9.7C22 6.6 17.5 2 12 2Z"
          />
        </svg>
        Clayton
      </a>
    </p>
  </footer>
</div>

<style>
  .site-layout {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  :global(.page-shell) {
    flex: 1;
  }

  .site-footer {
    width: min(100%, 1100px);
    margin: 0 auto;
    padding: 0 1.25rem 2rem;
    text-align: center;
    color: var(--muted);
  }

  .site-footer p {
    margin: 0;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.45rem;
    flex-wrap: wrap;
  }

  .heart-mark,
  .github-link {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
  }

  .heart-mark {
    color: #d65a4a;
  }

  .heart-mark svg,
  .github-mark {
    width: 1rem;
    height: 1rem;
    fill: currentColor;
  }

  .site-footer a {
    color: var(--accent);
    font-weight: 700;
    text-decoration: none;
  }

  .site-footer a:hover,
  .site-footer a:focus-visible {
    text-decoration: underline;
  }

  @media (max-width: 640px) {
    .site-footer {
      padding-inline: 0.85rem;
    }
  }
</style>
