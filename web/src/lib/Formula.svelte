<script>
  import katex from 'katex';

  export let expression = '';
  export let displayMode = true;

  $: rendered = katex.renderToString(expression, {
    displayMode,
    throwOnError: false,
    strict: 'ignore'
  });
</script>

<div class:display={displayMode} class="formula">
  {@html rendered}
</div>

<style>
  .formula {
    overflow-x: auto;
  }

  .formula.display :global(.katex-display) {
    margin: 0;
  }
</style>
