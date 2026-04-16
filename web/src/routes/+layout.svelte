<script>
  import { onMount } from "svelte";
  import { initializeTheme } from "$lib/theme";
  import '../app.css';
  import 'katex/dist/katex.min.css';

  onMount(() => initializeTheme());
</script>

<slot />
