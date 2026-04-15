#!/usr/bin/env python3
"""
LLM VRAM/RAM Calculator - TUI for determining if your system can run a specific LLM.
"""

import os
import re
import sys
import time
import math
from contextlib import contextmanager
from typing import Optional, List, Tuple
from huggingface_hub import HfApi, hf_hub_download
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
import json


class ModelConfig:
    """Holds model configuration and available quantizations."""
    
    def __init__(self, name: str, num_layers: int, num_heads: int, head_dim: int,
                 max_context_length: int, total_params: float):
        self.name = name
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_context_length = max_context_length
        self.total_params = total_params  # in billions
        self.quantizations: List[Tuple[str, int]] = []  # [(filename, bits_per_weight), ...]
    
    def add_quantization(self, filename: str, bits: int):
        """Add a quantization option."""
        self.quantizations.append((filename, bits))
    
    @property
    def sorted_quantizations(self) -> List[Tuple[str, int]]:
        """Return quantizations sorted by bits (highest first)."""
        return sorted(self.quantizations, key=lambda x: x[1], reverse=True)


class LLMCalculator:
    """Main calculator class for LLM memory requirements."""
    
    # Context size presets (in tokens)
    CONTEXT_PRESETS = [
        ("4K", 4096),
        ("8K", 8192),
        ("16K", 16384),
        ("32K", 32768),
        ("64K", 65536),
        ("128K", 131072),
        ("256K", 262144),
    ]
    
    def __init__(self, vram_gb: float, ram_gb: float):
        self.vram_total = vram_gb
        self.ram_total = ram_gb
        self.model_config: Optional[ModelConfig] = None
        self.selected_context_tokens: int = 4096
        self.selected_quantization: Optional[str] = None
        self.console = Console()
    
    def extract_base_model(self, hf_url: str) -> tuple[str, str]:
        """
        Extract base model name from HuggingFace GGUF URL.
        Returns (base_model_name, gguf_repo_name)
        
        Uses the model card's base_model field to find the true base model,
        following the chain: quantized -> finetuned -> base
        """
        # Remove trailing slashes and clean up
        hf_url = hf_url.strip().rstrip('/')
        
        # Handle both full URLs and just repo names
        if '://' in hf_url:
            # Full URL like https://huggingface.co/unsloth/Qwen3.5-4B-GGUF
            match = re.search(r'huggingface\.co/(.+?)/(.+)', hf_url)
            if not match:
                raise ValueError(f"Invalid HuggingFace URL format: {hf_url}")
            owner, repo = match.groups()
        else:
            # Just repo name like unsloth/Qwen3.5-4B-GGUF
            parts = hf_url.split('/')
            if len(parts) != 2:
                raise ValueError(f"Invalid format. Expected 'owner/repo' or full URL, got: {hf_url}")
            owner, repo = parts
        
        gguf_repo_name = f"{owner}/{repo}"
        
        # Follow the base_model chain to find the true base model
        current_model = gguf_repo_name
        visited = set()
        max_depth = 10  # Prevent infinite loops
        last_with_base = None
        
        for depth in range(max_depth):
            if current_model in visited:
                break
            visited.add(current_model)
            
            try:
                api = HfApi()
                info = api.model_info(current_model)
                
                # Check card_data for base_model field
                if hasattr(info, 'card_data') and info.card_data is not None:
                    base_models = getattr(info.card_data, 'base_model', [])
                    if isinstance(base_models, list) and len(base_models) > 0:
                        last_with_base = current_model
                        next_model = base_models[0]
                        current_model = next_model
                        continue
            except Exception as e:
                self.console.print(f"[yellow]Warning: Could not fetch info for {current_model}: {e}[/yellow]")
            
            # No more base_model found, return the last one we found
            if last_with_base is not None:
                api = HfApi()
                info = api.model_info(last_with_base)
                base_models = getattr(info.card_data, 'base_model', [])
                if isinstance(base_models, list) and len(base_models) > 0:
                    return base_models[0], gguf_repo_name
            break
        
        # Heuristic fallback (original method)
        base_repo = repo
        if '-GGUF' in base_repo:
            base_repo = base_repo.replace('-GGUF', '')
        
        owner_mapping = {
            'unsloth': lambda r: self._infer_owner_from_model(r),
            'bartowski': lambda r: self._infer_owner_from_model(r),
            'TheBloke': lambda r: self._infer_owner_from_model(r),
            'MaziyarPanahi': lambda r: self._infer_owner_from_model(r),
        }
        
        base_owner = owner_mapping.get(owner, lambda r: owner)(repo)
        base_model_name = f"{base_owner}/{base_repo}"
        
        return base_model_name, gguf_repo_name
    
    def _infer_owner_from_model(self, model_name: str) -> str:
        """Infer the original owner based on model name patterns."""
        # Common patterns for popular models
        if 'llama' in model_name.lower() or 'llama3' in model_name.lower():
            return 'meta-llama'
        elif 'qwen' in model_name.lower():
            return 'Qwen'
        elif 'mistral' in model_name.lower():
            return 'mistralai'
        elif 'gemma' in model_name.lower():
            return 'google'
        elif 'phi' in model_name.lower():
            return 'microsoft'
        elif 'olmo' in model_name.lower():
            return 'allenai'
        else:
            # Default fallback
            return 'meta-llama'
    
    def fetch_model_config(self, base_model: str) -> ModelConfig:
        """Fetch and parse config.json from the base model."""
        try:
            # Download config.json
            config_path = hf_hub_download(
                repo_id=base_model,
                filename="config.json",
                repo_type="model",
                force_download=True
            )
            
            with open(config_path, 'r') as f:
                config_data = json.load(f)

            resolved_config = self._resolve_text_config(config_data)
            
            # Extract relevant parameters
            num_layers = resolved_config.get('num_hidden_layers', 
                          resolved_config.get('n_layer', 
                          resolved_config.get('hidden_layers', 32)))
            
            num_heads = resolved_config.get(
                'num_key_value_heads',
                resolved_config.get(
                    'num_attention_heads',
                    resolved_config.get('n_head', 32),
                ),
            )
            
            head_dim = resolved_config.get('head_dim')
            if not head_dim:
                head_dim = resolved_config.get('hidden_size',
                           resolved_config.get('n_embd', 4096)) // max(num_heads, 1)
            
            max_context = self._extract_max_context_length(resolved_config)
            
            # Get total parameters (may need to calculate)
            total_params = self._calculate_total_params(
                resolved_config,
                num_layers,
                num_heads,
                head_dim,
                base_model,
            )
            
            return ModelConfig(
                name=base_model,
                num_layers=num_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                max_context_length=max_context,
                total_params=total_params
            )
            
        except Exception as e:
            self.console.print(f"[red]Error fetching config from {base_model}: {e}[/red]")
            raise

    def _resolve_text_config(self, config_data: dict) -> dict:
        """Flatten nested text-model config fields when present."""
        text_config = config_data.get('text_config')
        if isinstance(text_config, dict):
            return {**config_data, **text_config}
        return config_data

    def _extract_max_context_length(self, config_data: dict) -> int:
        """Extract the best available max context length from model config."""
        candidate_keys = (
            'max_position_embeddings',
            'max_sequence_length',
            'model_max_length',
            'max_seq_len',
            'max_context_length',
            'seq_len',
            'n_positions',
        )

        candidates = [
            int(config_data[key])
            for key in candidate_keys
            if isinstance(config_data.get(key), int) and config_data[key] > 0
        ]

        rope_scaling = config_data.get('rope_scaling')
        if isinstance(rope_scaling, dict):
            original_max = rope_scaling.get('original_max_position_embeddings')
            if isinstance(original_max, int) and original_max > 0:
                candidates.append(original_max)

                factor = rope_scaling.get('factor')
                if isinstance(factor, (int, float)) and factor > 0:
                    candidates.append(int(math.ceil(original_max * factor)))

        return max(candidates, default=4096)
    
    def _extract_param_count_from_identifiers(self, *identifiers: Optional[str]) -> Optional[float]:
        """Infer a marketed parameter count from model identifiers like `Qwen3.5-27B`."""
        matches: List[float] = []
        for identifier in identifiers:
            if not isinstance(identifier, str):
                continue
            for match in re.findall(r'(?<!\d)(\d+(?:\.\d+)?)B(?![A-Za-z])', identifier, flags=re.IGNORECASE):
                try:
                    matches.append(float(match))
                except ValueError:
                    continue
        return max(matches) if matches else None

    def _calculate_total_params(
        self,
        config_data: dict,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        base_model: Optional[str] = None,
    ) -> float:
        """Calculate total parameters in billions if not explicitly provided."""
        # Check if params are explicitly stated
        if 'num_parameters' in config_data:
            return config_data['num_parameters'] / 1e9

        if 'parameter_count' in config_data:
            parameter_count = config_data['parameter_count']
            if isinstance(parameter_count, (int, float)) and parameter_count > 0:
                return float(parameter_count) / 1e9

        inferred_from_name = self._extract_param_count_from_identifiers(
            config_data.get('model_name'),
            config_data.get('_name_or_path'),
            config_data.get('name'),
            base_model,
        )
        if inferred_from_name is not None:
            return inferred_from_name
        
        # Estimate based on architecture
        hidden_size = config_data.get('hidden_size', head_dim * num_heads)
        intermediate_size = config_data.get('intermediate_size',
                           config_data.get('ffn_hidden_size',
                           hidden_size * 4))
        vocab_size = config_data.get('vocab_size', 32000)
        
        # Rough estimate for transformer models
        # Each layer: QKV projections + output projection + MLP + norms
        params_per_layer = (
            num_heads * head_dim * head_dim * 4 +  # Q, K, V, O projections (simplified)
            hidden_size * intermediate_size * 2 +   # MLP up/down
            hidden_size * 2                          # Layer norms
        )
        
        total = params_per_layer * num_layers + hidden_size * vocab_size  # Embeddings
        return total / 1e9
    
    def fetch_gguf_quantizations(self, gguf_repo: str) -> List[Tuple[str, int]]:
        """Fetch available GGUF files and their quantization levels."""
        api = HfApi()
        
        try:
            files = api.list_repo_files(gguf_repo, repo_type="model")
            gguf_files = [f for f in files if f.endswith('.gguf') and not f.startswith('.')]
            
            quantizations = []
            for filename in gguf_files:
                bits = self._parse_quantization_bits(filename)
                if bits > 0:
                    quantizations.append((filename, bits))
            
            return quantizations
            
        except Exception as e:
            self.console.print(f"[red]Error fetching GGUF files from {gguf_repo}: {e}[/red]")
            return []
    
    def _parse_quantization_bits(self, filename: str) -> int:
        """Extract bits per weight from GGUF filename."""
        # Common patterns: Q8_0, Q6_K, Q5_K_M, Q4_K_S, FP16, etc.
        filename_upper = filename.upper()
        
        if 'FP32' in filename_upper or 'F32' in filename_upper:
            return 32
        elif 'FP16' in filename_upper or 'F16' in filename_upper:
            return 16
        elif 'Q8' in filename_upper:
            return 8
        elif 'Q6' in filename_upper:
            return 6
        elif 'Q5' in filename_upper:
            return 5
        elif 'Q4' in filename_upper:
            return 4
        elif 'Q3' in filename_upper:
            return 3
        elif 'Q2' in filename_upper:
            return 2
        else:
            # Try to extract number from filename
            match = re.search(r'(\d+)', filename)
            if match:
                return int(match.group(1))
            return 0  # Unknown
    
    def calculate_model_size_gb(self, params_billions: float, bits_per_weight: int) -> float:
        """Calculate model size in GB: (Parameters * Bits) / 8."""
        return (params_billions * bits_per_weight) / 8
    
    def calculate_context_size_gb(self, layers: int, heads: int, head_dim: int, context_length: int) -> float:
        """Calculate context size in GB: (2 * Layers * Heads * Head Dim * Context Length) / 10^9."""
        return (2 * layers * heads * head_dim * context_length) / 1e9
    
    def calculate_total_memory(self, model_size_gb: float, context_size_gb: float) -> tuple[float, float]:
        """
        Calculate VRAM and RAM usage.
        VRAM fills first, then overflow goes to RAM.
        Returns (vram_used, ram_used)
        """
        total_needed = model_size_gb + context_size_gb
        
        vram_used = min(total_needed, self.vram_total)
        remaining = total_needed - vram_used
        ram_used = max(0, remaining) if remaining > 0 else 0
        
        return vram_used, ram_used
    
    def get_available_contexts(self, max_context: int) -> list[tuple[str, int]]:
        """Get context presets capped at model's max."""
        available = []
        for label, tokens in self.CONTEXT_PRESETS:
            if tokens <= max_context:
                available.append((label, tokens))
        return available
    
    def _show_static_view(self, available_contexts: List[Tuple[str, int]]):
        """Show a static view of the first context/quantization combination."""
        sorted_quantizations = self.model_config.sorted_quantizations
        if not sorted_quantizations or not available_contexts:
            return
        
        # Calculate for first options
        params_billions = self.model_config.total_params
        bits_per_weight = sorted_quantizations[0][1]
        ctx_tokens = available_contexts[0][1]
        
        model_size_gb = self.calculate_model_size_gb(params_billions, bits_per_weight)
        context_size_gb = self.calculate_context_size_gb(
            self.model_config.num_layers,
            self.model_config.num_heads,
            self.model_config.head_dim,
            ctx_tokens
        )
        vram_used, ram_used = self.calculate_total_memory(model_size_gb, context_size_gb)
        total_needed = model_size_gb + context_size_gb
        
        # Print static view
        self.console.print(Panel(
            f"Model: {self.model_config.name}\n"
            f"  Parameters: {params_billions:.2f}B | Layers: {self.model_config.num_layers} | "
            f"KV Heads: {self.model_config.num_heads} | Head Dim: {self.model_config.head_dim}\n\n"
            f"Context: {available_contexts[0][0]} | Quantization: {sorted_quantizations[0][0][:40]}...\n"
            f"Model Size:    {model_size_gb:.2f} GB\n"
            f"Context Size:  {context_size_gb:.2f} GB\n"
            f"Total Needed:  {total_needed:.2f} GB\n\n"
            f"VRAM ({self.vram_total}GB): {vram_used:.1f}/{self.vram_total} GB\n"
            f"RAM ({self.ram_total}GB): {ram_used:.1f}/{self.ram_total} GB",
            title="LLM Memory Calculator (Static View)",
            border_style="blue"
        ))

    @contextmanager
    def _terminal_input_mode(self):
        """Put the terminal in a mode where key presses are available immediately."""
        if not sys.stdin.isatty() or sys.platform == 'win32':
            yield
            return

        import termios
        import tty

        fd = sys.stdin.fileno()
        old_term = termios.tcgetattr(fd)

        try:
            tty.setcbreak(fd)
            yield
        finally:
            termios.tcsetattr(fd, termios.TCSANOW, old_term)

    def _read_key(self) -> Optional[str]:
        """Read one normalized key event."""
        if not sys.stdin.isatty():
            return None

        if sys.platform == 'win32':
            import msvcrt

            if not msvcrt.kbhit():
                return None

            ch = msvcrt.getwch()
            if ch in ('\x00', '\xe0'):
                special = msvcrt.getwch()
                return {
                    'H': 'up',
                    'P': 'down',
                    'K': 'left',
                    'M': 'right',
                }.get(special)
            if ch == '\t':
                return 'tab'
            if ch == '\x1b':
                return 'escape'
            return ch

        fd = sys.stdin.fileno()
        import select

        ready, _, _ = select.select([fd], [], [], 0)
        if not ready:
            return None

        try:
            data = os.read(fd, 32)
        except OSError:
            return None

        if not data:
            return None

        if data == b'\t':
            return 'tab'

        if data == b'\x1b':
            ready, _, _ = select.select([fd], [], [], 0.05)
            if ready:
                try:
                    data += os.read(fd, 32)
                except OSError:
                    pass

        sequence = data.decode('utf-8', errors='ignore')
        if not sequence:
            return None
        if sequence == '\t':
            return 'tab'
        if sequence.startswith('\x1b'):
            return self._normalize_escape_sequence(sequence)
        return sequence[0]

    def _normalize_escape_sequence(self, sequence: str) -> str:
        """Normalize terminal escape sequences to logical key names."""
        if sequence == '\x1b':
            return 'escape'

        exact_matches = {
            '\x1b[A': 'up',
            '\x1b[B': 'down',
            '\x1b[C': 'right',
            '\x1b[D': 'left',
            '\x1bOA': 'up',
            '\x1bOB': 'down',
            '\x1bOC': 'right',
            '\x1bOD': 'left',
            '\x1b[Z': 'backtab',
        }
        if sequence in exact_matches:
            return exact_matches[sequence]

        csi_match = re.fullmatch(r'\x1b\[(?:\d+;)*(\d+)?([ABCD])', sequence)
        if csi_match:
            direction = csi_match.group(2)
            return {
                'A': 'up',
                'B': 'down',
                'C': 'right',
                'D': 'left',
            }[direction]

        return sequence

    def _build_panel(self, available_contexts: List[Tuple[str, int]], sorted_quantizations: List[Tuple[str, int]]) -> Panel:
        """Build the current TUI view from the active state."""
        # Calculate memory requirements
        params_billions = self.model_config.total_params
        bits_per_weight = sorted_quantizations[self.selected_quant_index][1]

        model_size_gb = self.calculate_model_size_gb(params_billions, bits_per_weight)
        context_size_gb = self.calculate_context_size_gb(
            self.model_config.num_layers,
            self.model_config.num_heads,
            self.model_config.head_dim,
            available_contexts[self.selected_context_index][1]
        )

        vram_used, ram_used = self.calculate_total_memory(model_size_gb, context_size_gb)
        total_needed = model_size_gb + context_size_gb

        # Build content as a list of strings for Panel
        lines: List[str] = []

        # Model info header
        lines.append(f"Model: {self.model_config.name}")
        lines.append(f"  Parameters: {params_billions:.2f}B | Layers: {self.model_config.num_layers} | "
                     f"KV Heads: {self.model_config.num_heads} | Head Dim: {self.model_config.head_dim}")
        lines.append(f"  Active Pane: {self.current_menu.title()}")
        lines.append("")

        # Context selection menu
        lines.append("Context Size:")
        for i, (ctx_label, _) in enumerate(available_contexts):
            is_selected = i == self.selected_context_index
            prefix = "> " if is_selected else "  "
            if self.current_menu != "context" and is_selected:
                prefix = "* "
            lines.append(f"{prefix}{ctx_label}")
        lines.append("")

        # Quantization selection menu
        lines.append("Quantization:")
        for i, (q_name, _) in enumerate(sorted_quantizations):
            is_selected = i == self.selected_quant_index
            prefix = "> " if is_selected else "  "
            if self.current_menu != "quantization" and is_selected:
                prefix = "* "
            # Truncate long filenames
            display_name = q_name[:48] if len(q_name) > 50 else q_name
            lines.append(f"{prefix}{display_name}")
        lines.append("")

        # Memory breakdown
        selected_ctx_label = available_contexts[self.selected_context_index][0]
        bits_str = f"Q{bits_per_weight}" if bits_per_weight < 16 else f"FP{bits_per_weight}"

        lines.append("Memory Breakdown:")
        lines.append(f"  Context: {selected_ctx_label} | Quantization: {bits_str}")
        lines.append(f"  Model Size:    {model_size_gb:.2f} GB")
        lines.append(f"  Context Size:  {context_size_gb:.2f} GB")
        lines.append(f"  Total Needed:  {total_needed:.2f} GB")
        lines.append("")

        # VRAM visualization
        vram_pct = (vram_used / self.vram_total) * 100 if self.vram_total > 0 else 0
        vram_bar_len = 40
        vram_filled = int((vram_pct / 100) * vram_bar_len)
        vram_bar = "█" * min(vram_filled, vram_bar_len) + "░" * max(0, vram_bar_len - vram_filled)

        lines.append(f"VRAM ({self.vram_total}GB):")
        lines.append(f"[{vram_bar}] {vram_used:.1f}/{self.vram_total} GB ({vram_pct:.1f}%)")

        # RAM visualization
        ram_pct = (ram_used / self.ram_total) * 100 if self.ram_total > 0 else 0
        ram_bar_len = 40
        ram_filled = int((ram_pct / 100) * ram_bar_len)
        ram_bar = "█" * min(ram_filled, ram_bar_len) + "░" * max(0, ram_bar_len - ram_filled)

        lines.append("")
        lines.append(f"RAM ({self.ram_total}GB):")
        lines.append(f"[{ram_bar}] {ram_used:.1f}/{self.ram_total} GB ({ram_pct:.1f}%)")

        # Status message
        total_available = self.vram_total + self.ram_total
        if total_needed > total_available:
            shortfall = total_needed - total_available
            lines.append("")
            lines.append(f"⚠️  INSUFFICIENT MEMORY! Shortfall: {shortfall:.2f} GB")
        else:
            remaining = total_available - total_needed
            lines.append("")
            lines.append("✓ Model can run on your system")
            lines.append(f"  Memory remaining: {remaining:.2f} GB")

        # Controls
        lines.append("")
        lines.append("Controls: ↑/↓ Move | ←/→ or Tab Switch Pane | j/k and h/l also work | Ctrl+C Exit")

        return Panel(
            "\n".join(lines),
            title="LLM Memory Calculator",
            border_style="blue"
        )
    
    def run(self):
        """Main TUI loop."""
        # Get system specs
        self.console.print(Panel(
            "[bold blue]LLM VRAM/RAM Calculator[/bold blue]\n\n"
            "This tool helps you determine if your system can run a specific LLM."
        ))
        
        # Input loop for model URL
        while True:
            hf_url = input("\nEnter HuggingFace model URL (e.g., unsloth/Qwen3.5-4B-GGUF): ").strip()
            if hf_url:
                try:
                    base_model, gguf_repo = self.extract_base_model(hf_url)
                    break
                except ValueError as e:
                    self.console.print(f"[red]Error: {e}[/red]")
        
        # Fetch model config with progress indicator
        self.console.print("Fetching model configuration...")
        try:
            self.model_config = self.fetch_model_config(base_model)
        except Exception:
            self.console.print("[red]Failed to fetch model config. Please check the URL and try again.[/red]")
            return
        
        # Fetch quantizations
        self.console.print("Fetching available quantizations...")
        quantizations = self.fetch_gguf_quantizations(gguf_repo)
        for filename, bits in quantizations:
            self.model_config.add_quantization(filename, bits)
        
        if not self.model_config.quantizations:
            self.console.print("[red]No GGUF files found. This might not be a GGUF repository.[/red]")
            return
        
        # Get available contexts
        available_contexts = self.get_available_contexts(self.model_config.max_context_length)
        
        # Set defaults
        self.selected_context_tokens = available_contexts[0][1] if available_contexts else 4096
        self.selected_quant_index = 0
        self.selected_context_index = 0
        self.current_menu = "quantization" if len(available_contexts) <= 1 else "context"
        
        # Check if running in a proper terminal for TUI
        if not sys.stdin.isatty():
            self.console.print("[yellow]Warning: Not running in an interactive terminal. Showing static view.[/yellow]")
            self._show_static_view(available_contexts)
            return
        
        try:
            self._render_tui(available_contexts)
        except KeyboardInterrupt:
            self.console.print("\n[bold yellow]\nExiting...[/bold yellow]")
    
    def _render_tui(self, available_contexts: List[Tuple[str, int]]):
        """Render the TUI with real-time updates."""
        sorted_quantizations = self.model_config.sorted_quantizations

        with self._terminal_input_mode():
            with Live(
                self._build_panel(available_contexts, sorted_quantizations),
                refresh_per_second=30,
                auto_refresh=False,
                redirect_stdout=False,
                redirect_stderr=False,
            ) as live:
                while True:
                    should_continue = self._handle_input(available_contexts, sorted_quantizations)
                    if not should_continue:
                        break
                    live.update(self._build_panel(available_contexts, sorted_quantizations), refresh=True)
                    time.sleep(0.02)
    
    def _handle_input(self, available_contexts: List[Tuple[str, int]], 
                      sorted_quantizations: List[Tuple[str, int]]) -> bool:
        """
        Handle keyboard input for navigation.
        """
        key = self._read_key()

        if key in (None, ''):
            return True

        # Handle input
        if key in ('\n', '\r'):  # Enter - switch menus
            self.current_menu = "quantization" if self.current_menu == "context" else "context"
        elif key in ('tab', 'backtab'):
            self.current_menu = "quantization" if self.current_menu == "context" else "context"
        elif key in ('up', 'k'):  # Up arrow or k (vim style)
            if self.current_menu == "context":
                self.selected_context_index = max(0, self.selected_context_index - 1)
            else:
                self.selected_quant_index = max(0, self.selected_quant_index - 1)
        elif key in ('down', 'j'):  # Down arrow or j (vim style)
            if self.current_menu == "context":
                self.selected_context_index = min(len(available_contexts) - 1, self.selected_context_index + 1)
            else:
                self.selected_quant_index = min(len(sorted_quantizations) - 1, self.selected_quant_index + 1)
        elif key in ('left', 'h'):  # Left arrow or h
            self.current_menu = "context"
        elif key in ('right', 'l'):  # Right arrow or l
            self.current_menu = "quantization"

        return True


def main():
    """Entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="LLM VRAM/RAM Calculator")
    parser.add_argument("--vram", type=float, default=12.0, help="VRAM in GB (default: 12)")
    parser.add_argument("--ram", type=float, default=64.0, help="RAM in GB (default: 64)")
    args = parser.parse_args()
    
    calculator = LLMCalculator(vram_gb=args.vram, ram_gb=args.ram)
    calculator.run()


if __name__ == "__main__":
    main()
