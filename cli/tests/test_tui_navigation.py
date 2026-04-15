import json
import os
import tempfile
import unittest
from unittest.mock import patch

from cli.llm_calculator import LLMCalculator, ModelConfig


class FakeTTY:
    def __init__(self, data: str):
        self.buffer = data.encode()

    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        return 0

    def read(self, size: int = 1) -> str:
        if not self.buffer:
            return ""
        chunk = self.buffer[:size].decode()
        self.buffer = self.buffer[size:]
        return chunk

    def pop_bytes(self, size: int) -> bytes:
        if not self.buffer:
            return b""
        chunk = self.buffer[:size]
        self.buffer = self.buffer[size:]
        return chunk


class TuiNavigationTests(unittest.TestCase):
    def setUp(self):
        self.calculator = LLMCalculator(vram_gb=12.0, ram_gb=64.0)
        self.calculator.model_config = ModelConfig(
            name="test/model",
            num_layers=32,
            num_heads=32,
            head_dim=128,
            max_context_length=16384,
            total_params=7.0,
        )
        self.calculator.model_config.add_quantization("model-Q8_0.gguf", 8)
        self.calculator.model_config.add_quantization("model-Q4_K_M.gguf", 4)
        self.calculator.selected_context_index = 0
        self.calculator.selected_quant_index = 0
        self.calculator.current_menu = "context"
        self.available_contexts = self.calculator.get_available_contexts(
            self.calculator.model_config.max_context_length
        )
        self.sorted_quantizations = self.calculator.model_config.sorted_quantizations

    def test_unix_escape_sequence_maps_to_arrow_key(self):
        fake_stdin = FakeTTY("\x1b[A")

        with patch("sys.platform", "linux"), patch("sys.stdin", fake_stdin), patch(
            "os.read",
            side_effect=lambda fd, size: fake_stdin.pop_bytes(size),
        ), patch(
            "select.select",
            side_effect=lambda read, write, error, timeout: ([0], [], []) if fake_stdin.buffer else ([], [], []),
        ):
            self.assertEqual(self.calculator._read_key(), "up")

    def test_unix_ss3_escape_sequence_maps_to_arrow_key(self):
        fake_stdin = FakeTTY("\x1bOB")

        with patch("sys.platform", "linux"), patch("sys.stdin", fake_stdin), patch(
            "os.read",
            side_effect=lambda fd, size: fake_stdin.pop_bytes(size),
        ), patch(
            "select.select",
            side_effect=lambda read, write, error, timeout: ([0], [], []) if fake_stdin.buffer else ([], [], []),
        ):
            self.assertEqual(self.calculator._read_key(), "down")

    def test_modifier_escape_sequence_maps_to_arrow_key(self):
        self.assertEqual(self.calculator._normalize_escape_sequence("\x1b[1;2C"), "right")

    def test_bare_escape_maps_to_escape(self):
        fake_stdin = FakeTTY("\x1b")

        with patch("sys.platform", "linux"), patch("sys.stdin", fake_stdin), patch(
            "os.read",
            side_effect=lambda fd, size: fake_stdin.pop_bytes(size),
        ), patch(
            "select.select",
            side_effect=lambda read, write, error, timeout: ([0], [], []) if fake_stdin.buffer else ([], [], []),
        ):
            self.assertEqual(self.calculator._read_key(), "escape")

    def test_tab_maps_to_tab(self):
        fake_stdin = FakeTTY("\t")

        with patch("sys.platform", "linux"), patch("sys.stdin", fake_stdin), patch(
            "os.read",
            side_effect=lambda fd, size: fake_stdin.pop_bytes(size),
        ), patch(
            "select.select",
            side_effect=lambda read, write, error, timeout: ([0], [], []) if fake_stdin.buffer else ([], [], []),
        ):
            self.assertEqual(self.calculator._read_key(), "tab")

    def test_down_moves_context_selection(self):
        self.calculator._read_key = lambda: "down"

        self.calculator._handle_input(self.available_contexts, self.sorted_quantizations)

        self.assertEqual(self.calculator.selected_context_index, 1)

    def test_right_then_down_moves_quantization_selection(self):
        self.calculator._read_key = lambda: "right"
        self.calculator._handle_input(self.available_contexts, self.sorted_quantizations)
        self.assertEqual(self.calculator.current_menu, "quantization")

        self.calculator._read_key = lambda: "down"
        self.calculator._handle_input(self.available_contexts, self.sorted_quantizations)

        self.assertEqual(self.calculator.selected_quant_index, 1)

    def test_escape_does_not_exit_loop(self):
        self.calculator._read_key = lambda: "escape"

        should_continue = self.calculator._handle_input(
            self.available_contexts, self.sorted_quantizations
        )

        self.assertTrue(should_continue)

    def test_tab_switches_menu(self):
        self.calculator._read_key = lambda: "tab"

        self.calculator._handle_input(self.available_contexts, self.sorted_quantizations)

        self.assertEqual(self.calculator.current_menu, "quantization")

    def test_single_context_defaults_to_quantization_menu(self):
        single_context_calculator = LLMCalculator(vram_gb=12.0, ram_gb=64.0)
        single_context_calculator.model_config = ModelConfig(
            name="test/model",
            num_layers=32,
            num_heads=32,
            head_dim=128,
            max_context_length=4096,
            total_params=7.0,
        )
        single_context_calculator.model_config.add_quantization("model-Q8_0.gguf", 8)
        single_context_calculator.model_config.add_quantization("model-Q4_K_M.gguf", 4)

        available_contexts = single_context_calculator.get_available_contexts(
            single_context_calculator.model_config.max_context_length
        )
        single_context_calculator.selected_context_tokens = available_contexts[0][1]
        single_context_calculator.selected_quant_index = 0
        single_context_calculator.selected_context_index = 0
        single_context_calculator.current_menu = (
            "quantization" if len(available_contexts) <= 1 else "context"
        )

        self.assertEqual(single_context_calculator.current_menu, "quantization")

    def test_fetch_model_config_prefers_nested_text_config(self):
        config_payload = {
            "architectures": ["Qwen3ForConditionalGeneration"],
            "model_name": "qwen/Qwen3.5-27B",
            "text_config": {
                "num_hidden_layers": 36,
                "num_attention_heads": 32,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "max_position_embeddings": 262144,
                "vocab_size": 151936,
            },
        }

        with tempfile.NamedTemporaryFile("w", delete=False) as config_file:
            json.dump(config_payload, config_file)
            config_path = config_file.name

        try:
            with patch("cli.llm_calculator.hf_hub_download", return_value=config_path):
                model_config = self.calculator.fetch_model_config("Qwen/Qwen3.5-4B-Base")
        finally:
            os.unlink(config_path)

        self.assertEqual(model_config.num_layers, 36)
        self.assertEqual(model_config.num_heads, 4)
        self.assertEqual(model_config.head_dim, 128)
        self.assertEqual(model_config.max_context_length, 262144)
        self.assertEqual(model_config.total_params, 27.0)

    def test_extract_param_count_from_identifiers_uses_largest_b_match(self):
        self.assertEqual(
            self.calculator._extract_param_count_from_identifiers(
                "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled",
                "Qwen/Qwen3.5-27B",
            ),
            27.0,
        )


if __name__ == "__main__":
    unittest.main()
