"""CHAT_MODELS: chat parsers by architecture, and the tie to what Arctic can train."""

import ast
import re
from pathlib import Path

import pytest

from cpu_support import load_library

load_library()
from arctic_platform.inference.server.chat import CHAT_MODELS, ChatModel, resolve_chat_model

MODEL_PACKAGE = Path(__file__).resolve().parents[3] / "arctic_platform" / "model"

# Architectures (as vLLM resolves them from the checkpoint's config) that each
# training loader builds. The default loader takes any HuggingFace causal LM;
# these are the ones Arctic trains through it.
LOADER_ARCHITECTURES = {
    "huggingface": {
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5ForCausalLM",
    },
    "qwen3_5_moe": {"Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeForCausalLM"},
    "glm_moe_dsa": {"GlmMoeDsaForCausalLM"},
}
# Trainable architectures deliberately left without chat mode, with the reason.
TRAINED_WITHOUT_CHAT: dict[str, str] = {}


def registered_loaders():
    names = set()
    for path in (MODEL_PACKAGE / "loaders").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "register_loader":
                names.add(node.args[0].value)
    return names


def implemented_architectures():
    names = set()
    for path in (MODEL_PACKAGE / "implementations").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ClassDef) and re.fullmatch(
                r"\w+For(CausalLM|ConditionalGeneration)", node.name
            ):
                names.add(node.name)
    return names


def test_every_training_loader_lists_the_architectures_it_trains():
    assert registered_loaders() == set(LOADER_ARCHITECTURES), (
        "A training loader was added or removed: list the architectures it "
        "trains in LOADER_ARCHITECTURES"
    )


def test_every_implemented_model_belongs_to_a_loader():
    trained = set().union(*LOADER_ARCHITECTURES.values())
    assert implemented_architectures() <= trained


def test_every_trainable_architecture_has_a_chat_decision():
    trained = set().union(*LOADER_ARCHITECTURES.values()) | implemented_architectures()
    undecided = trained - CHAT_MODELS.keys() - TRAINED_WITHOUT_CHAT.keys()
    assert not undecided, (
        f"Add {sorted(undecided)} to CHAT_MODELS, or to TRAINED_WITHOUT_CHAT with a reason"
    )
    assert not CHAT_MODELS.keys() & TRAINED_WITHOUT_CHAT.keys()


@pytest.mark.parametrize(
    "architecture,reasoning_parser,tool_call_parser,thinking_optional",
    [
        ("Qwen3ForCausalLM", "qwen3", "hermes", True),
        ("Qwen3MoeForCausalLM", "qwen3", "hermes", True),
        ("Qwen3_5ForConditionalGeneration", "qwen3", "qwen3_coder", True),
        ("Qwen3_5MoeForConditionalGeneration", "qwen3", "qwen3_coder", True),
        ("GlmMoeDsaForCausalLM", "glm47", "glm47", True),
        ("DeepseekV4ForCausalLM", "deepseek_v4", "deepseek_v4", True),
        ("DeepseekV4ForConditionalGeneration", "deepseek_v4", "deepseek_v4", True),
        ("GptOssForCausalLM", "openai_gptoss", "openai", False),
    ],
)
def test_catalog_architectures_get_their_parsers(
    architecture, reasoning_parser, tool_call_parser, thinking_optional
):
    model = resolve_chat_model(architecture)
    assert (model.reasoning_parser, model.tool_call_parser, model.thinking_optional) == (
        reasoning_parser,
        tool_call_parser,
        thinking_optional,
    )


def test_explicit_parsers_override_the_table():
    model = resolve_chat_model(
        "Qwen3ForCausalLM", reasoning_parser="deepseek_r1", tool_call_parser="qwen3_xml"
    )
    assert (model.reasoning_parser, model.tool_call_parser) == ("deepseek_r1", "qwen3_xml")
    assert model.thinking_optional is True
    only_tools = resolve_chat_model("Qwen3ForCausalLM", tool_call_parser="qwen3_xml")
    assert only_tools.reasoning_parser == "qwen3"


def test_unknown_architecture_has_no_chat_unless_parsers_are_given():
    assert resolve_chat_model("MysteryForCausalLM") is None
    model = resolve_chat_model("MysteryForCausalLM", tool_call_parser="hermes")
    assert model == ChatModel(reasoning_parser=None, tool_call_parser="hermes")
    # Nothing says thinking can be turned off.
    assert model.thinking_optional is False


@pytest.mark.parametrize(
    "architecture,requested,template",
    [
        # Qwen3's template ignores the value; vLLM turns "none" into enable_thinking=False.
        ("Qwen3ForCausalLM", "high", "high"),
        ("Qwen3ForCausalLM", "none", "none"),
        # Qwen3.8's template takes low, medium and xhigh; Qwen3.5's ignores it.
        ("Qwen3_5ForConditionalGeneration", "high", "xhigh"),
        ("Qwen3_5ForConditionalGeneration", "medium", "medium"),
        ("Qwen3_5MoeForConditionalGeneration", "high", "xhigh"),
        # GLM-5's template has High and Max, and reads anything but "high" as Max.
        ("GlmMoeDsaForCausalLM", "low", "high"),
        ("GlmMoeDsaForCausalLM", "medium", "high"),
        ("GlmMoeDsaForCausalLM", "high", "high"),
        ("GlmMoeDsaForCausalLM", "max", "max"),
        ("GlmMoeDsaForCausalLM", "none", "none"),
        # vLLM's DeepSeek-V4 tokenizer and gpt-oss's Harmony map these themselves.
        ("DeepseekV4ForCausalLM", "medium", "medium"),
        ("GptOssForCausalLM", "low", "low"),
        ("GptOssForCausalLM", None, None),
    ],
)
def test_reasoning_effort_reaches_the_template_as_the_family_names_it(
    architecture, requested, template
):
    assert resolve_chat_model(architecture).template_reasoning_effort(requested) == template
