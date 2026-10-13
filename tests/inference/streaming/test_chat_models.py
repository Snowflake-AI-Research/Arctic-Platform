"""CHAT_MODELS: chat parsers by architecture, and the tie to what Arctic can train."""

import ast
import re
from dataclasses import replace
from pathlib import Path

import pytest

from cpu_support import load_library

load_library()
from arctic_platform.inference.server.chat import (
    CHAT_MODELS,
    REASONING_EFFORTS,
    ChatInputError,
    ChatModel,
    resolve_chat_model,
)

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
    "generic_moe": {
        "Qwen3MoeForCausalLM",
        "Glm4MoeForCausalLM",
        "MiniMaxM2ForCausalLM",
        "AfmoeForCausalLM",
        "NemotronHForCausalLM",
    },
    "glm5_next": {"Glm5NextForConditionalGeneration"},
    "qwen4_exp": {"Qwen4ExpForConditionalGeneration"},
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
        # vLLM 0.31 recipes and docs pair these parsers with each family.
        ("Glm4MoeForCausalLM", "glm47", "glm47", True),
        ("Glm5NextForConditionalGeneration", "glm47", "glm47", False),
        ("Glm5NextForCausalLM", "glm47", "glm47", False),
        ("Qwen4ExpForConditionalGeneration", "qwen3", "qwen3_coder", True),
        ("Qwen4ExpForCausalLM", "qwen3", "qwen3_coder", True),
        ("MiniMaxM2ForCausalLM", "minimax_m2", "minimax_m2", False),
        # Trinity-Large-Preview: no thinking, hermes-style JSON tool calls.
        ("AfmoeForCausalLM", None, "hermes", True),
        ("NemotronHForCausalLM", "nemotron_v3", "qwen3_coder", True),
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
    # No reasoner, so there is no thinking to turn off.
    assert model == ChatModel(
        reasoning_parser=None, tool_call_parser="hermes", thinking_optional=True
    )
    model = resolve_chat_model("MysteryForCausalLM", reasoning_parser="deepseek_r1")
    # Nothing says this one's thinking can be turned off.
    assert model.thinking_optional is False


def test_a_family_without_a_reasoner_takes_every_reasoning_effort():
    model = CHAT_MODELS["AfmoeForCausalLM"]
    assert model.reasoning_parser is None
    # Nothing to turn off, so "none" is honoured rather than refused.
    assert [model.template_reasoning_effort(e) for e in ("none", "high")] == ["none", "high"]


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
        ("GlmMoeDsaForCausalLM", "xhigh", "max"),
        # vLLM's DeepSeek-V4 tokenizer and gpt-oss's Harmony map these themselves.
        ("DeepseekV4ForCausalLM", "medium", "medium"),
        ("GptOssForCausalLM", "low", "low"),
        ("GptOssForCausalLM", None, None),
        # Harmony takes low, medium and high only.
        ("GptOssForCausalLM", "minimal", "low"),
        ("GptOssForCausalLM", "xhigh", "high"),
        # Qwen3.8 refuses "minimal".
        ("Qwen3_5ForConditionalGeneration", "minimal", "low"),
        ("Qwen4ExpForConditionalGeneration", "high", "xhigh"),
        # GLM-5.3-Flash takes low and high, and reads anything else as Max.
        ("Glm5NextForConditionalGeneration", "minimal", "low"),
        ("Glm5NextForConditionalGeneration", "medium", "high"),
        ("Glm5NextForConditionalGeneration", "xhigh", "max"),
    ],
)
def test_reasoning_effort_reaches_the_template_as_the_family_names_it(
    architecture, requested, template
):
    assert resolve_chat_model(architecture).template_reasoning_effort(requested) == template


@pytest.mark.parametrize("architecture", sorted(CHAT_MODELS))
@pytest.mark.parametrize("effort", sorted(REASONING_EFFORTS - {"none"}))
def test_the_table_never_maps_a_level_the_template_knows(architecture, effort):
    model = replace(CHAT_MODELS[architecture], recognized_efforts=frozenset({effort}))
    assert model.template_reasoning_effort(effort) == effort


@pytest.mark.parametrize("architecture", sorted(CHAT_MODELS))
def test_every_effort_table_ends(architecture):
    # template_reasoning_effort follows the table until it reaches a level the
    # table leaves alone, so a cycle would never return.
    table = CHAT_MODELS[architecture].reasoning_efforts
    for effort in table:
        for _ in range(len(table)):
            effort = table.get(effort, effort)
        assert effort not in table


def test_glm5_3_keeps_low_where_glm5_2_maps_it_to_high():
    glm5_2 = CHAT_MODELS["GlmMoeDsaForCausalLM"]
    # Recognized as the probe finds them on vLLM 0.31's renderer.
    glm5_3 = replace(glm5_2, recognized_efforts=frozenset({"low", "high"}))
    glm5_2 = replace(glm5_2, recognized_efforts=frozenset({"high"}))
    efforts = ["minimal", "low", "medium", "high", "xhigh", "max"]
    assert [glm5_3.template_reasoning_effort(e) for e in efforts] == [
        "low", "low", "high", "high", "max", "max"
    ]
    assert [glm5_2.template_reasoning_effort(e) for e in efforts] == [
        "high", "high", "high", "high", "max", "max"
    ]


ANY = None
# Levels each family's template (or vLLM's renderer for it) accepts; ANY for
# templates that ignore the value. From the checkpoints' chat templates and
# vLLM 0.31: Qwen3.8 raises outside low/medium/xhigh; GLM-5.3-Flash reads
# anything but low/high as Max; Harmony raises outside low/medium/high
# (harmony_utils.py:68); vLLM's DeepSeek-V4 tokenizer maps every value
# (tokenizers/deepseek_v4.py:51).
TEMPLATE_EFFORTS = {
    "Qwen3ForCausalLM": ANY,
    "Qwen3MoeForCausalLM": ANY,
    "Qwen3_5ForCausalLM": {"low", "medium", "xhigh"},
    "Qwen3_5ForConditionalGeneration": {"low", "medium", "xhigh"},
    "Qwen3_5MoeForCausalLM": {"low", "medium", "xhigh"},
    "Qwen3_5MoeForConditionalGeneration": {"low", "medium", "xhigh"},
    "Qwen4ExpForCausalLM": {"low", "medium", "xhigh"},
    "Qwen4ExpForConditionalGeneration": {"low", "medium", "xhigh"},
    # GLM-5.2's; GLM-5.3's adds low, which the probe finds.
    "GlmMoeDsaForCausalLM": {"high", "max"},
    "Glm4MoeForCausalLM": ANY,
    "Glm5NextForCausalLM": {"low", "high", "max"},
    "Glm5NextForConditionalGeneration": {"low", "high", "max"},
    "DeepseekV4ForCausalLM": ANY,
    "DeepseekV4ForConditionalGeneration": ANY,
    "GptOssForCausalLM": {"low", "medium", "high"},
    "MiniMaxM2ForCausalLM": ANY,
    "AfmoeForCausalLM": ANY,
    "NemotronHForCausalLM": ANY,
}


def test_every_chat_model_lists_the_efforts_its_template_takes():
    assert TEMPLATE_EFFORTS.keys() == CHAT_MODELS.keys()


@pytest.mark.parametrize("architecture", sorted(CHAT_MODELS))
@pytest.mark.parametrize("effort", sorted(REASONING_EFFORTS - {"none"}))
def test_every_accepted_effort_reaches_a_level_the_template_takes(architecture, effort):
    accepted = TEMPLATE_EFFORTS[architecture]
    level = CHAT_MODELS[architecture].template_reasoning_effort(effort)
    assert accepted is ANY or level in accepted


@pytest.mark.parametrize("architecture", sorted(CHAT_MODELS))
def test_none_turns_thinking_off_or_is_refused(architecture):
    model = CHAT_MODELS[architecture]
    if model.thinking_optional:
        # vLLM hands the template enable_thinking=False.
        assert model.template_reasoning_effort("none") == "none"
    else:
        # Thinking can't be turned off, so "none" would silently think.
        with pytest.raises(ChatInputError) as raised:
            model.template_reasoning_effort("none")
        assert (raised.value.code, raised.value.param) == (
            "invalid_chat_request",
            "reasoning_effort",
        )


def test_chat_parser_overrides_are_model_config_fields():
    from arctic_platform.inference.server.config import ModelConfig

    kwargs = ModelConfig(
        model="m", tool_call_parser="hermes", chat_reasoning_parser="deepseek_r1"
    ).to_engine_kwargs()
    assert (kwargs["tool_call_parser"], kwargs["chat_reasoning_parser"]) == ("hermes", "deepseek_r1")
    # Unset, they reach the worker as nothing, so the table decides.
    assert not {"tool_call_parser", "chat_reasoning_parser"} & ModelConfig(model="m").to_engine_kwargs().keys()
