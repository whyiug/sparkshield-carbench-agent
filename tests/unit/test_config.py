import pytest
from pydantic import SecretStr

from track_1_agent_under_test.car_guard.config import (
    AgentConfig,
    SUPPORTED_PROVIDER_PROFILES,
    provider_supports_cache_control,
)


def test_default_step_budget_reserves_the_fiftieth_evaluator_step() -> None:
    assert AgentConfig().soft_max_steps == 49
    assert AgentConfig.from_env({}).soft_max_steps == 49


def test_cli_explicit_wins_over_env_and_preserves_zero_temperature() -> None:
    config = AgentConfig.from_env(
        {
            "AGENT_LLM": "env-model",
            "AGENT_TEMPERATURE": "0.7",
            "AGENT_REASONING_EFFORT": "high",
        }
    ).with_cli_overrides(llm="cli-model", temperature=0.0)

    assert config.llm == "cli-model"
    assert config.temperature == 0.0
    assert config.reasoning_effort == "high"


def test_structured_output_mode_is_validated_and_normalized_from_env() -> None:
    config = AgentConfig.from_env({"AGENT_STRUCTURED_OUTPUT_MODE": "JSON_OBJECT"})

    assert config.structured_output_mode == "json_object"
    with pytest.raises(ValueError):
        AgentConfig.from_env({"AGENT_STRUCTURED_OUTPUT_MODE": "best_effort"})


def test_empty_optional_environment_values_are_unset() -> None:
    config = AgentConfig.from_env(
        {
            "AGENT_LLM": "test-model",
            "AGENT_API_KEY": "",
            "AGENT_API_BASE": "  ",
            "AGENT_TEMPERATURE": "",
            "AGENT_CRITIC_MODEL": "",
        }
    )

    assert config.api_key is None
    assert config.api_base is None
    assert config.temperature is None
    assert config.critic_model is None


def test_secret_is_not_exposed_by_repr() -> None:
    config = AgentConfig(
        llm="test",
        api_key=SecretStr("primary-sensitive"),
        critic_model="critic",
        critic_provider="openai",
        critic_api_base="https://critic.example/v1",
        critic_api_key=SecretStr("critic-sensitive"),
    )

    assert "primary-sensitive" not in repr(config)
    assert "critic-sensitive" not in repr(config)
    assert "primary-sensitive" not in config.model_dump_json()
    assert "critic-sensitive" not in config.model_dump_json()


def test_unsupported_temperature_and_cache_hints_are_omitted() -> None:
    config = AgentConfig(
        llm="claude-opus-4-6",
        provider="anthropic",
        temperature=0.0,
    )

    assert "temperature" not in config.completion_options()
    assert provider_supports_cache_control(config.llm, config.provider)


def test_generic_provider_receives_explicit_temperature_only() -> None:
    unset = AgentConfig(llm="provider/model")
    explicit = AgentConfig(llm="provider/model", temperature=0.0)

    assert "temperature" not in unset.completion_options()
    assert explicit.completion_options()["temperature"] == 0.0


@pytest.mark.parametrize(
    ("profile", "prefix"), sorted(SUPPORTED_PROVIDER_PROFILES.items())
)
def test_allowlisted_provider_profile_resolves_openai_compatible_route(
    profile: str, prefix: str
) -> None:
    config = AgentConfig.from_env(
        {
            "AGENT_PROVIDER_PROFILE": profile,
            f"{prefix}_MODEL_NAME": "profile-model",
            f"{prefix}_API_KEY": "profile-secret",
            f"{prefix}_BASE_URL": "https://provider.example/v1",
        }
    )

    options = config.completion_options()
    assert config.llm == "openai/profile-model"
    assert options["custom_llm_provider"] == "openai"
    assert options["api_base"] == "https://provider.example/v1"
    assert options["api_key"] == "profile-secret"
    assert "profile-secret" not in repr(config)


def test_profile_prefix_is_accepted_as_selector_alias() -> None:
    config = AgentConfig.from_env(
        {
            "AGENT_PROVIDER_PROFILE": "XOPDSV32IN",
            "XOPDSV32IN_MODEL_NAME": "profile-model",
            "XOPDSV32IN_API_KEY": "profile-secret",
            "XOPDSV32IN_BASE_URL": "https://provider.example/v1",
        }
    )

    assert config.llm == "openai/profile-model"


def test_explicit_agent_route_takes_precedence_over_selected_profile() -> None:
    config = AgentConfig.from_env(
        {
            "AGENT_PROVIDER_PROFILE": "xopdsv32in",
            "XOPDSV32IN_MODEL_NAME": "profile-model",
            "XOPDSV32IN_API_KEY": "profile-secret",
            "XOPDSV32IN_BASE_URL": "https://profile.example/v1",
            "AGENT_LLM": "openai/explicit-model",
            "AGENT_PROVIDER": "openai",
            "AGENT_API_KEY": "explicit-secret",
            "AGENT_API_BASE": "https://explicit.example/v1",
        }
    )

    options = config.completion_options()
    assert options["model"] == "openai/explicit-model"
    assert options["api_key"] == "explicit-secret"
    assert options["api_base"] == "https://explicit.example/v1"


@pytest.mark.parametrize(
    "single_override",
    ["AGENT_LLM", "AGENT_PROVIDER", "AGENT_API_BASE", "AGENT_API_KEY"],
)
def test_selected_profile_rejects_partial_direct_route_override(
    single_override: str,
) -> None:
    override_values = {
        "AGENT_LLM": "openai/other-model",
        "AGENT_PROVIDER": "openai",
        "AGENT_API_BASE": "https://other.example/v1",
        "AGENT_API_KEY": "other-secret",
    }
    values = {
        "AGENT_PROVIDER_PROFILE": "xopdsv32in",
        "XOPDSV32IN_MODEL_NAME": "profile-model",
        "XOPDSV32IN_API_KEY": "profile-secret",
        "XOPDSV32IN_BASE_URL": "https://profile.example/v1",
        single_override: override_values[single_override],
    }

    with pytest.raises(ValueError, match="must override.*atomically"):
        AgentConfig.from_env(values)


def test_critic_profile_uses_independent_route_and_secret() -> None:
    config = AgentConfig.from_env(
        {
            "AGENT_LLM": "openai/planner",
            "AGENT_PROVIDER": "openai",
            "AGENT_API_KEY": "planner-secret",
            "AGENT_API_BASE": "https://planner.example/v1",
            "AGENT_CRITIC_PROVIDER_PROFILE": "doubao-seed-2-0-lite-260215",
            "DOUBAO_SEED_2_0_LITE_260215_MODEL_NAME": "critic",
            "DOUBAO_SEED_2_0_LITE_260215_API_KEY": "critic-secret",
            "DOUBAO_SEED_2_0_LITE_260215_BASE_URL": ("https://critic.example/api/v3"),
        }
    )

    planner = config.completion_options()
    critic = config.completion_options(critic=True)
    assert planner["model"] == "openai/planner"
    assert planner["api_key"] == "planner-secret"
    assert critic["model"] == "openai/critic"
    assert critic["custom_llm_provider"] == "openai"
    assert critic["api_base"] == "https://critic.example/api/v3"
    assert critic["api_key"] == "critic-secret"


def test_partial_independent_critic_route_cannot_reuse_primary_secret() -> None:
    with pytest.raises(ValueError, match="independent.*route must be complete"):
        AgentConfig.from_env(
            {
                "AGENT_LLM": "openai/planner",
                "AGENT_PROVIDER": "openai",
                "AGENT_API_KEY": "planner-secret",
                "AGENT_API_BASE": "https://planner.example/v1",
                "AGENT_CRITIC_MODEL": "openai/critic",
                "AGENT_CRITIC_PROVIDER": "openai",
                "AGENT_CRITIC_API_BASE": "https://critic.example/v1",
            }
        )

    with pytest.raises(ValueError, match="independent critic route requires"):
        AgentConfig(
            llm="openai/planner",
            provider="openai",
            api_key=SecretStr("planner-secret"),
            api_base="https://planner.example/v1",
            critic_model="openai/critic",
            critic_provider="openai",
            critic_api_base="https://critic.example/v1",
        )


def test_critic_model_only_safely_inherits_primary_route() -> None:
    config = AgentConfig.from_env(
        {
            "AGENT_LLM": "openai/planner",
            "AGENT_PROVIDER": "openai",
            "AGENT_API_KEY": "planner-secret",
            "AGENT_API_BASE": "https://planner.example/v1",
            "AGENT_CRITIC_MODEL": "openai/critic",
        }
    )

    critic = config.completion_options(critic=True)
    assert critic["model"] == "openai/critic"
    assert critic["api_key"] == "planner-secret"
    assert critic["api_base"] == "https://planner.example/v1"


def test_unknown_or_incomplete_profile_is_rejected_without_secret_value() -> None:
    with pytest.raises(ValueError, match="unsupported provider profile"):
        AgentConfig.from_env({"AGENT_PROVIDER_PROFILE": "untrusted-prefix"})

    with pytest.raises(ValueError, match="profile is incomplete") as error:
        AgentConfig.from_env(
            {
                "AGENT_PROVIDER_PROFILE": "xopdsv32in",
                "XOPDSV32IN_MODEL_NAME": "profile-model",
                "XOPDSV32IN_API_KEY": "must-not-appear",
            }
        )
    assert "must-not-appear" not in str(error.value)


def test_remote_http_api_base_is_rejected_but_loopback_http_is_allowed() -> None:
    with pytest.raises(ValueError, match="remote API base must use HTTPS"):
        AgentConfig(
            llm="openai/model",
            api_base="http://203.0.113.10:3006/v1",
            api_key=SecretStr("must-not-appear"),
        )

    config = AgentConfig(
        llm="openai/local-model",
        api_base="http://127.0.0.1:8000/v1",
    )
    assert config.api_base == "http://127.0.0.1:8000/v1"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider", "anthropic"),
        ("api_base", "https://other.example/v1"),
    ],
)
def test_cli_override_cannot_mutate_credential_route(field: str, value: str) -> None:
    config = AgentConfig(
        llm="openai/model",
        provider="openai",
        api_base="https://provider.example/v1",
        api_key=SecretStr("profile-secret"),
    )

    with pytest.raises(ValueError, match="configure.*route atomically"):
        config.with_cli_overrides(**{field: value})


def test_cli_accepts_unchanged_route_values() -> None:
    config = AgentConfig(
        llm="openai/model",
        provider="openai",
        api_base="https://provider.example/v1",
    )

    overridden = config.with_cli_overrides(
        provider="openai",
        api_base="https://provider.example/v1",
    )

    assert overridden == config
