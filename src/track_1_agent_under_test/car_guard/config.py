"""Environment-driven CAR-Guard runtime configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)


SUPPORTED_PROVIDER_PROFILES: dict[str, str] = {
    "xopdsv32in": "XOPDSV32IN",
    "xop3qwen235b": "XOP3QWEN235B",
    "doubao-seed-2-0-lite-260215": "DOUBAO_SEED_2_0_LITE_260215",
    "gemini-2.5-pro": "GEMINI_2_5_PRO",
}
StructuredOutputMode = Literal["auto", "json_schema", "json_object"]


@dataclass(frozen=True, slots=True)
class _ProviderRoute:
    model: str
    provider: str
    api_base: str
    api_key: SecretStr


def _optional(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _float(value: str | float | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _structured_output_mode(value: str | None) -> StructuredOutputMode:
    normalized = (_optional(value) or "auto").lower()
    if normalized not in {"auto", "json_schema", "json_object"}:
        raise ValueError(
            "structured output mode must be auto, json_schema, or json_object"
        )
    return cast(StructuredOutputMode, normalized)


def _profile_name(value: str | None) -> str | None:
    selected = _optional(value)
    if selected is None:
        return None
    normalized = selected.lower()
    aliases = {
        prefix.lower(): name for name, prefix in SUPPORTED_PROVIDER_PROFILES.items()
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_PROVIDER_PROFILES:
        allowed = ", ".join(sorted(SUPPORTED_PROVIDER_PROFILES))
        raise ValueError(f"unsupported provider profile; choose one of: {allowed}")
    return normalized


def _profile_route(
    values: Mapping[str, str], selector_name: str
) -> _ProviderRoute | None:
    name = _profile_name(values.get(selector_name))
    if name is None:
        return None
    prefix = SUPPORTED_PROVIDER_PROFILES[name]
    fields = {
        "model": _optional(values.get(f"{prefix}_MODEL_NAME")),
        "api_key": _optional(values.get(f"{prefix}_API_KEY")),
        "api_base": _optional(values.get(f"{prefix}_BASE_URL")),
    }
    missing = [field for field, value in fields.items() if value is None]
    if missing:
        raise ValueError(
            f"{selector_name} profile is incomplete; missing: {', '.join(missing)}"
        )
    model = str(fields["model"])
    if "/" not in model:
        model = f"openai/{model}"
    return _ProviderRoute(
        model=model,
        provider="openai",
        api_base=str(fields["api_base"]),
        api_key=SecretStr(str(fields["api_key"])),
    )


def _secure_api_base(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("API base must not contain embedded credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("API base must not contain a query or fragment")
    if parsed.scheme == "https" and parsed.hostname:
        return value
    if parsed.scheme == "http" and parsed.hostname in {
        "127.0.0.1",
        "::1",
        "localhost",
    }:
        return value
    raise ValueError("remote API base must use HTTPS")


def provider_supports_temperature(model: str, provider: str | None = None) -> bool:
    """Return whether temperature is safe to send for the configured route."""

    route = f"{provider or ''}/{model}".lower()
    adaptive_markers = ("claude-opus-4-6", "gpt-5.5", "o1", "o3", "o4")
    return not any(marker in route for marker in adaptive_markers)


def provider_supports_cache_control(model: str, provider: str | None = None) -> bool:
    """Only enable prompt cache hints for routes known to support them."""

    route = f"{provider or ''}/{model}".lower()
    return "anthropic" in route or "claude" in route


class AgentConfig(BaseModel):
    """Validated runtime settings with secrets excluded from representations."""

    model_config = ConfigDict(extra="forbid")

    llm: str = "gemini/gemini-2.5-flash"
    provider: str | None = None
    api_key: SecretStr | None = Field(default=None, repr=False)
    api_base: str | None = None
    deployment: str | None = None
    service_tier: str | None = None
    reasoning_effort: str | None = None
    temperature: float | None = None
    structured_output_mode: StructuredOutputMode = "auto"
    thinking: bool = False
    interleaved_thinking: bool = False
    enable_critic: bool = True
    critic_model: str | None = None
    critic_provider: str | None = None
    critic_api_key: SecretStr | None = Field(default=None, repr=False)
    critic_api_base: str | None = None
    timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    max_retries: int = Field(default=2, ge=0, le=5)
    soft_max_steps: int = Field(default=49, ge=1, le=49)
    max_user_turns: int = Field(default=12, ge=1, le=50)
    session_ttl_seconds: int = Field(default=1800, ge=30, le=86400)
    max_sessions: int = Field(default=256, ge=1, le=4096)

    @field_validator(
        "provider",
        "api_base",
        "deployment",
        "service_tier",
        "reasoning_effort",
        "critic_model",
        "critic_provider",
        "critic_api_base",
        mode="before",
    )
    @classmethod
    def normalize_optional_strings(cls, value: Any) -> Any:
        return _optional(value) if isinstance(value, str) or value is None else value

    @field_validator("api_base", "critic_api_base")
    @classmethod
    def require_secure_api_base(cls, value: str | None) -> str | None:
        return _secure_api_base(value) if value is not None else None

    @model_validator(mode="after")
    def require_atomic_independent_critic_route(self) -> "AgentConfig":
        route = (
            self.critic_model,
            self.critic_provider,
            self.critic_api_base,
            self.critic_api_key,
        )
        if any(value is not None for value in route[1:]) and not all(
            value is not None for value in route
        ):
            raise ValueError(
                "an independent critic route requires model, provider, base, and key"
            )
        return self

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "AgentConfig":
        values = os.environ if env is None else env
        primary_profile = _profile_route(values, "AGENT_PROVIDER_PROFILE")
        critic_profile = _profile_route(values, "AGENT_CRITIC_PROVIDER_PROFILE")
        direct_primary = {
            "model": _optional(values.get("AGENT_LLM")),
            "provider": _optional(values.get("AGENT_PROVIDER")),
            "api_base": _optional(values.get("AGENT_API_BASE")),
            "api_key": _optional(values.get("AGENT_API_KEY")),
        }
        if primary_profile is not None and any(direct_primary.values()):
            missing = [name for name, value in direct_primary.items() if value is None]
            if missing:
                raise ValueError(
                    "direct AGENT_* route must override a selected profile "
                    f"atomically; missing: {', '.join(missing)}"
                )
            primary_profile = _ProviderRoute(
                model=str(direct_primary["model"]),
                provider=str(direct_primary["provider"]),
                api_base=str(direct_primary["api_base"]),
                api_key=SecretStr(str(direct_primary["api_key"])),
            )

        direct_critic = {
            "model": _optional(values.get("AGENT_CRITIC_MODEL")),
            "provider": _optional(values.get("AGENT_CRITIC_PROVIDER")),
            "api_base": _optional(values.get("AGENT_CRITIC_API_BASE")),
            "api_key": _optional(values.get("AGENT_CRITIC_API_KEY")),
        }
        if critic_profile is not None and any(direct_critic.values()):
            missing = [name for name, value in direct_critic.items() if value is None]
            if missing:
                raise ValueError(
                    "direct AGENT_CRITIC_* route must override a selected profile "
                    f"atomically; missing: {', '.join(missing)}"
                )
            critic_profile = _ProviderRoute(
                model=str(direct_critic["model"]),
                provider=str(direct_critic["provider"]),
                api_base=str(direct_critic["api_base"]),
                api_key=SecretStr(str(direct_critic["api_key"])),
            )
        elif critic_profile is None and any(
            direct_critic[name] is not None
            for name in ("provider", "api_base", "api_key")
        ):
            missing = [name for name, value in direct_critic.items() if value is None]
            if missing:
                raise ValueError(
                    "an independent AGENT_CRITIC_* route must be complete; "
                    f"missing: {', '.join(missing)}"
                )

        key = (
            primary_profile.api_key
            if primary_profile is not None
            else (
                SecretStr(str(direct_primary["api_key"]))
                if direct_primary["api_key"] is not None
                else None
            )
        )
        critic_key = (
            critic_profile.api_key
            if critic_profile is not None
            else (
                SecretStr(str(direct_critic["api_key"]))
                if direct_critic["api_key"] is not None
                else None
            )
        )
        return cls(
            llm=(
                (primary_profile.model if primary_profile else None)
                or direct_primary["model"]
                or cls.model_fields["llm"].default
            ),
            provider=(
                (primary_profile.provider if primary_profile else None)
                or direct_primary["provider"]
            ),
            api_key=key,
            api_base=(
                (primary_profile.api_base if primary_profile else None)
                or direct_primary["api_base"]
            ),
            deployment=_optional(values.get("AGENT_DEPLOYMENT")),
            service_tier=_optional(values.get("AGENT_SERVICE_TIER")),
            reasoning_effort=_optional(values.get("AGENT_REASONING_EFFORT")),
            temperature=_float(values.get("AGENT_TEMPERATURE")),
            structured_output_mode=_structured_output_mode(
                values.get("AGENT_STRUCTURED_OUTPUT_MODE")
            ),
            thinking=_bool(values.get("AGENT_THINKING"), False),
            interleaved_thinking=_bool(values.get("AGENT_INTERLEAVED_THINKING"), False),
            enable_critic=_bool(values.get("AGENT_ENABLE_CRITIC"), True),
            critic_model=(
                (critic_profile.model if critic_profile else None)
                or direct_critic["model"]
            ),
            critic_provider=(
                (critic_profile.provider if critic_profile else None)
                or direct_critic["provider"]
            ),
            critic_api_key=critic_key,
            critic_api_base=(
                (critic_profile.api_base if critic_profile else None)
                or direct_critic["api_base"]
            ),
            timeout_seconds=float(values.get("AGENT_TIMEOUT_SECONDS", "60")),
            max_retries=int(values.get("AGENT_MAX_RETRIES", "2")),
            soft_max_steps=int(values.get("AGENT_SOFT_MAX_STEPS", "49")),
            max_user_turns=int(values.get("AGENT_MAX_USER_TURNS", "12")),
            session_ttl_seconds=int(values.get("AGENT_SESSION_TTL_SECONDS", "1800")),
            max_sessions=int(values.get("AGENT_MAX_SESSIONS", "256")),
        )

    def with_cli_overrides(self, **overrides: Any) -> "AgentConfig":
        """Apply only explicit CLI values; ``None`` means not provided."""

        explicit = {key: value for key, value in overrides.items() if value is not None}
        protected_route_fields = {
            field
            for field in ("provider", "api_base")
            if field in explicit and explicit[field] != getattr(self, field)
        }
        if protected_route_fields:
            fields = ", ".join(sorted(protected_route_fields))
            raise ValueError(
                "CLI cannot change credential route fields "
                f"({fields}); configure a complete AGENT_* route atomically"
            )
        return AgentConfig.model_validate({**self.model_dump(), **explicit})

    def completion_options(self, *, critic: bool = False) -> dict[str, Any]:
        """Build provider kwargs without sending unsupported or empty values."""

        model = self.critic_model if critic and self.critic_model else self.llm
        provider = (
            self.critic_provider if critic and self.critic_provider else self.provider
        )
        api_base = (
            self.critic_api_base if critic and self.critic_api_base else self.api_base
        )
        api_key = (
            self.critic_api_key
            if critic and self.critic_api_key is not None
            else self.api_key
        )
        options: dict[str, Any] = {
            "model": model,
            "timeout": self.timeout_seconds,
            "num_retries": self.max_retries,
        }
        optional = {
            "api_base": api_base,
            "custom_llm_provider": provider,
            "deployment_id": self.deployment,
            "service_tier": self.service_tier,
        }
        options.update({key: value for key, value in optional.items() if value})
        if api_key is not None:
            options["api_key"] = api_key.get_secret_value()
        if self.reasoning_effort:
            options["reasoning_effort"] = self.reasoning_effort
        if self.temperature is not None and provider_supports_temperature(
            model, provider
        ):
            options["temperature"] = self.temperature
        if self.thinking:
            if "claude-opus-4-6" in model.lower():
                options["thinking"] = {"type": "adaptive"}
            elif self.reasoning_effort and self.reasoning_effort.isdigit():
                options["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": int(self.reasoning_effort),
                }
        if self.interleaved_thinking and provider_supports_cache_control(
            model, provider
        ):
            options["extra_headers"] = {
                "anthropic-beta": "interleaved-thinking-2025-05-14"
            }
        return options
