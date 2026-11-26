"""Credential and base-URL resolution.

The ``/train`` cases are the ones that have actually cost people time, but the
answer changed with the unified root: the SDK now appends ``/api/mindrsi/v1``
itself, so a base URL is the relay origin (optionally with that same prefix)
and nothing else. The old plane suffixes are rejected rather than guessed at.
Every setting also has to honour its whole fallback chain of env names.
"""

from __future__ import annotations

import pytest

from mind_recursive._config import (
    DEFAULT_WEB_URL,
    ENV_API_KEYS,
    build_config,
    normalize_origin,
    resolve_api_key,
    resolve_base_urls,
    resolve_web_url,
    web_session_url,
)
from mind_recursive._errors import ConfigurationError


class TestNormalizeOrigin:
    @pytest.mark.parametrize(
        "given",
        [
            "https://mintcn.macaron.xin",
            "https://mintcn.macaron.xin/",
            "https://mintcn.macaron.xin/api/mindrsi/v1",
            "https://mintcn.macaron.xin/api/mindrsi/v1/",
            "  https://mintcn.macaron.xin/api/mindrsi/v1  ",
        ],
    )
    def test_all_accepted_spellings_reduce_to_one_origin(self, given):
        """The origin, with or without the SDK's own fixed suffix."""
        assert normalize_origin(given) == "https://mintcn.macaron.xin"

    def test_host_ending_in_train_is_not_truncated(self):
        """The host is never parsed as a path segment."""
        assert normalize_origin("https://train.example.com") == "https://train.example.com"

    def test_the_old_train_suffix_is_rejected(self):
        """Silently reinterpreting /train is exactly the confusion we removed."""
        with pytest.raises(ConfigurationError, match="base_url must be the relay origin"):
            normalize_origin("https://mintcn.macaron.xin/train")

    def test_other_paths_are_rejected(self):
        for path in ("/pretrain", "/mint/train", "/api/mindrsi/v1/foo"):
            with pytest.raises(ConfigurationError):
                normalize_origin(f"https://gw.example.com{path}")

    @pytest.mark.parametrize(
        "bad",
        ["", "   ", "mintcn.macaron.xin", "ftp://x.example.com", "https:///train"],
    )
    def test_unusable_urls_are_rejected(self, bad):
        with pytest.raises(ConfigurationError):
            normalize_origin(bad)

    def test_full_endpoint_is_rejected_not_silently_accepted(self):
        """A query string means the caller passed an endpoint, not a base."""
        with pytest.raises(ConfigurationError):
            normalize_origin("https://mintcn.macaron.xin/api/mindrsi/v1?x=1")


class TestResolveBaseUrls:
    def test_one_root_and_three_segments(self):
        urls = resolve_base_urls("https://mintcn.macaron.xin", env={})
        assert urls["api"] == "https://mintcn.macaron.xin/api/mindrsi/v1"
        assert urls["train"] == "https://mintcn.macaron.xin/api/mindrsi/v1/train"
        assert urls["sample"] == "https://mintcn.macaron.xin/api/mindrsi/v1/sample"
        assert urls["report"] == "https://mintcn.macaron.xin/api/mindrsi/v1/service/report"

    def test_suffixed_input_does_not_double_the_root(self):
        """Passing the full fixed root is allowed, but must not double it."""
        urls = resolve_base_urls("https://mintcn.macaron.xin/api/mindrsi/v1", env={})
        assert urls["api"] == "https://mintcn.macaron.xin/api/mindrsi/v1"
        assert urls["origin"] == "https://mintcn.macaron.xin"

    def test_both_accepted_spellings_agree(self):
        suffixed = resolve_base_urls(
            "https://mintcn.macaron.xin/api/mindrsi/v1", env={}
        )
        bare = resolve_base_urls("https://mintcn.macaron.xin", env={})
        assert suffixed == bare

    def test_the_old_train_spelling_is_rejected(self):
        with pytest.raises(ConfigurationError):
            resolve_base_urls("https://mintcn.macaron.xin/train", env={})

    def test_argument_beats_environment(self):
        urls = resolve_base_urls(
            "https://alpha.example.com",
            env={"MIND_RECURSIVE_BASE_URL": "https://mintcn.macaron.xin"},
        )
        assert urls["origin"] == "https://alpha.example.com"

    def test_environment_used_when_no_argument(self):
        urls = resolve_base_urls(
            None, env={"MIND_RECURSIVE_BASE_URL": "https://alpha-overseas.example.com/api/mindrsi/v1"}
        )
        assert urls["origin"] == "https://alpha-overseas.example.com"

    def test_old_env_names_are_not_read(self):
        from mind_recursive._config import DEFAULT_BASE_URL

        urls = resolve_base_urls(
            None,
            env={"MINDRSI_BASE_URL": "https://old.example.com", "MINT_BASE_URL": "https://old2.example.com"},
        )
        assert urls["origin"] == DEFAULT_BASE_URL

    def test_blank_base_url_falls_back_to_production(self):
        from mind_recursive._config import DEFAULT_BASE_URL

        urls = resolve_base_urls(None, env={"MIND_RECURSIVE_BASE_URL": "   "})
        assert urls["origin"] == DEFAULT_BASE_URL

    def test_no_base_url_anywhere_falls_back_to_production(self):
        """With nothing set, the SDK talks to the production relay, so a user
        only needs MIND_RECURSIVE_API_KEY."""
        from mind_recursive._config import DEFAULT_BASE_URL

        urls = resolve_base_urls(None, env={})
        assert urls["origin"] == DEFAULT_BASE_URL == "https://mintcn.macaron.xin"
        assert urls["api"] == "https://mintcn.macaron.xin/api/mindrsi/v1"

    def test_environment_and_argument_override_the_default(self):
        """The default is a fallback, never an override."""
        assert resolve_base_urls(
            None, env={"MIND_RECURSIVE_BASE_URL": "https://alpha.example.com"}
        )["origin"] == "https://alpha.example.com"
        assert resolve_base_urls(
            "http://127.0.0.1:28900", env={"MIND_RECURSIVE_BASE_URL": "https://alpha.example.com"}
        )["origin"] == "http://127.0.0.1:28900"

    def test_an_explicit_empty_base_url_is_an_error(self):
        """base_url="" is a caller mistake, not a request for the default."""
        with pytest.raises(ConfigurationError):
            resolve_base_urls("   ", env={})

    def test_no_plane_carries_a_built_in_domain(self):
        """A caller's origin is the only origin; nothing is hardcoded."""
        urls = resolve_base_urls("https://gw.example.com", env={})
        for value in urls.values():
            assert value.startswith("https://gw.example.com")

    def test_sampling_has_its_own_segment(self):
        urls = resolve_base_urls("https://gw.example.com", env={})
        assert set(urls) == {
            "origin",
            "api",
            "service",
            "session",
            "train",
            "sample",
            "service_sample",
            "service_train",
            "report",
            "platform",
        }
        assert urls["platform"] == "https://gw.example.com/api/mint/cli/v1"
        assert urls["sample"] == "https://gw.example.com/api/mindrsi/v1/sample"
        assert (
            urls["service_sample"]
            == "https://gw.example.com/api/mindrsi/v1/service/sample"
        )

    def test_service_has_its_own_segment(self):
        """Health, capabilities and client config need no session and no GPU."""
        urls = resolve_base_urls("https://gw.example.com", env={})
        assert urls["service"] == "https://gw.example.com/api/mindrsi/v1/service"

    def test_session_lifecycle_has_its_own_segment(self):
        """A session outlives any one plane; it is not a training concept."""
        urls = resolve_base_urls("https://gw.example.com", env={})
        assert urls["session"] == "https://gw.example.com/api/mindrsi/v1/service/session"

    def test_report_plane_uses_the_report_segment(self):
        urls = resolve_base_urls("https://gw.example.com", env={})
        assert urls["report"] == "https://gw.example.com/api/mindrsi/v1/service/report"


class TestResolveApiKey:
    def test_argument_beats_environment(self):
        assert resolve_api_key("sk-arg", env={"MIND_RECURSIVE_API_KEY": "sk-env"}) == "sk-arg"

    @pytest.mark.parametrize("name", ENV_API_KEYS)
    def test_every_accepted_name_is_read(self, name):
        assert resolve_api_key(None, env={name: "  sk-env \n"}) == "sk-env"

    def test_old_env_names_are_not_read(self):
        with pytest.raises(ConfigurationError):
            resolve_api_key(None, env={"MINDRSI_API_KEY": "sk-a", "MINT_API_KEY": "sk-b"})

    def test_blank_key_is_missing(self):
        with pytest.raises(ConfigurationError):
            resolve_api_key(None, env={"MIND_RECURSIVE_API_KEY": "  "})

    def test_missing_key_explains_where_to_put_one(self):
        with pytest.raises(ConfigurationError) as excinfo:
            resolve_api_key(None, env={})
        message = str(excinfo.value)
        for name in ENV_API_KEYS:
            assert name in message

    def test_sk_prefix_is_accepted_without_disguise(self):
        """The whole point of owning auth: no tml- prefix check."""
        assert resolve_api_key("sk-whatever-shape", env={}) == "sk-whatever-shape"

    def test_non_sk_keys_are_not_rejected_either(self):
        """We validate nothing locally; the server is the authority."""
        assert resolve_api_key("tml-upstream-style", env={}) == "tml-upstream-style"


class TestBuildConfig:
    def test_defaults_are_the_safe_ones(self):
        config = build_config(api_key="sk-x", base_url="https://gw.example.com", env={})
        # trust_env=False keeps an ambient SOCKS proxy from hijacking traffic.
        assert config.trust_env is False
        assert config.max_retries == 3

    def test_redacted_never_leaks_the_key(self):
        config = build_config(
            api_key="sk-supersecretvalue", base_url="https://gw.example.com", env={}
        )
        redacted = config.redacted()
        assert "supersecret" not in str(redacted)
        assert redacted["api_key"].endswith("alue")


class TestWebUrl:
    """The console link is display-only, so it resolves separately from the API."""

    def test_default_is_the_console_path(self):
        assert resolve_web_url({}) == DEFAULT_WEB_URL
        assert DEFAULT_WEB_URL == (
            "https://mintrecursive.mindlab.im/v2train/records/sessions/"
        )

    def test_environment_overrides_the_default(self):
        url = resolve_web_url(
            {"MIND_RECURSIVE_WEB_URL": "  https://web.example.com/sessions/  "}
        )
        assert url == "https://web.example.com/sessions/"

    def test_blank_environment_falls_back_to_the_default(self):
        assert resolve_web_url({"MIND_RECURSIVE_WEB_URL": "   "}) == DEFAULT_WEB_URL

    def test_build_config_carries_the_resolved_url(self):
        config = build_config(
            api_key="sk-x",
            base_url="https://gw.example.com",
            env={"MIND_RECURSIVE_WEB_URL": "https://web.example.com/sessions"},
        )
        assert config.web_url == "https://web.example.com/sessions"

    def test_session_link_has_exactly_one_slash(self):
        assert web_session_url("https://web.example.com/s/", "abc") == (
            "https://web.example.com/s/abc"
        )
        assert web_session_url("https://web.example.com/s", "abc") == (
            "https://web.example.com/s/abc"
        )
