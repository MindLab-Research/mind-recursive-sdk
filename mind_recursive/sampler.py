"""``Sampler``: token-level sampling.

Tokens in, tokens plus logprobs out. This is the only sampling surface the SDK
exposes: algorithms that need it need exact token ids and token-aligned
logprobs, which a message-level chat API cannot give. Building an
``importance_sampling`` datum for GRPO is the case that decides this.

Sampling reaches the training service through the same relay ``/train`` pass-through as
training: the relay is a general proxy, not a sampling-specific one. The relay's
separate hardcoded Tinker routes (whose upstream is the inference service) are deliberately
not used here -- their field spellings differ, e.g. ``topk_sample_logprobs``
versus the native ``topk_sampled_logprobs``.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Sequence

from . import _protocol as P
from ._errors import ConfigurationError
from ._futures import APIFuture
from .types import ModelInput, SampleResponse, SamplingParams, TensorData

__all__ = ["Sampler"]

log = logging.getLogger("mind_recursive.sampler")


class Sampler:
    """A sampling session over one set of weights.

    Obtained from :meth:`mind_recursive.ServiceClient.create_sampler` or
    :meth:`save_weights_and_get_sampler`.
    """

    def __init__(
        self,
        *,
        service: Any,
        sampling_session_id: str,
        base_model: str | None = None,
        model_path: str | None = None,
        kind: str = "snapshot",
        record_stability_info: bool = False,
    ) -> None:
        self._service = service
        self._sampling_session_id = sampling_session_id
        self._base_model = base_model
        self._model_path = model_path
        self._kind = kind
        # Session-wide default for sample(record_stability_info=...). Set when
        # the session came from an ephemeral export, which has no
        # create_sampling_session call to carry the flag.
        self._record_stability_info = record_stability_info
        # Per-sampler request counter, as Tinker keeps one per sampling client:
        # The inference service requires a non-negative ``seq_id`` on every ``asample``
        # (it is the idempotency key of the request), so a caller who does not
        # pass one gets the next value. Guarded because samplers are shared by
        # rollout threads.
        self._next_seq_id = 0
        self._seq_lock = threading.Lock()

    def _allocate_seq_id(self) -> int:
        with self._seq_lock:
            seq_id = self._next_seq_id
            self._next_seq_id += 1
            return seq_id

    @property
    def sampling_session_id(self) -> str:
        return self._sampling_session_id

    @property
    def model_path(self) -> str | None:
        return self._model_path

    @property
    def base_model(self) -> str | None:
        return self._base_model

    @property
    def kind(self) -> str:
        """``snapshot`` for fixed weights, ``inflight`` for a live model's."""
        return self._kind

    def _sample_body(
        self,
        prompt: ModelInput | Sequence[int],
        num_samples: int,
        sampling_params: SamplingParams | None,
        include_prompt_logprobs: bool | None,
        topk_prompt_logprobs: int,
        topk_sample_logprobs: int,
        target_prompt_logprobs: TensorData | None,
        prompt_alt_tokens_k: int,
        prompt_logprobs_last_n: int | None,
        seq_id: int | None,
        prompt_logprobs: bool,
        record_stability_info: bool,
    ) -> dict[str, Any]:
        """Shared request body for :meth:`sample` and :meth:`sample_async`."""
        if include_prompt_logprobs is not None:
            if prompt_logprobs and not include_prompt_logprobs:
                raise ValueError(
                    "include_prompt_logprobs=False conflicts with "
                    "prompt_logprobs=True"
                )
            prompt_logprobs = include_prompt_logprobs
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")
        model_input = (
            prompt if isinstance(prompt, ModelInput) else ModelInput.from_ints(prompt)
        )
        params = sampling_params or SamplingParams()

        if prompt_logprobs_last_n is not None:
            if not prompt_logprobs:
                raise ValueError(
                    "prompt_logprobs_last_n scores prompt tokens, so it needs "
                    "prompt_logprobs=True"
                )
            if prompt_logprobs_last_n < 1:
                raise ValueError("prompt_logprobs_last_n must be at least 1")
            limit = model_input.length - 1
            if limit >= 1 and prompt_logprobs_last_n > limit:
                raise ValueError(
                    f"prompt_logprobs_last_n must be at most len(prompt) - 1 "
                    f"({limit}), got {prompt_logprobs_last_n}"
                )
        for name, value in (
            ("topk_prompt_logprobs", topk_prompt_logprobs),
            ("topk_sample_logprobs", topk_sample_logprobs),
            ("prompt_alt_tokens_k", prompt_alt_tokens_k),
        ):
            if value < 0:
                raise ValueError(f"{name} cannot be negative, got {value}")
        if seq_id is not None and seq_id < 0:
            raise ValueError("seq_id cannot be negative")

        body: dict[str, Any] = {
            "num_samples": num_samples,
            "prompt": _encode_model_input(model_input),
            "sampling_params": params.to_wire(),
            "sampling_session_id": self._sampling_session_id,
            "type": "sample",
        }
        body["seq_id"] = seq_id if seq_id is not None else self._allocate_seq_id()
        if prompt_logprobs:
            body["prompt_logprobs"] = True
        if prompt_logprobs_last_n is not None:
            body["prompt_logprobs_last_n"] = prompt_logprobs_last_n
        if topk_prompt_logprobs:
            body["topk_prompt_logprobs"] = topk_prompt_logprobs
        if topk_sample_logprobs:
            body["topk_sample_logprobs"] = topk_sample_logprobs
        if target_prompt_logprobs is not None:
            body["target_prompt_logprobs"] = target_prompt_logprobs.to_wire()
        if prompt_alt_tokens_k:
            body["prompt_alt_tokens_k"] = prompt_alt_tokens_k
        if record_stability_info or self._record_stability_info:
            body["record_stability_info"] = True
        return body

    def sample(
        self,
        prompt: ModelInput | Sequence[int],
        num_samples: int = 1,
        sampling_params: SamplingParams | None = None,
        include_prompt_logprobs: bool | None = None,
        topk_prompt_logprobs: int = 0,
        topk_sample_logprobs: int = 0,
        target_prompt_logprobs: TensorData | None = None,
        prompt_alt_tokens_k: int = 0,
        prompt_logprobs_last_n: int | None = None,
        *,
        seq_id: int | None = None,
        prompt_logprobs: bool = False,
        record_stability_info: bool = False,
    ) -> APIFuture[SampleResponse]:
        """Sample continuations for one prompt.

        Per-token logprobs for the *sampled* tokens come back unconditionally in
        ``SampledSequence.logprobs`` -- policy-gradient losses need them, so
        they are not behind a flag. Everything about the *prompt* costs extra
        compute, so it is opt-in through the parameters below.

        Args:
            prompt: a :class:`~mind_recursive.types.ModelInput`, or token ids.
            num_samples: how many continuations to draw.
            sampling_params: decoding parameters; defaults are used if omitted.
            include_prompt_logprobs: Tinker's spelling of ``prompt_logprobs``.
                Passing one spelling is enough; giving both with conflicting
                values raises locally.
            seq_id: sequence index within the sampling session. Multi-turn
                callers pass a distinct one per turn so the request id is
                deterministic and a retry cannot be mistaken for a new turn.
                Omitted, the sampler allocates the next one (0, 1, 2, ...);
                the server requires one on every request.
            prompt_logprobs: score the prompt tokens too, returned in
                ``SampleResponse.prompt_logprobs``.
            prompt_logprobs_last_n: score only the last N prompt tokens;
                earlier entries come back as ``None``. The prefix before them
                can then be served from the server's prefix cache, so scoring a
                short suffix of a long prompt costs about what the suffix costs.
                Requires ``prompt_logprobs``; must be between 1 and
                ``len(prompt) - 1``.
            topk_prompt_logprobs: return the top-k at each prompt position.
            topk_sample_logprobs: return the top-k at each sampled position, in
                ``SampledSequence.topk_logprobs``. Distillation against a
                teacher's distribution is the case that wants this.
            target_prompt_logprobs: token ids to score at chosen prompt
                positions, as an int64 tensor of shape ``[len(prompt) - 1, K]``:
                cell ``[i, j]`` is scored at prompt position ``i + 1``, since
                position 0 has no preceding context. ``-1`` requests nothing.
                May be sparse CSR, in which case only the listed cells are sent
                and returned. Pair it with ``max_tokens=1`` to make the request
                a pure prefill scoring pass.
            prompt_alt_tokens_k: draw this many alternative tokens at every
                prompt position after the first, at the request's temperature,
                within the same prefill.
            record_stability_info: have the server record per-sequence
                stability information, referenced later by ``sequence_id``
                (``stable_reinforce``).

        Returns:
            A future resolving to :class:`~mind_recursive.types.SampleResponse`.
        """
        body = self._sample_body(
            prompt,
            num_samples,
            sampling_params,
            include_prompt_logprobs,
            topk_prompt_logprobs,
            topk_sample_logprobs,
            target_prompt_logprobs,
            prompt_alt_tokens_k,
            prompt_logprobs_last_n,
            seq_id,
            prompt_logprobs,
            record_stability_info,
        )
        payload = self._service._post_json(
            P.PATH_ASAMPLE,
            body,
            pool="train",
            base_url=self._service.config.sample_base_url,
        )
        request_id = payload.get("request_id")
        if not request_id:
            raise ConfigurationError(f"asample returned no request_id: {payload!r}")
        return APIFuture(
            str(request_id),
            # Poll the future on the plane that produced it, not on /train.
            self._service.resolver(base_url=self._service.config.sample_base_url),
            P.parse_sample_response,
        )

    async def sample_async(
        self,
        prompt: ModelInput | Sequence[int],
        num_samples: int = 1,
        sampling_params: SamplingParams | None = None,
        include_prompt_logprobs: bool | None = None,
        topk_prompt_logprobs: int = 0,
        topk_sample_logprobs: int = 0,
        target_prompt_logprobs: TensorData | None = None,
        prompt_alt_tokens_k: int = 0,
        prompt_logprobs_last_n: int | None = None,
        *,
        seq_id: int | None = None,
        prompt_logprobs: bool = False,
        record_stability_info: bool = False,
    ) -> SampleResponse:
        """Async twin of :meth:`sample`."""
        body = self._sample_body(
            prompt,
            num_samples,
            sampling_params,
            include_prompt_logprobs,
            topk_prompt_logprobs,
            topk_sample_logprobs,
            target_prompt_logprobs,
            prompt_alt_tokens_k,
            prompt_logprobs_last_n,
            seq_id,
            prompt_logprobs,
            record_stability_info,
        )
        payload = await self._service._apost_json(
            P.PATH_ASAMPLE,
            body,
            pool="train",
            base_url=self._service.config.sample_base_url,
        )
        request_id = payload.get("request_id")
        if not request_id:
            raise ConfigurationError(f"asample returned no request_id: {payload!r}")
        return await self._service.async_resolver(
            base_url=self._service.config.sample_base_url
        ).resolve(request_id, timeout=None, parse=P.parse_sample_response)

    async def compute_logprobs_async(
        self,
        prompt: ModelInput | Sequence[int],
        prompt_logprobs_last_n: int | None = None,
    ) -> list[float] | None:
        """Async twin of :meth:`compute_logprobs`."""
        response = await self.sample_async(
            prompt,
            num_samples=1,
            sampling_params=SamplingParams(max_tokens=1),
            include_prompt_logprobs=True,
            prompt_logprobs_last_n=prompt_logprobs_last_n,
        )
        return response.prompt_logprobs

    async def get_base_model_async(self) -> str | None:
        """Async twin of :meth:`get_base_model`."""
        return self._base_model

    def compute_logprobs(
        self,
        prompt: ModelInput | Sequence[int],
        prompt_logprobs_last_n: int | None = None,
    ) -> APIFuture[list[float] | None]:
        """Score the prompt tokens, Tinker-style.

        This is a thin shell over :meth:`sample`: one prefill with
        ``max_tokens=1`` and ``prompt_logprobs=True``, reshaping the response
        to the prompt logprobs. It does not add a second wire endpoint.

        Args:
            prompt: a :class:`~mind_recursive.types.ModelInput`, or token ids.
            prompt_logprobs_last_n: score only the last N prompt tokens; earlier
                entries come back as ``None``. Must be between 1 and
                ``len(prompt) - 1``.

        Returns:
            A future resolving to the per-token prompt logprobs.
        """
        future = self.sample(
            prompt,
            num_samples=1,
            sampling_params=SamplingParams(max_tokens=1),
            include_prompt_logprobs=True,
            prompt_logprobs_last_n=prompt_logprobs_last_n,
        )
        return future.map(lambda response: response.prompt_logprobs)

    def get_base_model(self) -> str | None:
        """The base model name, when the sampler knows it.

        Tinker resolves this server-side for checkpoint-only samplers; this SDK
        only knows the name when it was passed to ``create_sampler``, so a
        checkpoint-only sampler returns ``None``.
        """
        return self._base_model

    def result(self, timeout: float | None = None) -> Sampler:
        """Tinker-style future compatibility: return this sampler.

        ``ServiceClient.create_sampling_client`` returns a sampler directly;
        upstream returns a future of one. Creation already blocked by the time
        this object exists, so ``.result()`` is the identity here.
        """
        return self

    def __repr__(self) -> str:
        target = self._model_path or self._base_model or self._kind
        return f"<Sampler {self._sampling_session_id} ({target})>"


def _encode_model_input(model_input: ModelInput) -> dict[str, Any]:
    """JSON encoding of a ``ModelInput`` for the sampling request.

    Sampling requests are JSON (only ``forward_backward`` is protobuf), so
    chunks are serialised as objects here rather than packed bytes.
    """
    from .types import DmelChunk, EncodedTextChunk, ImageChunk

    chunks: list[dict[str, Any]] = []
    for chunk in model_input.chunks:
        if isinstance(chunk, EncodedTextChunk):
            chunks.append({"type": "encoded_text", "tokens": list(chunk.tokens)})
        elif isinstance(chunk, ImageChunk):
            import base64

            entry: dict[str, Any] = {
                "type": "image",
                "data": base64.b64encode(chunk.data).decode("ascii"),
                "format": chunk.format,
            }
            if chunk.expected_tokens is not None:
                entry["expected_tokens"] = chunk.expected_tokens
            chunks.append(entry)
        elif isinstance(chunk, DmelChunk):
            import base64

            chunks.append(
                {"type": "dmel", "dmel": base64.b64encode(chunk.dmel).decode("ascii")}
            )
        else:
            raise ValueError(f"unsupported chunk type: {type(chunk).__name__}")
    return {"chunks": chunks}
