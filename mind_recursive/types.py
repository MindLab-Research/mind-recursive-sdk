"""Public data types.

Plain dataclasses, no pydantic. The wire shapes are fixed by the server, so
validation buys little here; what matters is that the JSON and protobuf
encoders below produce byte-identical output to the stock SDK.

Names and field names match Tinker's so recipe code ports without edits, but
these are our classes: ``isinstance(x, tinker.Datum)`` is false.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import numpy as np

__all__ = [
    "LossFnType",
    "SamplerKind",
    "TensorDtype",
    "TensorData",
    "EncodedTextChunk",
    "ImageChunk",
    "DmelChunk",
    "ModelInputChunk",
    "ModelInput",
    "Datum",
    "AdamParams",
    "LoraConfig",
    "SamplingParams",
    "StopReason",
    "TopkLogprobs",
    "PromptAltTokens",
    "SampledSequence",
    "SampleResponse",
    "ForwardBackwardOutput",
    "ModelInfo",
    "SupportedModel",
    "ServerCapabilities",
    "MINT_ADAM_CONSTRAINT",
]

#: Loss functions the server accepts. Mirrors tinker.types.loss_fn_type.
LossFnType = Literal[
    "cross_entropy",
    "importance_sampling",
    "ppo",
    "cispo",
    "dro",
    "gspo",
    "stable_reinforce",
    "custom",
]

#: What a sampling session's weights are bound to. The runtime tuple that
#: validates it lives in ``_protocol.SAMPLER_KINDS``; this is the annotation,
#: so a wrong spelling is caught in the editor rather than at the call.
SamplerKind = Literal["snapshot", "inflight"]

#: The public wire collapses all dtypes to these two.
TensorDtype = Literal["float32", "int64"]

_NUMPY_DTYPE: dict[str, np.dtype] = {
    "float32": np.dtype(np.float32),
    "int64": np.dtype(np.int64),
}

#: Wire dtype implied by a ``loss_fn_inputs`` key, for callers who pass plain
#: Python lists. Mirrors Tinker's ``_KEY_TO_TYPE``; keys absent here fall back
#: to inference from the array's numpy kind.
_LOSS_FN_INPUT_DTYPE: dict[str, TensorDtype] = {
    "target_tokens": "int64",
    "weights": "float32",
    "advantages": "float32",
    "logprobs": "float32",
    "clip_low_threshold": "float32",
    "clip_high_threshold": "float32",
    "sequence_ratio_mask": "float32",
}


@dataclass
class TensorData:
    """A dense or sparse-CSR tensor, as the public wire represents one.

    ``data`` is coerced to the numpy dtype implied by ``dtype`` on
    construction, so the protobuf encoder can call ``.tobytes()`` without
    re-checking. Supplying ``sparse_crow_indices`` requires
    ``sparse_col_indices`` too.
    """

    data: np.ndarray
    dtype: TensorDtype = "float32"
    shape: list[int] | None = None
    sparse_crow_indices: list[int] | None = None
    sparse_col_indices: list[int] | None = None

    def __post_init__(self) -> None:
        if self.dtype not in _NUMPY_DTYPE:
            raise ValueError(
                f"TensorData.dtype must be 'float32' or 'int64', got {self.dtype!r}"
            )
        want = _NUMPY_DTYPE[self.dtype]
        arr = np.asarray(self.data)
        self.data = arr if arr.dtype == want else arr.astype(want)
        if self.shape is None:
            self.shape = list(self.data.shape)
        if (self.sparse_crow_indices is None) != (self.sparse_col_indices is None):
            raise ValueError(
                "sparse_crow_indices and sparse_col_indices must be given together"
            )

    @property
    def is_sparse(self) -> bool:
        return self.sparse_crow_indices is not None

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> TensorData:
        """Infer the public dtype from a numpy array's kind."""
        a = np.asarray(arr)
        dtype: TensorDtype = "float32" if a.dtype.kind == "f" else "int64"
        return cls(data=a, dtype=dtype)

    def to_numpy(self) -> np.ndarray:
        """Dense view, reshaped to ``shape`` when that is possible."""
        if self.is_sparse or not self.shape:
            return self.data
        try:
            return self.data.reshape(self.shape)
        except ValueError:
            return self.data

    def tolist(self) -> list:
        return self.to_numpy().tolist()

    def to_wire(self) -> dict[str, Any]:
        """JSON form: a flat number list plus dtype, shape and CSR indices.

        Sampling requests are JSON (only ``forward_backward`` is protobuf), so
        a ``TensorData`` travelling as a request parameter is serialised here
        rather than by the protobuf encoder.
        """
        out: dict[str, Any] = {
            "data": self.data.ravel().tolist(),
            "dtype": self.dtype,
        }
        if self.shape is not None:
            out["shape"] = list(self.shape)
        if self.sparse_crow_indices is not None:
            out["sparse_crow_indices"] = list(self.sparse_crow_indices)
            out["sparse_col_indices"] = list(self.sparse_col_indices or [])
        return out


@dataclass
class EncodedTextChunk:
    """Pre-tokenised text. ``tokens`` are packed as int32 on the wire."""

    tokens: Sequence[int]
    type: Literal["encoded_text"] = "encoded_text"

    def __len__(self) -> int:
        return len(self.tokens)


@dataclass
class ImageChunk:
    """Raw image bytes. The server uploads them and computes dimensions.

    Multimodal input stays inline (base64 data URI at the user-facing layer,
    raw bytes here) to match Tinker and Fireworks; we do not do asset
    references.
    """

    data: bytes
    format: str
    expected_tokens: int | None = None
    type: Literal["image"] = "image"

    def __len__(self) -> int:
        # Token count is the server's call; expected_tokens is a hint.
        return self.expected_tokens or 0


@dataclass
class DmelChunk:
    """Serialised audio TensorContainer bytes, passed through untouched."""

    dmel: bytes
    type: Literal["dmel"] = "dmel"

    def __len__(self) -> int:
        return 0


ModelInputChunk = EncodedTextChunk | ImageChunk | DmelChunk


@dataclass
class ModelInput:
    """An ordered chunk sequence forming one model input."""

    chunks: list[ModelInputChunk] = field(default_factory=list)

    @classmethod
    def from_ints(cls, tokens: Sequence[int]) -> ModelInput:
        """The common case: one text chunk of token ids."""
        return cls(chunks=[EncodedTextChunk(tokens=list(tokens))])

    @classmethod
    def empty(cls) -> ModelInput:
        return cls(chunks=[])

    def append_int(self, token: int) -> None:
        if self.chunks and isinstance(self.chunks[-1], EncodedTextChunk):
            last = self.chunks[-1]
            last.tokens = [*last.tokens, token]
        else:
            self.chunks.append(EncodedTextChunk(tokens=[token]))

    def to_ints(self) -> list[int]:
        """Flatten text chunks to token ids.

        Raises:
            ValueError: a non-text chunk is present, so there is no faithful
                integer representation.
        """
        out: list[int] = []
        for chunk in self.chunks:
            if not isinstance(chunk, EncodedTextChunk):
                raise ValueError(
                    f"ModelInput contains a {type(chunk).__name__}; cannot flatten to ints"
                )
            out.extend(chunk.tokens)
        return out

    @property
    def length(self) -> int:
        return sum(len(c) for c in self.chunks)

    def __len__(self) -> int:
        return self.length


def _coerce_loss_fn_input(key: str, value: Any) -> TensorData:
    """Wrap one ``loss_fn_inputs`` value in :class:`TensorData`.

    Lists, numpy arrays and anything implementing ``__array__`` (CPU torch
    tensors included) are accepted, so recipe code written against Tinker --
    which coerces in ``Datum.__post_init__`` the same way -- ports unchanged.
    Without this, a plain list survives construction and fails much later
    inside the protobuf encoder, far from the call site that built it.

    The dtype comes from :data:`_LOSS_FN_INPUT_DTYPE` for known keys, so
    ``target_tokens=[2, 3, 0]`` lands as int64 rather than being inferred as
    one by accident; unknown keys fall back to the array's numpy kind.
    """
    if isinstance(value, TensorData):
        return value
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"loss_fn_inputs[{key!r}] must be a TensorData, a numpy array, or "
            f"something convertible to one, got {type(value).__name__}"
        ) from exc
    if array.dtype.kind not in ("f", "i", "u", "b"):
        raise TypeError(
            f"loss_fn_inputs[{key!r}] must hold numbers, got dtype {array.dtype}"
        )
    dtype = _LOSS_FN_INPUT_DTYPE.get(key)
    if dtype is None:
        dtype = "float32" if array.dtype.kind == "f" else "int64"
    return TensorData(data=array, dtype=dtype)


@dataclass(frozen=True)
class Datum:
    """One training example: a model input plus the loss function's tensors.

    ``loss_fn_inputs`` keys depend on ``loss_fn``. ``cross_entropy`` wants
    ``weights`` and ``target_tokens``; the policy-gradient losses additionally
    want ``advantages`` and ``logprobs``. Values may be given as plain lists or
    numpy arrays; construction coerces them to :class:`TensorData`.

    Frozen, like Tinker's and Fireworks' datums: a training example is a value,
    and rebinding one after it has been batched or sent would silently change
    what a step trained on.
    """

    model_input: ModelInput
    loss_fn_inputs: Mapping[str, TensorData] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "loss_fn_inputs",
            {k: _coerce_loss_fn_input(k, v) for k, v in self.loss_fn_inputs.items()},
        )

    @classmethod
    def from_tokens(
        cls,
        tokens: Sequence[int],
        *,
        weights: Sequence[float] | None = None,
        target_tokens: Sequence[int] | None = None,
    ) -> Datum:
        """Build a ``cross_entropy`` datum from token ids.

        ``target_tokens`` defaults to ``tokens`` shifted left by one (standard
        next-token prediction) and ``weights`` to all-ones. The caller is
        responsible for the loss mask in real training -- an all-ones mask
        trains on the prompt too.
        """
        toks = list(tokens)
        if target_tokens is None:
            target_tokens = [*toks[1:], 0]
        if weights is None:
            weights = [1.0] * len(toks)
        return cls(
            model_input=ModelInput.from_ints(toks),
            loss_fn_inputs={
                "weights": TensorData(np.asarray(weights, dtype=np.float32), "float32"),
                "target_tokens": TensorData(
                    np.asarray(target_tokens, dtype=np.int64), "int64"
                ),
            },
        )


#: The MinT backend accepts exactly one set of Adam hyperparameters. Anything
#: else is rejected server-side, verbatim::
#:
#:     UNSUPPORTED: FSDP2 requires beta1=0.9, beta2=0.999, eps=1e-8
#:     and lora_plus_ratio=1
#:
#: Note these differ from the stock Tinker defaults (beta2=0.95, eps=1e-12),
#: which is why we override rather than inherit them.
MINT_ADAM_CONSTRAINT = {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8}


@dataclass
class AdamParams:
    """Per-step Adam hyperparameters.

    Defaults are the values MinT requires, so ``AdamParams(learning_rate=...)``
    is always accepted. Overriding ``beta1``/``beta2``/``eps`` is allowed --
    we do not silently correct them -- but the server will reject the step.
    """

    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.0

    def violations(self) -> dict[str, tuple[float, float]]:
        """``{field: (given, required)}`` for each value MinT will reject."""
        out: dict[str, tuple[float, float]] = {}
        for name, required in MINT_ADAM_CONSTRAINT.items():
            given = getattr(self, name)
            if given != required:
                out[name] = (given, required)
        return out

    def to_wire(self) -> dict[str, float]:
        return {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "grad_clip_norm": self.grad_clip_norm,
        }


@dataclass
class LoraConfig:
    """LoRA placement and rank.

    ``train_unembed`` defaults to ``False`` here while the stock Tinker SDK
    defaults it to ``True``. Training the LM head produces weights that fail
    to deploy, so the safe value is the default and raising it has to be
    deliberate.
    """

    rank: int = 32
    seed: int | None = None
    train_unembed: bool = False
    train_mlp: bool = True
    train_attn: bool = True

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "rank": self.rank,
            "train_unembed": self.train_unembed,
            "train_mlp": self.train_mlp,
            "train_attn": self.train_attn,
        }
        if self.seed is not None:
            out["seed"] = self.seed
        return out


@dataclass
class SamplingParams:
    """Decoding parameters for ``sample``."""

    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    seed: int | None = None
    stop: list[str] | None = None
    stop_tokens: list[int] | None = None

    def to_wire(self) -> dict[str, Any]:
        """JSON form.

        ``stop`` is one untagged union upstream -- text, a string array, or
        token ids -- so ``stop_tokens`` is sugar for the token-id arm, not a
        second field. Sending both keys would silently drop one: MinT has no
        ``stop_tokens``. Mixing text and token ids is rejected here because the
        union cannot represent it.
        """
        out: dict[str, Any] = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.top_k is not None:
            out["top_k"] = self.top_k
        if self.seed is not None:
            out["seed"] = self.seed
        if self.stop and self.stop_tokens:
            raise ValueError(
                "pass stop= (text) or stop_tokens= (token ids), not both: "
                "the wire's stop union cannot carry text and ids together"
            )
        if self.stop:
            out["stop"] = list(self.stop)
        elif self.stop_tokens:
            out["stop"] = [int(t) for t in self.stop_tokens]
        return out


StopReason = Literal["stop", "length"]


@dataclass
class TopkLogprobs:
    """A dense ``length x k`` top-k block, as the wire sends one.

    ``token_ids`` and ``logprobs`` are row-major flattened: position ``i``,
    rank ``j`` sits at index ``i * k + j``. Empty cells carry token id 0 and
    logprob ``-99999.0`` (the server's filler, not a real score).
    """

    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    k: int = 0
    length: int = 0

    def row(self, position: int) -> list[tuple[int, float]]:
        """The ``(token_id, logprob)`` pairs at one position, best first."""
        if self.k <= 0:
            return []
        start = position * self.k
        return list(
            zip(
                self.token_ids[start : start + self.k],
                self.logprobs[start : start + self.k],
            )
        )

    def to_rows(self) -> list[list[tuple[int, float]]]:
        """Every position as its own list of pairs."""
        return [self.row(i) for i in range(self.length)]


@dataclass
class PromptAltTokens:
    """Alternative draws at each prompt position after the first.

    Both tensors are ``[len(prompt) - 1, k]``; row ``i`` covers prompt position
    ``i + 1``, because position 0 has no preceding context to draw from.
    """

    tokens: TensorData | None = None
    logprobs: TensorData | None = None


@dataclass
class SampledSequence:
    """One sampled continuation."""

    tokens: list[int]
    stop_reason: StopReason = "stop"
    logprobs: list[float] | None = None
    sequence_id: str | None = None
    topk_logprobs: TopkLogprobs | None = None
    """Top-k at each sampled position. None unless ``topk_sample_logprobs``
    was asked for."""

    def __len__(self) -> int:
        return len(self.tokens)


@dataclass
class SampleResponse:
    """The result of one ``sample`` call.

    ``prompt_cache_hit_tokens`` is the hit count for a *single* copy of the
    prompt; with ``num_samples=N`` the other ``N-1`` copies bill as cached.
    """

    sequences: list[SampledSequence] = field(default_factory=list)
    prompt_logprobs: list[float] | None = None
    """Per-token logprobs for the prompt. None unless ``prompt_logprobs`` was
    requested. With ``prompt_logprobs_last_n`` set, only the last N entries are
    scored and the earlier ones are ``None``."""

    prompt_cache_hit_tokens: int = 0
    topk_prompt_logprobs: TopkLogprobs | None = None
    """Top-k at each prompt position. None unless ``topk_prompt_logprobs`` was
    requested."""

    target_prompt_logprobs: TensorData | None = None
    """Logprobs of the ids named by the request's ``target_prompt_logprobs``,
    same shape and layout, 0.0 where the request held ``-1``."""

    prompt_alt_tokens: PromptAltTokens | None = None
    """The draws ``prompt_alt_tokens_k`` asked for. None unless requested."""


@dataclass
class ForwardBackwardOutput:
    """The result of one ``forward_backward`` call.

    ``metrics`` carries the scalars the server computed, including
    ``"loss:mean"`` and ``"num_tokens:sum"``. ``loss_fn_outputs`` holds the
    per-datum tensors (e.g. ``logprobs``) when the server returned them.
    """

    loss_fn_output_type: str = "ArrayRecord"
    loss_fn_outputs: list[dict[str, TensorData]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def loss(self) -> float | None:
        """``metrics["loss:mean"]`` when present."""
        return self.metrics.get("loss:mean")

    @property
    def num_tokens(self) -> int | None:
        """``metrics["num_tokens:sum"]`` as an int, when present.

        This is the quantity token-based billing meters, which is why it has a
        named accessor rather than being left in the dict.
        """
        value = self.metrics.get("num_tokens:sum")
        return None if value is None else int(value)


@dataclass
class ModelInfo:
    """The server's record of one training model, from ``get_info``.

    Every field is optional and unknown keys are kept in :attr:`extra`,
    because this response is the server's own projection and it grows: a
    missing field must read as "this server did not say", not as an error.
    Use :attr:`lora_rank` rather than guessing the key name -- the wire spells
    it ``lora_rank``, not ``rank``.
    """

    model_id: str | None = None
    model_name: str | None = None
    is_lora: bool | None = None
    lora_rank: int | None = None
    optimizer_config: dict[str, Any] | None = None
    model_data: dict[str, Any] | None = None
    #: Anything the server sent that this class does not name yet.
    extra: dict[str, Any] = field(default_factory=dict)

    #: Keys mapped onto named attributes; everything else lands in ``extra``.
    _NAMED = (
        "model_id",
        "model_name",
        "is_lora",
        "lora_rank",
        "optimizer_config",
        "model_data",
    )

    @classmethod
    def from_wire(cls, payload: dict[str, Any]) -> ModelInfo:
        """Build from a ``get_info`` body, keeping unmodelled keys.

        ``type`` is dropped: it is the envelope's own discriminator, not
        information about the model.
        """
        known = {key: payload.get(key) for key in cls._NAMED}
        rank = known.get("lora_rank")
        if rank is not None:
            known["lora_rank"] = int(rank)
        extra = {
            key: value
            for key, value in payload.items()
            if key not in cls._NAMED and key != "type"
        }
        return cls(**known, extra=extra)


@dataclass
class SupportedModel:
    """One entry from ``get_server_capabilities``."""

    model_name: str | None = None
    max_context_length: int | None = None
    trainable: bool | None = None
    sampleable: bool | None = None


@dataclass
class ServerCapabilities:
    """What the backend currently serves."""

    supported_models: list[SupportedModel] = field(default_factory=list)

    def names(self) -> list[str]:
        return [m.model_name for m in self.supported_models if m.model_name]

    def find(self, model_name: str) -> SupportedModel | None:
        for m in self.supported_models:
            if m.model_name == model_name:
                return m
        return None
