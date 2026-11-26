"""mind_recursive -- the Mind Recursive platform's Python SDK.

Low-level training and sampling against the Mind Recursive platform, for people who write their own
training loop, plus the session-less platform resources on ``UserClient``:
datasets (``uc.datasets``), managed training runs with their records, metrics
and events (``uc.training``), models (``uc.models``), deployments
(``uc.deployments``) and calls to them in the OpenAI format
(``uc.chat.completions.create``). Evaluation stays with the web app.

The module name is ``mind_recursive``; ``import mind_recursive as mr`` is the
short form used in the docs.

    import mind_recursive

    sc = mind_recursive.ServiceClient()                      # opens a session immediately
    trainer = sc.create_lora_trainer(
        base_model="Qwen/Qwen3.8-27B",
        rank=32,
    )

    sc.log("starting SFT")

    for step, batch in enumerate(batches):
        out = trainer.forward_backward(batch).result()
        trainer.optim_step().result()
        sc.metric({"loss": out.loss}, step=step, stage="training", total=800)

    path = trainer.save_weights_only("step-100").result()
    sc.artifact(path, type="sampler_weights")
    sc.summary(final_loss=out.loss)

One client, two things made from it. ``ServiceClient`` is the only object that
holds a connection, a session and the billing identity; the trainer and
sampler are handles onto server-side resources, so closing the service client
is the only cleanup there is. Reporting belongs to the client itself: there is
one writer per session, so ``log`` / ``event`` / ``metric`` / ``artifact`` /
``summary`` are methods on ``ServiceClient``.

``from mind_recursive import tinker`` exposes the Tinker-shaped namespace for porting
existing Tinker or cookbook code, matching Tinker's lowercase import. The
native ``mind_recursive`` classes keep their own shape; the compatibility namespace is
where Tinker method shapes live.

Importing this package has no side effects: no monkeypatching, no
``os.environ`` rewriting. Credentials and URLs resolve when a client is
constructed, which is what makes it possible to talk to Mind Recursive and
upstream Tinker from one process.
"""

from __future__ import annotations

from ._config import Config, build_config, normalize_origin
from ._version import __version__
from ._errors import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConfigurationError,
    ConflictError,
    FutureTimeoutError,
    GoneError,
    MindRecursiveError,
    NotFoundError,
    PayloadTooLargeError,
    PermissionDeniedError,
    RateLimitError,
    RequestFailedError,
    ServerError,
    SessionClosedError,
)
from ._futures import APIFuture
from ._session import ClientConfig
from .sampler import Sampler
from .service import SDK_VERSION, ServiceClient
from .trainer import LoraTrainer
from .user import UserClient
from .platform import (
    Account,
    Dataset,
    DatasetFinding,
    DatasetValidationError,
    Deployment,
    DeploymentError,
    Execution,
    ExecutionEvent,
    ExecutionResult,
    MetricPoint,
    ModelVersion,
    Page,
    PlatformCapabilities,
    ValidationReport,
)
from .types import (
    AdamParams,
    Datum,
    LossFnType,
    ModelInfo,
    SamplerKind,
    TensorDtype,
    DmelChunk,
    EncodedTextChunk,
    ForwardBackwardOutput,
    ImageChunk,
    LoraConfig,
    ModelInput,
    PromptAltTokens,
    SampledSequence,
    SampleResponse,
    TopkLogprobs,
    SamplingParams,
    ServerCapabilities,
    SupportedModel,
    TensorData,
)

# 版本号单一事实源见 mind_recursive/_version.py（读 pyproject.toml 的 version）。

__all__ = [
    "__version__",
    "SDK_VERSION",
    # the client, and what it creates
    "ServiceClient",
    "UserClient",
    "LoraTrainer",
    "Sampler",
    # config
    "Config",
    "ClientConfig",
    "build_config",
    "normalize_origin",
    # futures
    "APIFuture",
    # data types
    "AdamParams",
    "Datum",
    # the Literals that type the public parameters, so callers can annotate
    # their own helpers with the same names the SDK uses
    "LossFnType",
    "ModelInfo",
    "SamplerKind",
    "TensorDtype",
    "DmelChunk",
    "EncodedTextChunk",
    "ForwardBackwardOutput",
    "ImageChunk",
    "LoraConfig",
    "ModelInput",
    "PromptAltTokens",
    "SampleResponse",
    "SampledSequence",
    "TopkLogprobs",
    "SamplingParams",
    "ServerCapabilities",
    "SupportedModel",
    "TensorData",
    # platform resources (UserClient namespaces)
    "Account",
    "Dataset",
    "DatasetFinding",
    "Deployment",
    "Execution",
    "ExecutionEvent",
    "ExecutionResult",
    "MetricPoint",
    "ModelVersion",
    "Page",
    "PlatformCapabilities",
    "ValidationReport",
    # errors
    "MindRecursiveError",
    "ConfigurationError",
    "APIError",
    "APIConnectionError",
    "APITimeoutError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "BadRequestError",
    "ConflictError",
    "GoneError",
    "PayloadTooLargeError",
    "RateLimitError",
    "ServerError",
    "RequestFailedError",
    "FutureTimeoutError",
    "SessionClosedError",
    "DatasetValidationError",
    "DeploymentError",
]
