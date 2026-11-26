"""Platform namespaces on UserClient, against a fake relay CLI entry.

The fake speaks the platform service's contract as the relay CLI entry serves
it: ``/api/mint/cli/v1/*``, ``{code, message, metadata}`` envelopes, int64 as
strings, and the relay's own ``{error: {code, message}}`` rejections.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import pytest

import mind_recursive
from mind_recursive import platform as P

ORIGIN = "https://relay.test"
ENTRY = "/api/mint/cli/v1"
OBJ = "https://oss.test"


def ok(metadata: Any) -> httpx.Response:
    return httpx.Response(200, json={"code": 200, "message": "ok", "metadata": metadata})


class _Stream(httpx.SyncByteStream):
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __iter__(self):
        yield self._body[: len(self._body) // 2]
        yield self._body[len(self._body) // 2 :]


class FakeEntry:
    """Records requests; routes ``(method, path)`` to canned handlers."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.routes: dict[tuple[str, str], Any] = {}
        self.objects: dict[str, bytes] = {}
        self.dataset_polls = 0

    def route(self, method: str, path: str, fn: Any) -> None:
        self.routes[(method, ENTRY + path)] = fn

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.host == "oss.test":
            if request.method == "PUT":
                self.objects[request.url.path] = request.read()
                return httpx.Response(200)
            body = self.objects.get(request.url.path)
            if body is None:
                return httpx.Response(404)
            # A real stream, so the SDK's iter_raw() path is what is tested.
            return httpx.Response(
                200, headers={"Content-Length": str(len(body))}, stream=_Stream(body)
            )
        raw_path = request.url.raw_path.split(b"?")[0].decode()
        fn = self.routes.get((request.method, raw_path))
        if fn is None:
            return httpx.Response(
                404, json={"error": {"code": "route_not_allowed", "message": "no"}}
            )
        return fn(request) if callable(fn) else fn

    def to(self, path: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path == ENTRY + path]


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("mind_recursive.platform.time.sleep", lambda _s: None)
    monkeypatch.setattr("mind_recursive._transport.time.sleep", lambda _s: None)


@pytest.fixture
def fake(monkeypatch):
    entry = FakeEntry()
    mock = httpx.MockTransport(entry.handler)

    real_client = httpx.Client

    def client(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs["transport"] = mock
        return real_client(*args, **kwargs)

    # Both the transport pools and the clean object-URL client build httpx.Client.
    monkeypatch.setattr("mind_recursive._transport.httpx.Client", client)
    monkeypatch.setattr("mind_recursive.platform.httpx.Client", client)
    return entry


@pytest.fixture
def uc(fake):
    client = mind_recursive.UserClient(api_key="sk-test", base_url=ORIGIN, max_retries=2)
    yield client
    client.close()


DATASET = {
    "dataset_version": "ds-1",
    "type": "sft",
    "status": "available",
    "size_bytes": "12",
    "rows": "3",
    "created_at": "1700000000000",
}

EXECUTION = {
    "execution_id": "ex-1",
    "kind": "training",
    "state": "running",
    "dataset_version": "ds-1",
    "base_model": "Qwen/Qwen3.8-27B",
    "frozen_train_kind": "sft",
    "progress": {"step": "5", "total": "100", "loss": "0.5"},
}


class TestEntryAndEnvelope:
    def test_config_points_at_the_cli_entry(self):
        cfg = mind_recursive.build_config(api_key="sk-x", base_url=ORIGIN, env={})
        assert cfg.platform_base_url == ORIGIN + ENTRY

    def test_whoami_uses_x_api_key_on_the_cli_entry(self, fake, uc):
        fake.route("GET", "/me", ok({"user_id": "42", "role": "user", "channel": "cli",
                                     "admission_create": True}))
        me = uc.whoami()
        assert me.user_id == "42" and me.channel == "cli" and me.admission_create
        req = fake.to("/me")[0]
        assert req.headers["x-api-key"] == "sk-test"
        assert "authorization" not in req.headers

    def test_int64_strings_become_ints(self, fake, uc):
        fake.route("GET", "/datasets/ds-1", ok(DATASET))
        ds = uc.datasets.get("ds-1")
        assert ds.rows == 3 and ds.size_bytes == 12 and ds.created_at == 1700000000000

    def test_business_error_on_http_200(self, fake, uc):
        fake.route("DELETE", "/datasets/ds-1",
                   httpx.Response(200, json={"code": 140058, "message": "referenced"}))
        with pytest.raises(mind_recursive.ConflictError) as info:
            uc.datasets.delete("ds-1")
        assert info.value.code == 140058

    def test_auth_business_code(self, fake, uc):
        fake.route("GET", "/me", httpx.Response(200, json={"code": 100020, "message": "bad"}))
        with pytest.raises(mind_recursive.AuthenticationError):
            uc.whoami()

    def test_relay_entry_rejection_keeps_its_code(self, fake, uc):
        fake.route("GET", "/me", httpx.Response(
            401, json={"error": {"code": "credential_rejected", "message": "nope"},
                       "request_id": "rq-1"}))
        with pytest.raises(mind_recursive.AuthenticationError) as info:
            uc.whoami()
        assert info.value.code == "credential_rejected"
        assert info.value.request_id == "rq-1"

    def test_non_envelope_response_is_an_error(self, fake, uc):
        fake.route("GET", "/me", httpx.Response(200, json={"user_id": "1"}))
        with pytest.raises(mind_recursive.APIError, match="envelope"):
            uc.whoami()

    def test_ids_are_escaped_as_one_segment(self, fake, uc):
        fake.route("GET", "/datasets/a%2Fb", ok(DATASET))
        uc.datasets.get("a/b")
        assert fake.requests[-1].url.raw_path.startswith(b"/api/mint/cli/v1/datasets/a%2Fb")

    def test_closed_client_refuses_platform_calls(self, uc):
        uc.close()
        with pytest.raises(mind_recursive.ConfigurationError):
            uc.training.list()


class TestDatasets:
    def test_list_with_filter_and_total(self, fake, uc):
        fake.route("GET", "/datasets", ok({"list": [DATASET], "total": "7"}))
        page = uc.datasets.list(type="sft", page=2)
        assert [d.dataset_version for d in page] == ["ds-1"] and page.total == 7
        q = fake.to("/datasets")[0].url.params
        assert q["type"] == "sft" and q["page"] == "2" and "page_size" not in q

    def test_upload_is_intent_put_complete_poll(self, fake, uc, tmp_path):
        f = tmp_path / "train.jsonl"
        f.write_bytes(b'{"messages":[]}\n')
        fake.route("POST", "/datasets/uploads", ok({
            "dataset_version": "ds-1", "status": "uploading", "max_bytes": "1000",
            "upload": {"method": "PUT", "url": OBJ + "/bucket/ds-1?sig=secret",
                       "headers": {"x-oss-forbid-overwrite": "true"}},
        }))
        fake.route("POST", "/datasets/ds-1/upload-complete",
                   ok({"dataset_version": "ds-1", "status": "building"}))

        def get_ds(_r):
            fake.dataset_polls += 1
            status = "building" if fake.dataset_polls < 2 else "available"
            return ok({**DATASET, "status": status})

        fake.route("GET", "/datasets/ds-1", get_ds)

        ds = uc.datasets.upload(f, type="sft", note="n", idempotency_key="k1")
        assert ds.available and fake.dataset_polls == 2

        intent = fake.to("/datasets/uploads")[0]
        assert intent.headers["idempotency-key"] == "k1"
        assert json.loads(intent.content)["original_filename"] == "train.jsonl"

        put = [r for r in fake.requests if r.method == "PUT"][0]
        assert "x-api-key" not in put.headers and "content-type" not in put.headers
        assert put.headers["x-oss-forbid-overwrite"] == "true"
        assert fake.objects["/bucket/ds-1"] == f.read_bytes()

        complete = fake.to("/datasets/ds-1/upload-complete")[0]
        assert complete.headers["idempotency-key"] == "k1-complete"
        assert json.loads(complete.content)["size"] == f.stat().st_size

    def test_upload_failure_raises_with_reason(self, fake, uc, tmp_path):
        f = tmp_path / "bad.jsonl"
        f.write_bytes(b"x")
        fake.route("POST", "/datasets/uploads", ok({"dataset_version": "ds-1", "upload": None}))
        fake.route("GET", "/datasets/ds-1",
                   ok({**DATASET, "status": "failed", "failure_reason": "bad json"}))
        with pytest.raises(mind_recursive.DatasetValidationError, match="bad json") as info:
            uc.datasets.upload(f)
        assert info.value.dataset.status == "failed"
        assert not [r for r in fake.requests if r.method == "PUT"]

    def test_upload_rejects_oversize_before_put(self, fake, uc, tmp_path):
        f = tmp_path / "big.jsonl"
        f.write_bytes(b"0123456789")
        fake.route("POST", "/datasets/uploads", ok({
            "dataset_version": "ds-1", "max_bytes": "5",
            "upload": {"method": "PUT", "url": OBJ + "/b/x"}}))
        with pytest.raises(mind_recursive.ConfigurationError, match="at most 5"):
            uc.datasets.upload(f)

    def test_download_writes_file_without_credentials(self, fake, uc, tmp_path):
        fake.objects["/bucket/ds-1"] = b"line1\nline2\n"
        fake.route("GET", "/datasets/ds-1/download-url",
                   ok({"url": OBJ + "/bucket/ds-1?sig=secret", "expires_at": "1"}))
        dest = tmp_path / "out.jsonl"
        assert uc.datasets.download("ds-1", dest) == str(dest)
        assert dest.read_bytes() == b"line1\nline2\n"
        get = [r for r in fake.requests if r.url.host == "oss.test"][0]
        assert "x-api-key" not in get.headers
        assert [p.name for p in tmp_path.iterdir()] == ["out.jsonl"]

    def test_download_refuses_to_overwrite(self, fake, uc, tmp_path):
        dest = tmp_path / "out.jsonl"
        dest.write_text("keep")
        with pytest.raises(FileExistsError):
            uc.datasets.download("ds-1", dest)
        assert not fake.requests and dest.read_text() == "keep"

    def test_download_error_hides_signed_url(self, fake, uc, tmp_path):
        fake.route("GET", "/datasets/ds-1/download-url",
                   ok({"url": OBJ + "/missing?sig=secret"}))
        with pytest.raises(mind_recursive.APIError) as info:
            uc.datasets.download("ds-1", tmp_path / "o.jsonl")
        assert "secret" not in str(info.value)
        assert os.listdir(tmp_path) == []

    def test_rows_and_report(self, fake, uc):
        fake.route("GET", "/datasets/ds-1/rows", ok({"rows": [{"a": 1}, {"a": 2}]}))
        fake.route("GET", "/datasets/ds-1/validation-report", ok({
            "dataset_version": "ds-1", "status": "failed", "fatal_count": "1",
            "findings": [{"severity": "fatal", "rule": "json", "line": "3"}], "total": "1"}))
        assert uc.datasets.rows("ds-1", offset=0, limit=2) == [{"a": 1}, {"a": 2}]
        rep = uc.datasets.validation_report("ds-1")
        assert rep.fatal_count == 1 and rep.findings[0].line == 3

    def test_update_note(self, fake, uc):
        fake.route("PATCH", "/datasets/ds-1", ok({**DATASET, "note": "new"}))
        assert uc.datasets.update_note("ds-1", "new").note == "new"
        assert json.loads(fake.to("/datasets/ds-1")[0].content) == {"note": "new"}


class TestTraining:
    def test_submit_body_and_idempotency_key(self, fake, uc):
        fake.route("POST", "/training/runs", ok({**EXECUTION, "state": "pending"}))
        run = uc.training.submit(dataset_version="ds-1", base_model="Qwen/Qwen3.8-27B",
                                 config={"learning_rate": 1e-4}, display_name="r1")
        assert run.execution_id == "ex-1" and run.state == "pending"
        req = fake.to("/training/runs")[0]
        assert json.loads(req.content) == {
            "dataset_version": "ds-1", "base_model": "Qwen/Qwen3.8-27B",
            "config": {"learning_rate": 1e-4}, "display_name": "r1", "train_kind": "sft"}
        assert req.headers["idempotency-key"]

    def test_submit_retries_5xx_with_the_same_key(self, fake, uc):
        calls = iter([httpx.Response(503), ok(EXECUTION)])
        fake.route("POST", "/training/runs", lambda _r: next(calls))
        uc.training.submit(dataset_version="ds-1", base_model="m")
        keys = {r.headers["idempotency-key"] for r in fake.to("/training/runs")}
        assert len(fake.to("/training/runs")) == 2 and len(keys) == 1

    def test_writes_without_a_key_are_not_retried(self, fake, uc):
        fake.route("POST", "/executions/ex-1/cancel", httpx.Response(503))
        with pytest.raises(mind_recursive.ServerError):
            uc.training.cancel("ex-1")
        assert len(fake.to("/executions/ex-1/cancel")) == 1

    def test_retry_continue(self, fake, uc):
        fake.route("POST", "/training/runs/ex-1/retry", ok({**EXECUTION, "retry_of": "ex-1"}))
        run = uc.training.retry("ex-1", continue_training=True)
        assert run.retry_of == "ex-1"
        assert json.loads(fake.to("/training/runs/ex-1/retry")[0].content) == {
            "continue_training": True}

    def test_no_separate_packages_or_executions(self, uc):
        assert not hasattr(uc, "packages") and not hasattr(uc, "executions")
        assert not hasattr(uc, "versions")


class TestTrainingRecords:
    def test_list_my_training_records(self, fake, uc):
        fake.route("GET", "/executions", ok({"list": [EXECUTION], "total": "1"}))
        page = uc.training.list(state="running")
        run = page.items[0]
        assert run.progress.step == 5 and run.train_kind == "sft" and not run.terminal
        q = fake.to("/executions")[0].url.params
        assert q["kind"] == "training" and q["state"] == "running"

    def test_wait_until_terminal(self, fake, uc):
        states = iter(["running", "running", "succeeded"])
        fake.route("GET", "/executions/ex-1",
                   lambda _r: ok({**EXECUTION, "state": next(states), "version_id": "v-1"}))
        run = uc.training.wait("ex-1", poll_interval=0.01)
        assert run.state == "succeeded" and run.version_id == "v-1"

    def test_wait_timeout(self, fake, uc, monkeypatch):
        fake.route("GET", "/executions/ex-1", ok(EXECUTION))
        clock = iter([0.0, 0.0, 100.0])
        monkeypatch.setattr("mind_recursive.platform.time.monotonic", lambda: next(clock))
        with pytest.raises(mind_recursive.FutureTimeoutError):
            uc.training.wait("ex-1", timeout=5)

    def test_iter_metrics_follows_cursor(self, fake, uc):
        pages = {
            None: {"points": [{"step": "1", "metrics": {"loss": 1.0}}], "next_cursor": "c1"},
            "c1": {"points": [{"step": "2", "metrics": {"loss": 0.5}}], "next_cursor": "c2",
                   "caught_up": True},
        }
        fake.route("GET", "/executions/ex-1/metrics",
                   lambda r: ok(pages[r.url.params.get("cursor")]))
        points = list(uc.training.iter_metrics("ex-1", keys=["loss", "lr"]))
        assert [p.step for p in points] == [1, 2] and points[1].metrics == {"loss": 0.5}
        assert fake.to("/executions/ex-1/metrics")[0].url.params.get_list("keys") == [
            "loss", "lr"]

    def test_observability_reads(self, fake, uc):
        fake.route("GET", "/executions/ex-1/progress", ok({"lines": ["a", "b"]}))
        fake.route("GET", "/executions/ex-1/metric-keys",
                   ok({"keys": [{"key": "loss", "type": "scalar", "last_step": "9"}]}))
        fake.route("GET", "/executions/ex-1/result", ok({
            "status": "succeeded", "sealed": True,
            "artifacts": [{"type": "sampler_weights", "uri": "mint://x/adapters/y"}]}))
        fake.route("GET", "/executions/ex-1/events", ok({
            "events": [{"id": "e1", "seq": "1", "type": "log", "data_json": "{}"}],
            "total": "1"}))
        assert uc.training.progress("ex-1", tail=2) == ["a", "b"]
        assert uc.training.metric_keys("ex-1")[0].last_step == 9
        res = uc.training.result("ex-1")
        assert res.sealed and res.artifacts[0].uri == "mint://x/adapters/y"
        ev = uc.training.events("ex-1", page=1)
        assert ev.total == 1 and ev.items[0].data_json == "{}"

    def test_rename_and_delete(self, fake, uc):
        fake.route("POST", "/executions/ex-1/rename", ok({**EXECUTION, "display_name": "n"}))
        fake.route("DELETE", "/executions/ex-1", ok(None))
        assert uc.training.rename("ex-1", "n").display_name == "n"
        uc.training.delete("ex-1")


class TestModels:
    def test_get_list_lineage(self, fake, uc):
        v = {"version_id": "v-2", "parent_id": "v-1", "upstream": "mint",
             "train_data_rows": "10"}
        fake.route("GET", "/versions", ok({"list": [v], "total": "1"}))
        fake.route("GET", "/versions/v-2", ok(v))
        fake.route("GET", "/versions/v-2/lineage", ok({"chain": [v, {"version_id": "v-1"}]}))
        assert uc.models.list().items[0].train_data_rows == 10
        assert uc.models.get("v-2").upstream == "mint"
        assert [x.version_id for x in uc.models.lineage("v-2")] == ["v-2", "v-1"]


def test_platform_module_exports_resolve():
    for name in P.__all__:
        assert hasattr(P, name), name


DEPLOYMENT = {
    "deployment_id": "dep-1",
    "version_id": "v-1",
    "display_name": "sft-v1",
    "desired_state": "active",
    "state": "pending",
    "generation": "1",
    "created_at": "1700000000000",
}


class TestDeployments:
    def test_create_is_sent_once(self, fake, uc):
        fake.route("POST", "/deployments", httpx.Response(503))
        with pytest.raises(mind_recursive.ServerError):
            uc.deployments.create(version_id="v-1")
        assert len(fake.to("/deployments")) == 1

    def test_create_body(self, fake, uc):
        fake.route("POST", "/deployments", ok(DEPLOYMENT))
        dep = uc.deployments.create(version_id="v-1", display_name="sft-v1")
        assert dep.deployment_id == "dep-1" and dep.state == "pending" and not dep.ready
        assert json.loads(fake.to("/deployments")[0].content) == {
            "version_id": "v-1", "display_name": "sft-v1"}

    def test_list_get_stop_start_delete(self, fake, uc):
        fake.route("GET", "/deployments", ok({"list": [DEPLOYMENT], "total": "1"}))
        fake.route("GET", "/deployments/dep-1", ok(DEPLOYMENT))
        fake.route("POST", "/deployments/dep-1/stop",
                   ok({**DEPLOYMENT, "desired_state": "stopped"}))
        fake.route("POST", "/deployments/dep-1/start", ok(DEPLOYMENT))
        fake.route("DELETE", "/deployments/dep-1", ok({**DEPLOYMENT, "state": "stopped"}))
        assert uc.deployments.list(state="pending").total == 1
        assert fake.to("/deployments")[0].url.params["state"] == "pending"
        assert uc.deployments.get("dep-1").version_id == "v-1"
        assert uc.deployments.stop("dep-1").desired_state == "stopped"
        assert uc.deployments.start("dep-1").desired_state == "active"
        assert uc.deployments.delete("dep-1").state == "stopped"

    def test_wait_until_ready(self, fake, uc):
        states = iter(["pending", "pending", "ready"])
        fake.route("GET", "/deployments/dep-1", lambda _r: ok(
            {**DEPLOYMENT, "state": next(states), "public_model_id": "u1-sft-v1"}))
        dep = uc.deployments.wait("dep-1", poll_interval=0.01)
        assert dep.ready and dep.public_model_id == "u1-sft-v1"

    def test_wait_lagging_stopped_after_start_keeps_polling(self, fake, uc):
        states = iter(["stopped", "ready"])
        fake.route("GET", "/deployments/dep-1",
                   lambda _r: ok({**DEPLOYMENT, "state": next(states)}))
        assert uc.deployments.wait("dep-1").ready

    def test_wait_unknown_raises(self, fake, uc):
        fake.route("GET", "/deployments/dep-1", ok({**DEPLOYMENT, "state": "unknown"}))
        with pytest.raises(mind_recursive.DeploymentError) as info:
            uc.deployments.wait("dep-1")
        assert info.value.deployment.state == "unknown"

    def test_wait_switched_the_other_way_raises(self, fake, uc):
        fake.route("GET", "/deployments/dep-1",
                   ok({**DEPLOYMENT, "state": "stopped", "desired_state": "stopped"}))
        with pytest.raises(mind_recursive.DeploymentError):
            uc.deployments.wait("dep-1", until="ready")

    def test_wait_until_stopped(self, fake, uc):
        fake.route("GET", "/deployments/dep-1",
                   ok({**DEPLOYMENT, "state": "stopped", "desired_state": "stopped"}))
        assert uc.deployments.wait("dep-1", until="stopped").state == "stopped"


def sse(*events: Any, done: bool = True) -> bytes:
    out = b": keep-alive\n\n"
    for e in events:
        out += b"event: x\ndata: " + json.dumps(e).encode() + b"\n\n"
    if done:
        out += b"data: [DONE]\n\n"
    return out


class TestInference:
    def test_chat_completions_create_like_openai(self, fake, uc):
        fake.routes[("POST", "/v1/chat/completions")] = lambda r: httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "hey"}}],
                       "usage": {"total_tokens": 3}})
        resp = uc.chat.completions.create(
            model="u1-sft-v1", messages=[{"role": "user", "content": "hi"}],
            max_tokens=16, temperature=0.2)
        assert resp.choices[0].message.content == "hey"
        assert resp["choices"][0]["message"]["content"] == "hey"
        assert resp.usage.total_tokens == 3 and resp.missing_field is None
        req = fake.requests[-1]
        assert req.url.path == "/v1/chat/completions"
        assert req.headers["x-api-key"] == "sk-test"
        assert json.loads(req.content) == {
            "model": "u1-sft-v1", "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16, "temperature": 0.2}

    def test_chat_stream_yields_chunk_objects(self, fake, uc):
        body = sse({"choices": [{"delta": {"content": "he"}}]},
                   {"choices": [{"delta": {"content": "y"}}]},
                   {"choices": [{"delta": {}, "finish_reason": "stop"}]})
        fake.routes[("POST", "/v1/chat/completions")] = lambda r: httpx.Response(
            200, headers={"Content-Type": "text/event-stream"}, stream=_Stream(body))
        stream = uc.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "hi"}], stream=True)
        text = "".join(c.choices[0].delta.content or "" for c in stream)
        assert text == "hey"
        req = fake.requests[-1]
        assert json.loads(req.content)["stream"] is True
        assert req.headers["accept"] == "text/event-stream"

    def test_arguments_are_keyword_only(self, uc):
        with pytest.raises(TypeError):
            uc.chat.completions.create("m", [])  # type: ignore[misc]

    def test_error_is_mapped_and_not_retried(self, fake, uc):
        fake.routes[("POST", "/v1/chat/completions")] = lambda r: httpx.Response(
            503, json={"error": {"code": 503, "message": "busy", "type": "api_error"},
                       "request_id": "rq-9"})
        with pytest.raises(mind_recursive.ServerError, match="busy") as info:
            uc.chat.completions.create(model="m", messages=[])
        assert info.value.request_id == "rq-9"
        assert len([r for r in fake.requests if r.url.path == "/v1/chat/completions"]) == 1

    def test_redirect_is_not_followed(self, fake, uc):
        fake.routes[("POST", "/v1/chat/completions")] = lambda r: httpx.Response(
            307, headers={"Location": "https://elsewhere.test/x"})
        with pytest.raises(mind_recursive.APIError):
            uc.chat.completions.create(model="m", messages=[])
        assert not [r for r in fake.requests if r.url.host == "elsewhere.test"]

    def test_closed_client_refuses_inference(self, uc):
        uc.close()
        with pytest.raises(mind_recursive.ConfigurationError):
            uc.chat.completions.create(model="m", messages=[])
        with pytest.raises(mind_recursive.ConfigurationError):
            uc.chat.completions.create(model="m", messages=[], stream=True)

    def test_no_inference_or_messages_namespace(self, uc):
        # /v1/messages does not route to deployments on the relay yet (400, code 1211).
        assert not hasattr(uc, "inference") and not hasattr(uc, "messages")


def test_sse_parser_joins_multiline_data():
    lines = iter(["data: {\"a\":", "data: 1}", "", "data: [DONE]", ""])
    (obj,) = P._iter_sse(lines)
    assert obj == {"a": 1} and obj.a == 1
