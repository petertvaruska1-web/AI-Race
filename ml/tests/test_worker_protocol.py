import json

import pytest

from airace_ml.paths import REPO_ROOT
from airace_ml.train.config import TrainRunConfig
from airace_ml.worker.protocol import (
    JOB_TYPES,
    Lease,
    ProtocolError,
    WHello,
    dumps,
    parse_server_message,
    parse_worker_message,
)

FIXTURES = REPO_ROOT / "packages" / "protocol" / "fixtures"
WORKER_FIXTURES = sorted((FIXTURES / "worker").glob("*.json"))


def _fixture(path):
    d = json.loads(path.read_text(encoding="utf-8"))
    return d["schema"], d["message"]


def test_the_shared_fixtures_are_there():
    schemas = {_fixture(p)[0] for p in WORKER_FIXTURES}
    assert {"WorkerMessage", "ServerToWorker"} <= schemas
    assert len(WORKER_FIXTURES) >= 14


@pytest.mark.parametrize("path", WORKER_FIXTURES, ids=lambda p: p.stem)
def test_messages_round_trip(path):
    schema, message = _fixture(path)
    if schema == "ServerToWorker":
        assert json.loads(dumps(parse_server_message(json.dumps(message)))) == message
    elif schema == "WorkerMessage":
        assert json.loads(dumps(parse_worker_message(json.dumps(message)))) == message
    else:
        pytest.skip("a job result: checked by the job handlers' tests")


def test_a_lease_carries_its_job_type_and_spec():
    _, message = _fixture(FIXTURES / "worker" / "lease-generate.json")
    lease = parse_server_message(json.dumps(message))
    assert isinstance(lease, Lease)
    assert lease.job_type == "generate"
    assert lease.spec["history"][0] == ["user", "Tell me a story <|end|>"]


def test_a_feedback_lease_names_the_customer_groups():
    _, message = _fixture(FIXTURES / "worker" / "lease-feedback.json")
    lease = parse_server_message(json.dumps(message))
    assert isinstance(lease, Lease)
    assert lease.job_type == "feedback"
    assert "feedback" in JOB_TYPES
    assert lease.spec["groups"][0] == "students"
    assert lease.spec["perGroup"] == 1


def test_whello_names_the_capabilities_in_camel_case():
    _, message = _fixture(FIXTURES / "worker" / "whello.json")
    hello = parse_worker_message(json.dumps(message))
    assert isinstance(hello, WHello)
    assert hello.capabilities.data_identity == "sample-0001"
    wire = json.loads(dumps(hello))["capabilities"]
    assert set(wire) == {"device", "vramMb", "maxParams", "jobTypes", "slots", "dataIdentity"}


@pytest.mark.parametrize(
    "raw",
    [
        '{"v": 1, "type": "explode"}',
        '{"v": 2, "type": "wwelcome"}',
        '{"v": 1, "type": "lease", "jobId": "j", "jobType": "train", "spec": [], "leaseMs": 1}',
        '{"v": 1, "type": "lease", "jobType": "train", "spec": {}, "leaseMs": 1}',
        "not json",
        "[]",
    ],
)
def test_unknown_or_malformed_messages_raise(raw):
    with pytest.raises(ProtocolError):
        parse_server_message(raw)


def test_the_servers_default_training_config_is_a_valid_python_config():
    config = TrainRunConfig.from_json(
        (FIXTURES / "train-config.default.json").read_text(encoding="utf-8")
    )
    config.validate()
    assert config.steps == 1536
