import threading

import pytest

from airace_ml import awake
from airace_ml.awake import ES_CONTINUOUS, ES_SYSTEM_REQUIRED, stay_awake

NEEDED = ES_CONTINUOUS | ES_SYSTEM_REQUIRED


@pytest.fixture
def asked(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(awake, "_set_state", calls.append)
    return calls


def test_the_system_is_needed_for_the_block_and_released_after(asked):
    with stay_awake():
        assert asked == [NEEDED]
    assert asked == [NEEDED, ES_CONTINUOUS]


def test_nested_blocks_ask_once_and_release_once(asked):
    with stay_awake():
        with stay_awake():
            assert asked == [NEEDED]
        assert asked == [NEEDED]  # the outer block still holds the request
    assert asked == [NEEDED, ES_CONTINUOUS]


def test_work_that_fails_still_releases_the_system(asked):
    with pytest.raises(RuntimeError), stay_awake():
        raise RuntimeError("the run crashed")
    assert asked == [NEEDED, ES_CONTINUOUS]
    with stay_awake():  # and the next block asks again
        pass
    assert asked == [NEEDED, ES_CONTINUOUS, NEEDED, ES_CONTINUOUS]


def test_each_thread_holds_its_own_request(asked):
    # The state belongs to a thread, so a job's thread asks even while another block is open.
    def job() -> None:
        with stay_awake():
            pass

    with stay_awake():
        t = threading.Thread(target=job)
        t.start()
        t.join()
        assert asked == [NEEDED, NEEDED, ES_CONTINUOUS]  # the job's own request and release
    assert asked == [NEEDED, NEEDED, ES_CONTINUOUS, ES_CONTINUOUS]


def test_a_refused_request_never_fails_the_work(monkeypatch):
    def refuse(flags: int) -> None:
        raise OSError("no such call here")

    monkeypatch.setattr(awake, "_set_state", refuse)
    with stay_awake():
        done = True
    assert done


def test_the_real_request_works_on_this_system():
    with stay_awake():  # Windows: the real call; elsewhere: nothing
        pass
