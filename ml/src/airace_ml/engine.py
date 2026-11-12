"""The training engine's version: what trains a run.

A resume state is only ever continued by the engine that saved it, every run's summary names its
engine, and a helper tells the game which engine it runs, so a game never hands training to a
helper that would train differently (``packages/protocol/src/worker.ts`` holds the game's side;
the shared ``whello`` example ties the two numbers together).

* 1: AdamW, the last fifth of the run decays to a tenth of the peak rate.
* 2 (2026-11): the MuonAdamW optimizer, the last two fifths decay to a fiftieth, chat documents
  are read with casual typing (ADR 0009).
"""

ENGINE_VERSION = 2
