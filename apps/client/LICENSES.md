# Third-party code, fonts and data in the web client

The client bundles these; each is used under its license (ADR `docs/decisions/0011-look-redesign.md`).

| What | Used for | License |
| --- | --- | --- |
| [three.js](https://threejs.org) | the 3D objects (the AI core, the emblem badge, the map) | MIT |
| [Barlow Semi Condensed](https://github.com/jpt/barlow) (via `@fontsource/barlow-semi-condensed`) | all text | SIL Open Font License 1.1 |
| [Doto](https://github.com/oliverlalan/Doto) (via `@fontsource/doto`) | the displays' dot-matrix readouts | SIL Open Font License 1.1 |
| [us-atlas](https://github.com/topojson/us-atlas) | the map's state outlines (`states-albers-10m.json`) | ISC; the boundaries come from the US Census Bureau's cartographic boundary files, which are in the public domain |
| [topojson-client](https://github.com/topojson/topojson-client) | reading the outlines and merging them into the game's nine regions | ISC |
