# Formula 1 circuit catalogue

`f1-circuits.geojson` and `f1-locations-2026.json` are numerically unmodified
files from [`bacinger/f1-circuits`](https://github.com/bacinger/f1-circuits),
pinned at commit `394d8fbe70ef2c0b0c8d23ff7bee61fa09606055`.

The pinned file SHA-256 digests are:

- `f1-circuits.geojson`: `a0c8dfb3109a9181d096985eaa30bd692595eae9125b5b8686744600b24621b5`
- `f1-locations-2026.json`: `d7ff1bdabbc14dc88b94a257dbc5a3804cebd3aff0333593c2615c3709c04286`

The 2026 index selects 24 announced venues from the combined GeoJSON file.
The data is unofficial and contains geographic centrelines and nominal circuit
lengths, but no surveyed left/right boundary widths. At runtime,
`racing_line.circuits.make_f1_circuit` uses an equirectangular local projection,
scales each polyline to its supplied nominal length, and applies a provisional
constant `7 m` width on each side of the centreline.

The source is MIT-licensed. See `THIRD_PARTY_NOTICES.md` and
`THIRD_PARTY_LICENSES/bacinger-f1-circuits-MIT.txt` in the repository root.
