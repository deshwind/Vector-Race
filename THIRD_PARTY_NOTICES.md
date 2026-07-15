# Third-party notices

## Silverstone circuit dataset

The bundled `src/racing_line/data/Silverstone.csv` comes from the Technical
University of Munich's
[`TUMFTM/racetrack-database`](https://github.com/TUMFTM/racetrack-database),
pinned at commit `e59595d1f3573b30d1ded6a08984935b957688e0`.

The upstream repository distributes the dataset under GNU LGPL v3. A copy is
provided in
`THIRD_PARTY_LICENSES/TUMFTM-racetrack-database-LGPL-3.0.txt`.

Upstream states that its centerlines originated from OpenStreetMap. The
underlying contribution is © OpenStreetMap contributors and is made available
under the Open Data Commons Open Database License; see the
[OpenStreetMap copyright and license page](https://www.openstreetmap.org/copyright).

The upstream width-estimation process used satellite imagery whose provider is
not identified in the repository. The bundled geometry should therefore be
treated as a research reconstruction rather than an official or survey-grade
Silverstone circuit map.

## Formula 1 circuit catalogue

The bundled `src/racing_line/data/f1_circuits/` GeoJSON and 2026 season index
come from Tomislav Bacinger's
[`bacinger/f1-circuits`](https://github.com/bacinger/f1-circuits), pinned at
commit `394d8fbe70ef2c0b0c8d23ff7bee61fa09606055`.

The upstream repository distributes the data under the MIT License. A copy is
provided in `THIRD_PARTY_LICENSES/bacinger-f1-circuits-MIT.txt`.

The selected season index contains 24 announced 2026 venues. These unofficial
geographic centrelines do not include surveyed track boundaries. The web
simulator projects them into local metre coordinates, scales each line to the
source's nominal circuit length, and applies a provisional constant research
width. They are suitable for comparative visualization and simulation, not
real-world navigation, engineering, or safety work.
