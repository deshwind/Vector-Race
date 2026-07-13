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
