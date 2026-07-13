# Silverstone circuit data

`Silverstone.csv` is a numerically unmodified dataset from the Technical
University of Munich's `TUMFTM/racetrack-database` project, pinned at commit
`e59595d1f3573b30d1ded6a08984935b957688e0`:

[Pinned Silverstone source](https://github.com/TUMFTM/racetrack-database/blob/e59595d1f3573b30d1ded6a08984935b957688e0/tracks/Silverstone.csv)

The unmodified CSV has SHA-256 digest
`27c51454e81780a75c353d3e6c9f73dbbcd1e82139782e95bd722ade19736e39`.

The source schema is:

```text
x_m, y_m, w_tr_right_m, w_tr_left_m
```

`racing_line.track.make_silverstone_track` maps the final two columns into
this project's left/right convention without altering the source file. The
source contains 1,178 clockwise points, has a polyline length of 5,886.805 m,
and represents the modern 18-turn Grand Prix/Arena layout. It was derived
upstream circa 2020 from smoothed OpenStreetMap GPS centerline data and
satellite-image track-width estimates. Its authors caution that source-data
quality varies by circuit, so this is suitable for research simulation rather
than survey-grade circuit reconstruction.

The dataset is distributed under GNU LGPL v3 by its upstream project. A copy
is included at
`THIRD_PARTY_LICENSES/TUMFTM-racetrack-database-LGPL-3.0.txt`. The surrounding
project code remains MIT-licensed.

The underlying OpenStreetMap contribution is © OpenStreetMap contributors and
is available under the Open Data Commons Open Database License (ODbL); see
[OpenStreetMap copyright and license](https://www.openstreetmap.org/copyright).
