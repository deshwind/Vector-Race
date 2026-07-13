from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from racing_line.track import Track, make_demo_track, make_silverstone_track


def test_demo_track_is_deterministic_and_well_formed() -> None:
    first = make_demo_track(point_count=64)
    second = make_demo_track(point_count=64)

    assert first.name == "synthetic_club_circuit"
    assert first.point_count == 64
    assert first.length_m > 0.0
    assert np.all(first.segment_lengths_m > 0.0)
    np.testing.assert_array_equal(first.x_m, second.x_m)
    np.testing.assert_array_equal(first.y_m, second.y_m)
    assert not first.x_m.flags.writeable


def test_track_csv_round_trip_including_local_grip(tmp_path: Path) -> None:
    demo = make_demo_track(point_count=48)
    track = Track(
        demo.x_m,
        demo.y_m,
        demo.width_left_m,
        demo.width_right_m,
        local_grip_mu=np.linspace(1.2, 1.8, demo.point_count),
        name="source_name",
    )

    destination = track.to_csv(tmp_path / "nested" / "round_trip.csv")
    restored = Track.from_csv(destination)

    assert destination.exists()
    assert restored.name == "round_trip"
    assert restored.point_count == track.point_count
    for field in (
        "x_m",
        "y_m",
        "width_left_m",
        "width_right_m",
        "local_grip_mu",
    ):
        np.testing.assert_allclose(
            getattr(restored, field), getattr(track, field), rtol=1.0e-14
        )


def test_track_removes_a_duplicated_closure_point() -> None:
    demo = make_demo_track(point_count=32)

    closed = Track(
        np.append(demo.x_m, demo.x_m[0]),
        np.append(demo.y_m, demo.y_m[0]),
        np.append(demo.width_left_m, demo.width_left_m[0]),
        np.append(demo.width_right_m, demo.width_right_m[0]),
    )

    assert closed.point_count == demo.point_count
    np.testing.assert_array_equal(closed.x_m, demo.x_m)


def test_demo_track_rejects_too_few_points() -> None:
    with pytest.raises(ValueError, match="at least 32"):
        make_demo_track(point_count=31)


def test_bundled_silverstone_track_geometry_and_direction() -> None:
    track = make_silverstone_track()
    following_x = np.roll(track.x_m, -1)
    following_y = np.roll(track.y_m, -1)
    signed_area = 0.5 * np.sum(
        track.x_m * following_y - following_x * track.y_m
    )

    assert track.name == "Silverstone Circuit GP"
    assert track.point_count == 1178
    assert track.length_m == pytest.approx(5886.805, abs=0.01)
    assert signed_area < 0.0  # Clockwise, matching the Grand Prix direction.
    assert np.min(track.width_left_m) == pytest.approx(5.753)
    assert np.max(track.width_left_m) == pytest.approx(8.990)
    assert np.min(track.width_right_m) == pytest.approx(5.415)
    assert np.max(track.width_right_m) == pytest.approx(8.851)


def test_silverstone_track_can_be_periodically_resampled() -> None:
    track = make_silverstone_track(point_count=220)

    assert track.point_count == 220
    assert track.length_m == pytest.approx(5886.805, rel=3.0e-3)
    assert np.std(track.segment_lengths_m) / np.mean(track.segment_lengths_m) < 1.0e-2
    assert np.all(track.width_left_m > 0.0)
    assert np.all(track.width_right_m > 0.0)


@pytest.mark.parametrize("point_count", [31, 20.5, True])
def test_silverstone_track_rejects_invalid_point_count(
    point_count: int | float | bool,
) -> None:
    with pytest.raises(ValueError, match="point_count|at least 32"):
        make_silverstone_track(point_count=point_count)  # type: ignore[arg-type]
