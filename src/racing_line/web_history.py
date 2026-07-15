"""JSON-backed history store for local dashboard simulations."""

from __future__ import annotations

import copy
import gzip
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from .circuits import CircuitInfo
from .strategy import RacePlan


DEFAULT_HISTORY_LIMIT = 100
_HISTORY_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class SimulationHistoryStore:
    """Persist a compact index and immutable full-result sidecars."""

    def __init__(self, path: str | Path, *, limit: int = DEFAULT_HISTORY_LIMIT):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("history limit must be a positive integer")
        self.path = Path(path)
        self.details_path = self.path.parent / f"{self.path.name}.details"
        self.limit = limit
        self._lock = threading.Lock()

    def _read_unlocked(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return []
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, dict)]

    def _write_unlocked(self, records: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(
            json.dumps(records, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    @staticmethod
    def _validate_id(run_id: str) -> str:
        if not isinstance(run_id, str) or _HISTORY_ID_PATTERN.fullmatch(run_id) is None:
            raise ValueError("invalid simulation history id")
        return run_id

    def _detail_path(self, run_id: str) -> Path:
        return self.details_path / f"{self._validate_id(run_id)}.json.gz"

    def _write_detail_unlocked(
        self,
        run_id: str,
        payload: Mapping[str, Any],
    ) -> Path:
        detail_path = self._detail_path(run_id)
        self.details_path.mkdir(parents=True, exist_ok=True)
        if detail_path.exists():
            raise FileExistsError(f"history details already exist for {run_id}")
        temporary = detail_path.with_name(f"{detail_path.name}.{uuid4().hex}.tmp")
        try:
            encoded = json.dumps(
                dict(payload),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            temporary.write_bytes(gzip.compress(encoded, mtime=0))
            if detail_path.exists():
                raise FileExistsError(f"history details already exist for {run_id}")
            temporary.replace(detail_path)
        finally:
            temporary.unlink(missing_ok=True)
        return detail_path

    def _remove_detail_unlocked(self, run_id: Any) -> None:
        if not isinstance(run_id, str) or _HISTORY_ID_PATTERN.fullmatch(run_id) is None:
            return
        self._detail_path(run_id).unlink(missing_ok=True)

    def _public_record_unlocked(
        self,
        record: Mapping[str, Any],
    ) -> dict[str, Any]:
        result = copy.deepcopy(dict(record))
        run_id = result.get("id")
        result["details_available"] = bool(
            isinstance(run_id, str)
            and _HISTORY_ID_PATTERN.fullmatch(run_id) is not None
            and self._detail_path(run_id).is_file()
        )
        return result

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                self._public_record_unlocked(record)
                for record in self._read_unlocked()[: self.limit]
            ]

    def get(self, run_id: str) -> dict[str, Any] | None:
        """Return a stored result, or a summary-only payload for legacy records."""

        validated_id = self._validate_id(run_id)
        with self._lock:
            record = next(
                (
                    item
                    for item in self._read_unlocked()[: self.limit]
                    if item.get("id") == validated_id
                ),
                None,
            )
            if record is None:
                return None
            public_record = self._public_record_unlocked(record)
            detail_path = self._detail_path(validated_id)
            if public_record["details_available"]:
                try:
                    payload = json.loads(gzip.decompress(detail_path.read_bytes()))
                except (
                    OSError,
                    EOFError,
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                ):
                    payload = None
                if isinstance(payload, dict):
                    result = copy.deepcopy(payload)
                    result["history"] = public_record
                    result["details_available"] = True
                    return result

            # History created by older versions has no detailed result sidecar.
            public_record["details_available"] = False
            return {
                "track": copy.deepcopy(record.get("circuit", {})),
                "summary": copy.deepcopy(record.get("summary", {})),
                "strategy": copy.deepcopy(record.get("strategy", {})),
                "history": public_record,
                "details_available": False,
            }

    def append(
        self,
        *,
        circuit: CircuitInfo,
        race_plan: RacePlan,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        summary = payload.get("summary", {})
        if not isinstance(summary, Mapping):
            raise ValueError("simulation payload summary must be an object")
        record: dict[str, Any] = {
            "id": uuid4().hex,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "circuit": circuit.to_dict(),
            "summary": copy.deepcopy(dict(summary)),
            "strategy": race_plan.to_dict(),
        }
        with self._lock:
            detail_path = self._write_detail_unlocked(record["id"], payload)
            try:
                records = [record, *self._read_unlocked()]
                retained = records[: self.limit]
                evicted = records[self.limit :]
                self._write_unlocked(retained)
            except Exception:
                detail_path.unlink(missing_ok=True)
                raise
            for old_record in evicted:
                self._remove_detail_unlocked(old_record.get("id"))
            return self._public_record_unlocked(record)

    def clear(self) -> int:
        with self._lock:
            records = self._read_unlocked()
            if self.path.exists():
                self.path.unlink()
            if self.details_path.is_dir():
                for detail_path in self.details_path.glob("*.json.gz"):
                    detail_path.unlink(missing_ok=True)
                for temporary in self.details_path.glob("*.json.gz.*.tmp"):
                    temporary.unlink(missing_ok=True)
                try:
                    self.details_path.rmdir()
                except OSError:
                    # Do not remove unrelated files from this app-owned directory.
                    pass
            return min(len(records), self.limit)


__all__ = ["DEFAULT_HISTORY_LIMIT", "SimulationHistoryStore"]
