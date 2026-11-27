"""Write ASCII DXF (R12 entity subset: ``POLYLINE``/``VERTEX``, ``3DFACE``, ``TEXT``).

R12 entities are read by every CAD and mining package. Only the ``ENTITIES``
section is written (allowed by the DXF specification); layers are created
implicitly by name. Coordinates use 6 decimals.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

Floats = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Polyline:
    layer: str
    points: Floats  # n x 3
    closed: bool = True


@dataclass(frozen=True)
class Text:
    layer: str
    position: Floats  # x, y, z of the insertion point
    text: str
    height: float = 2.5
    rotation: float = 0.0  # degrees counter-clockwise from +x


def _num(v: float) -> str:
    return f"{float(v):.6f}"


def _layer(name: str) -> str:
    bad = '<>/\\":;?*|=`'
    clean = "".join("_" if c in bad or ord(c) < 32 else c for c in name.strip())
    return clean or "0"


def write_dxf(
    path: Path,
    polylines: Iterable[Polyline] = (),
    faces: Sequence[tuple[str, Floats]] = (),
    texts: Iterable[Text] = (),
) -> int:
    """``faces``: (layer, m x 3 x 3 triangle corners). Returns the entity count."""
    out: list[str] = ["0", "SECTION", "2", "ENTITIES"]
    count = 0
    for line in polylines:
        pts = np.asarray(line.points, np.float64)
        if len(pts) < 2:
            continue
        out += ["0", "POLYLINE", "8", _layer(line.layer), "66", "1", "70"]
        out.append(str(8 | (1 if line.closed else 0)))  # 8 = 3D polyline
        out += ["10", "0.0", "20", "0.0", "30", "0.0"]
        for x, y, z in pts:
            out += ["0", "VERTEX", "8", _layer(line.layer), "70", "32"]
            out += ["10", _num(x), "20", _num(y), "30", _num(z)]
        out += ["0", "SEQEND", "8", _layer(line.layer)]
        count += 1
    for layer, tris in faces:
        name = _layer(layer)
        for tri in np.asarray(tris, np.float64).reshape(-1, 3, 3):
            out += ["0", "3DFACE", "8", name]
            for k, corner in enumerate((tri[0], tri[1], tri[2], tri[2])):
                out += [f"1{k}", _num(corner[0]), f"2{k}", _num(corner[1])]
                out += [f"3{k}", _num(corner[2])]
            count += 1
    for t in texts:
        x, y, z = (float(v) for v in t.position)
        out += ["0", "TEXT", "8", _layer(t.layer)]
        out += ["10", _num(x), "20", _num(y), "30", _num(z), "40", _num(t.height)]
        out += ["1", t.text.replace("\n", " "), "50", _num(t.rotation)]
        count += 1
    out += ["0", "ENDSEC", "0", "EOF"]
    path.write_text("\n".join(out) + "\n", encoding="ascii", errors="replace")
    return count


# -- reading -----------------------------------------------------------------


@dataclass(frozen=True)
class DxfEntities:
    """Geometry read from the ``ENTITIES`` section of an ASCII DXF.

    ``points``: (layer, x, y, z) of ``POINT`` entities; ``polylines``: ``LINE``,
    ``POLYLINE`` (2-D and 3-D) and ``LWPOLYLINE`` (elevation from group 38);
    ``faces``: ``3DFACE`` and polyface-mesh triangles as (layer, 3 x 3) corners.
    Polygon meshes, splines, arcs and block inserts are counted in ``skipped``.
    """

    points: list[tuple[str, Floats]]
    polylines: list[Polyline]
    faces: list[tuple[str, Floats]]
    skipped: dict[str, int]

    def layers(self) -> list[str]:
        names = {p[0] for p in self.points} | {p.layer for p in self.polylines}
        return sorted(names | {f[0] for f in self.faces})


class DxfError(ValueError):
    """Unreadable DXF content."""


def _pairs(path: Path) -> list[tuple[str, str]]:
    raw = path.read_bytes()
    if raw.startswith(b"AutoCAD Binary DXF"):
        raise DxfError(f"{path.name}: binary DXF is not supported; save as ASCII DXF")
    text = raw.decode("utf-8", errors="replace")
    lines = [s.strip() for s in text.splitlines()]
    if len(lines) % 2:
        lines = lines[:-1]
    return list(zip(lines[0::2], lines[1::2], strict=True))


def _entities_section(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Group pairs of the ENTITIES section (the whole file if it has no sections)."""
    start = None
    for k in range(len(pairs) - 1):
        if pairs[k] == ("0", "SECTION") and pairs[k + 1] == ("2", "ENTITIES"):
            start = k + 2
            break
    if start is None:
        if any(p == ("0", "SECTION") for p in pairs):
            raise DxfError("DXF has no ENTITIES section")
        return pairs
    end = next((k for k in range(start, len(pairs)) if pairs[k] == ("0", "ENDSEC")), len(pairs))
    return pairs[start:end]


def _xyz(groups: list[tuple[str, str]], base: int) -> Floats:
    p = [0.0, 0.0, 0.0]
    for c, v in groups:
        for axis in range(3):
            if c == str(base + 10 * axis):
                p[axis] = float(v)
    return np.array(p)


def read_dxf(path: Path) -> DxfEntities:  # noqa: PLR0912, PLR0915 - one DXF pass
    """Read points, polylines and faces from an ASCII DXF (any release)."""
    try:
        pairs = _entities_section(_pairs(path))
    except UnicodeError as exc:  # pragma: no cover - decode uses errors="replace"
        raise DxfError(f"{path.name}: {exc}") from exc
    # Split into entities: each starts at a group-0 pair.
    starts = [k for k, (c, _) in enumerate(pairs) if c == "0"]
    entities = [
        (pairs[s][1], pairs[s + 1 : (starts[n + 1] if n + 1 < len(starts) else len(pairs))])
        for n, s in enumerate(starts)
    ]
    points: list[tuple[str, Floats]] = []
    polylines: list[Polyline] = []
    faces: list[tuple[str, Floats]] = []
    skipped: dict[str, int] = {}

    def layer_of(groups: list[tuple[str, str]]) -> str:
        return next((v for c, v in groups if c == "8"), "0")

    def flag_of(groups: list[tuple[str, str]]) -> int:
        return int(next((v for c, v in groups if c == "70"), "0") or 0)

    i = 0
    try:
        while i < len(entities):
            kind, groups = entities[i]
            i += 1
            if kind == "POINT":
                points.append((layer_of(groups), _xyz(groups, 10)))
            elif kind == "LINE":
                a, b = _xyz(groups, 10), _xyz(groups, 11)
                polylines.append(Polyline(layer_of(groups), np.vstack([a, b]), False))
            elif kind == "3DFACE":
                q = [_xyz(groups, 10 + k) for k in range(4)]
                tris = [np.vstack(q[:3])]
                if np.linalg.norm(q[3] - q[2]) > 1e-12:
                    tris.append(np.vstack([q[0], q[2], q[3]]))
                faces += [(layer_of(groups), t) for t in tris]
            elif kind == "LWPOLYLINE":
                pts: list[list[float]] = []
                elev = 0.0
                for c, v in groups:
                    if c == "38":
                        elev = float(v)
                    elif c == "10":
                        pts.append([float(v), 0.0])
                    elif c == "20" and pts:
                        pts[-1][1] = float(v)
                if len(pts) >= 2:
                    arr = np.column_stack([np.array(pts), np.full(len(pts), elev)])
                    polylines.append(Polyline(layer_of(groups), arr, bool(flag_of(groups) & 1)))
            elif kind == "POLYLINE":
                flag, layer = flag_of(groups), layer_of(groups)
                elev = float(next((v for c, v in groups if c == "30"), "0") or 0.0)
                vertices: list[Floats] = []
                face_refs: list[list[int]] = []
                while i < len(entities) and entities[i][0] == "VERTEX":
                    vg = entities[i][1]
                    vflag = flag_of(vg)
                    if flag & 64 and vflag & 128 and not vflag & 64:
                        refs = [int(v) for c, v in vg if c in ("71", "72", "73", "74")]
                        face_refs.append([abs(r) for r in refs if r != 0])
                    else:
                        vertices.append(_xyz(vg, 10))
                    i += 1
                if i < len(entities) and entities[i][0] == "SEQEND":
                    i += 1
                if flag & 16:
                    skipped["POLYLINE (polygon mesh)"] = (
                        skipped.get("POLYLINE (polygon mesh)", 0) + 1
                    )
                elif flag & 64:
                    for refs in face_refs:
                        corners = [vertices[r - 1] for r in refs if 0 < r <= len(vertices)]
                        for k in range(1, len(corners) - 1):
                            faces.append(
                                (layer, np.vstack([corners[0], corners[k], corners[k + 1]]))
                            )
                elif len(vertices) >= 2:
                    arr = np.vstack(vertices)
                    if not flag & 8:  # 2-D polyline: z from the entity elevation
                        arr[:, 2] = elev
                    polylines.append(Polyline(layer, arr, bool(flag & 1)))
            elif kind in ("VERTEX", "SEQEND"):
                continue
            else:
                skipped[kind] = skipped.get(kind, 0) + 1
    except (ValueError, IndexError) as exc:
        raise DxfError(f"{path.name}: malformed entity ({exc})") from exc
    return DxfEntities(points, polylines, faces, skipped)
