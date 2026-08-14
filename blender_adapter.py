#!/usr/bin/env python3
"""Adapt an AniMind renderer-neutral storyboard to Blender.

This is a dual-mode program:

* Normal Python performs validation, a dry-run plan, narration, and assembly.
* With ``--render``, it launches Blender in the background and runs this same
  file inside Blender to construct and render the selected scenes.

The adapter prefers locally downloaded, license-tracked production assets and
falls back to procedural geometry only when allowed by its asset policy.  A
separate Blender asset catalog can reference BlenderKit/Blendkit collections,
Poly Haven PBR textures and HDRIs, Mixamo FBX characters, glTF/GLB models, and
OBJ models without adding renderer-specific fields to the neutral JSON.

Examples
--------
Inspect the plan without opening Blender::

    python3 blender_adapter.py renderer_neutral_sb.json

Render only the first scene at draft quality::

    python3 blender_adapter.py renderer_neutral_sb.json \
      --render --scene scene_001 --quality draft

Render the complete storyboard::

    python3 blender_adapter.py renderer_neutral_sb.json \
      --render --quality preview
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


# Blender's embedded Python does not reliably add the directory containing a
# --python script to sys.path. Add it explicitly so companion modules remain
# importable regardless of Blender's working directory.
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))


EXPECTED_STORYBOARD_KIND = "animind.renderer-neutral-storyboard"
EXPECTED_STORYBOARD_VERSION = "1.0"
CATALOG_VERSION = "2.0"
SUPPORTED_CATALOG_VERSIONS = {"1.0", "2.0"}
STATE_VERSION = 1
ADAPTER_VERSION = "2.0.0"
GROUND_SURFACE_Z = -0.02
GROUND_CLEARANCE = 0.02
SUPPORTED_ASSET_FORMATS = {"blend", "fbx", "gltf", "glb", "obj"}


class BlenderAdapterError(RuntimeError):
    """Raised for a safe, user-facing adapter failure."""


@dataclass(frozen=True)
class Quality:
    name: str
    width: int
    height: int
    fps: int
    samples: int
    description: str


@dataclass(frozen=True)
class CameraProfile:
    name: str
    lens_mm: float
    distance: float
    aperture_fstop: float
    focus_height: float
    clip_start: float
    height_factor: float


MOVE_WORDS = {
    "advance", "approach", "crawl", "cross", "continue", "fly", "haul",
    "move", "run", "scuttle", "slide", "swim", "travel", "walk",
}
CLIMB_WORDS = {"ascend", "climb", "rise"}
LOWER_WORDS = {"descend", "drop", "fall", "lower"}
RAISE_WORDS = {"hoist", "lift", "raise"}
TURN_WORDS = {"look", "orient", "pause", "search", "turn", "watch"}
TWITCH_WORDS = {"flinch", "nod", "shake", "shiver", "tremble", "twitch"}
TOUCH_WORDS = {"click", "press", "push", "tap", "touch"}
APPEAR_WORDS = {"appear", "arrive", "enter", "materialize", "reveal"}
DISAPPEAR_WORDS = {"depart", "disappear", "exit", "fade", "vanish"}
SLIP_WORDS = {"recover", "slip", "stumble", "trip"}


def _program_arguments() -> List[str]:
    """Return user arguments both in regular Python and inside Blender."""
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    return sys.argv[1:]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adapt a renderer-neutral AniMind storyboard into an asset-backed "
            "or procedural Blender animation. Dry run is the default."
        )
    )
    parser.add_argument(
        "storyboard",
        type=Path,
        help="Renderer-neutral JSON created by story_board_maker.py",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--render",
        action="store_true",
        help="Launch Blender, render scenes, create narration, and assemble an MP4",
    )
    mode.add_argument(
        "--assemble-only",
        action="store_true",
        help="Skip Blender and assemble scene videos that already exist",
    )
    parser.add_argument(
        "--scene",
        help="Render or assemble one scene ID instead of the complete storyboard",
    )
    parser.add_argument(
        "--quality",
        choices=("draft", "preview", "final"),
        default="draft",
        help="Render profile (default: draft)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: outputs/blender beside the storyboard)",
    )
    parser.add_argument(
        "--asset-catalog",
        type=Path,
        help="Optional Blender-specific asset mapping JSON",
    )
    parser.add_argument(
        "--strict-assets",
        action="store_true",
        help="Stop instead of using procedural geometry for an unmapped entity",
    )
    parser.add_argument(
        "--camera-style",
        choices=("auto", "macro", "close", "medium", "wide", "overhead", "handheld"),
        default="auto",
        help="Override storyboard shot selection while preserving its subject (default: auto)",
    )
    parser.add_argument(
        "--no-motion-blur",
        action="store_true",
        help="Disable cinematic Eevee motion blur",
    )
    parser.add_argument(
        "--no-depth-of-field",
        action="store_true",
        help="Disable camera depth of field",
    )
    parser.add_argument(
        "--blender",
        type=Path,
        help="Path to Blender when it cannot be found automatically",
    )
    parser.add_argument(
        "--narration",
        choices=("auto", "on", "off"),
        default="auto",
        help="Use macOS 'say' for narration when available (default: auto)",
    )
    parser.add_argument(
        "--voice",
        default="Samantha",
        help="macOS narration voice (default: Samantha)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        help="Override the frame rate selected by the quality profile",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing Blender scene and final video outputs",
    )
    parser.add_argument(
        "--no-save-blend",
        action="store_true",
        help="Do not save a reusable .blend file for each rendered scene",
    )
    parser.add_argument(
        "--no-assemble",
        action="store_true",
        help="Render individual scene videos but do not assemble a final MP4",
    )
    parser.add_argument(
        "--inside-blender",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def read_json(path: Path, label: str) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise BlenderAdapterError(f"{label} not found: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise BlenderAdapterError(f"{label} must be UTF-8 JSON.") from exc
    except json.JSONDecodeError as exc:
        raise BlenderAdapterError(
            f"Invalid {label} JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise BlenderAdapterError(f"{label} root must be a JSON object.")
    return value


def validate_storyboard(storyboard: Dict[str, Any]) -> None:
    """Validate the neutral contract without importing the storyboard generator.

    Blender uses its own embedded Python interpreter, so the rendering adapter
    must not depend on a sibling module being importable at runtime.
    """
    def valid_text(value: Any, label: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise BlenderAdapterError(
                f"Invalid renderer-neutral storyboard: {label} must be a non-empty string."
            )
        return value

    def numeric(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    if storyboard.get("kind") != EXPECTED_STORYBOARD_KIND:
        raise BlenderAdapterError(
            f"Unsupported storyboard kind {storyboard.get('kind')!r}; expected "
            f"{EXPECTED_STORYBOARD_KIND!r}."
        )
    if storyboard.get("schemaVersion") != EXPECTED_STORYBOARD_VERSION:
        raise BlenderAdapterError(
            f"Unsupported storyboard schemaVersion {storyboard.get('schemaVersion')!r}; "
            f"expected {EXPECTED_STORYBOARD_VERSION!r}."
        )

    project = storyboard.get("project")
    if not isinstance(project, dict):
        raise BlenderAdapterError(
            "Invalid renderer-neutral storyboard: project must be an object."
        )
    valid_text(project.get("title"), "project.title")
    valid_text(project.get("visualStyle"), "project.visualStyle")
    frame = project.get("frame")
    if not isinstance(frame, dict):
        raise BlenderAdapterError(
            "Invalid renderer-neutral storyboard: project.frame must be an object."
        )
    for key in ("width", "height", "framesPerSecond"):
        value = frame.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise BlenderAdapterError(
                f"Invalid renderer-neutral storyboard: project.frame.{key} "
                "must be a positive integer."
            )

    entities = storyboard.get("entities")
    locations = storyboard.get("locations")
    if not isinstance(entities, list) or not isinstance(locations, list):
        raise BlenderAdapterError(
            "Invalid renderer-neutral storyboard: entities and locations must be arrays."
        )
    entity_ids = set()
    for index, entity in enumerate(entities):
        if not isinstance(entity, dict):
            raise BlenderAdapterError(
                f"Invalid renderer-neutral storyboard: entities[{index}] must be an object."
            )
        entity_id = valid_text(entity.get("entityId"), f"entities[{index}].entityId")
        if entity_id in entity_ids:
            raise BlenderAdapterError(
                f"Invalid renderer-neutral storyboard: duplicate entityId {entity_id!r}."
            )
        entity_ids.add(entity_id)

    location_ids = set()
    for index, location in enumerate(locations):
        if not isinstance(location, dict):
            raise BlenderAdapterError(
                f"Invalid renderer-neutral storyboard: locations[{index}] must be an object."
            )
        location_id = valid_text(
            location.get("locationId"), f"locations[{index}].locationId"
        )
        if location_id in location_ids:
            raise BlenderAdapterError(
                f"Invalid renderer-neutral storyboard: duplicate locationId {location_id!r}."
            )
        location_ids.add(location_id)

    scenes = storyboard.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise BlenderAdapterError(
            "Invalid renderer-neutral storyboard: scenes must be a non-empty array."
        )
    scene_ids = set()
    expected_start = 0.0
    for index, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            raise BlenderAdapterError(
                f"Invalid renderer-neutral storyboard: scenes[{index}] must be an object."
            )
        scene_id = valid_text(scene.get("sceneId"), f"scenes[{index}].sceneId")
        if scene_id in scene_ids:
            raise BlenderAdapterError(
                f"Invalid renderer-neutral storyboard: duplicate sceneId {scene_id!r}."
            )
        scene_ids.add(scene_id)
        if scene.get("sequence") != index + 1:
            raise BlenderAdapterError(
                "Invalid renderer-neutral storyboard: scene sequence values must "
                "start at 1 and be consecutive."
            )
        duration = scene.get("durationSeconds")
        start = scene.get("startSeconds")
        end = scene.get("endSeconds")
        if not numeric(duration) or float(duration) <= 0:
            raise BlenderAdapterError(
                f"Invalid renderer-neutral storyboard: {scene_id}.durationSeconds "
                "must be positive."
            )
        if not numeric(start) or not numeric(end):
            raise BlenderAdapterError(
                f"Invalid renderer-neutral storyboard: {scene_id} must have numeric "
                "start and end times."
            )
        if abs(float(start) - expected_start) > 0.001:
            raise BlenderAdapterError(
                f"Invalid renderer-neutral storyboard: {scene_id} does not begin "
                "at the preceding scene's end."
            )
        if abs(float(end) - (float(start) + float(duration))) > 0.001:
            raise BlenderAdapterError(
                f"Invalid renderer-neutral storyboard: {scene_id} has inconsistent timing."
            )
        expected_start = float(end)

        narration = scene.get("narration")
        visual = scene.get("visual")
        if not isinstance(narration, dict) or not isinstance(visual, dict):
            raise BlenderAdapterError(
                f"Invalid renderer-neutral storyboard: {scene_id} must contain "
                "narration and visual objects."
            )
        valid_text(narration.get("text"), f"{scene_id}.narration.text")
        location_id = visual.get("locationId")
        if location_id is not None and location_id not in location_ids:
            raise BlenderAdapterError(
                f"Invalid renderer-neutral storyboard: {scene_id} references "
                f"unknown locationId {location_id!r}."
            )
        referenced_entities = visual.get("entityIds")
        if not isinstance(referenced_entities, list) or any(
            not isinstance(value, str) for value in referenced_entities
        ):
            raise BlenderAdapterError(
                f"Invalid renderer-neutral storyboard: {scene_id}.visual.entityIds "
                "must be an array of strings."
            )
        missing_entities = sorted(set(referenced_entities) - entity_ids)
        if missing_entities:
            raise BlenderAdapterError(
                f"Invalid renderer-neutral storyboard: {scene_id} references "
                f"unknown entities {missing_entities}."
            )
        actions = visual.get("actions")
        if not isinstance(actions, list) or not actions:
            raise BlenderAdapterError(
                f"Invalid renderer-neutral storyboard: {scene_id} must contain an action."
            )
        action_ids = set()
        for action_index, action in enumerate(actions):
            if not isinstance(action, dict):
                raise BlenderAdapterError(
                    f"Invalid renderer-neutral storyboard: {scene_id}.visual.actions"
                    f"[{action_index}] must be an object."
                )
            action_id = valid_text(
                action.get("actionId"),
                f"{scene_id}.visual.actions[{action_index}].actionId",
            )
            if action_id in action_ids:
                raise BlenderAdapterError(
                    f"Invalid renderer-neutral storyboard: duplicate actionId "
                    f"{action_id!r} in {scene_id}."
                )
            action_ids.add(action_id)
            valid_text(action.get("verb"), f"{action_id}.verb")
            valid_text(action.get("description"), f"{action_id}.description")
            for reference_key in ("actorId", "targetId"):
                reference = action.get(reference_key)
                if reference is not None and reference not in entity_ids:
                    raise BlenderAdapterError(
                        f"Invalid renderer-neutral storyboard: {action_id}."
                        f"{reference_key} references unknown entity {reference!r}."
                    )

    timeline = storyboard.get("timeline")
    if not isinstance(timeline, dict):
        raise BlenderAdapterError(
            "Invalid renderer-neutral storyboard: timeline must be an object."
        )
    if timeline.get("sceneOrder") != [scene["sceneId"] for scene in scenes]:
        raise BlenderAdapterError(
            "Invalid renderer-neutral storyboard: timeline.sceneOrder must match scenes."
        )
    final_duration = timeline.get("finalDurationSeconds")
    if not numeric(final_duration) or abs(float(final_duration) - expected_start) > 0.001:
        raise BlenderAdapterError(
            "Invalid renderer-neutral storyboard: timeline final duration is inconsistent."
        )


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise BlenderAdapterError(f"{label} must be a JSON object.")
    return value


def require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BlenderAdapterError(f"{label} must be a non-empty string.")
    return value


def catalog_asset_path(catalog_path: Path, value: Any, label: str) -> Path:
    text = require_text(value, label)
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = catalog_path.parent / candidate
    return candidate.resolve()


def asset_entry_enabled(entry: Mapping[str, Any]) -> bool:
    return entry.get("enabled", True) is not False


def asset_entry_format(entry: Mapping[str, Any]) -> str:
    declared = entry.get("format")
    if isinstance(declared, str) and declared.strip():
        return declared.strip().lower().lstrip(".")
    value = entry.get("file", entry.get("blendFile"))
    if isinstance(value, str):
        return Path(value).suffix.lower().lstrip(".") or "blend"
    return ""


def asset_entry_file(entry: Mapping[str, Any]) -> Any:
    return entry.get("file", entry.get("blendFile"))


def asset_source_label(entry: Mapping[str, Any]) -> str:
    provider = str(entry.get("provider", "local")).strip() or "local"
    asset_format = asset_entry_format(entry) or "asset"
    return f"{provider} {asset_format.upper()}"


def read_asset_catalog(path: Optional[Path]) -> Tuple[Dict[str, Any], Optional[Path]]:
    if path is None:
        return {
            "schemaVersion": CATALOG_VERSION,
            "fallbackPolicy": "warn",
            "environment": {},
            "entities": {},
            "actionAliases": {},
        }, None
    resolved = path.expanduser().resolve()
    catalog = read_json(resolved, "Asset catalog")
    version = catalog.get("schemaVersion")
    if version not in SUPPORTED_CATALOG_VERSIONS:
        raise BlenderAdapterError(
            f"Unsupported asset catalog schemaVersion {catalog.get('schemaVersion')!r}; "
            f"expected one of {sorted(SUPPORTED_CATALOG_VERSIONS)!r}."
        )
    entities = catalog.get("entities", {})
    aliases = catalog.get("actionAliases", {})
    environment = catalog.get("environment", {})
    if not isinstance(entities, dict) or not isinstance(aliases, dict) or not isinstance(environment, dict):
        raise BlenderAdapterError(
            "Asset catalog entities, environment, and actionAliases must be objects."
        )
    fallback_policy = catalog.get("fallbackPolicy", "warn")
    if fallback_policy not in {"procedural", "warn", "error"}:
        raise BlenderAdapterError(
            "Asset catalog fallbackPolicy must be procedural, warn, or error."
        )
    for entity_id, entry in entities.items():
        if not isinstance(entry, dict):
            raise BlenderAdapterError(f"Asset catalog entry {entity_id!r} must be an object.")
        if not asset_entry_enabled(entry):
            continue
        asset_format = asset_entry_format(entry)
        if asset_format not in SUPPORTED_ASSET_FORMATS:
            raise BlenderAdapterError(
                f"Asset {entity_id!r} format must be one of "
                f"{sorted(SUPPORTED_ASSET_FORMATS)}."
            )
        if asset_format == "blend":
            collection = entry.get("collectionName")
            if not isinstance(collection, str) or not collection.strip():
                raise BlenderAdapterError(
                    f"Blender asset {entity_id!r} must specify collectionName."
                )
        asset_path = catalog_asset_path(
            resolved, asset_entry_file(entry), f"assets.{entity_id}.file"
        )
        if not asset_path.is_file():
            raise BlenderAdapterError(
                f"Custom asset for {entity_id!r} was not found: {asset_path}"
            )
        scale = entry.get("scale", 1.0)
        if not isinstance(scale, (int, float)) or isinstance(scale, bool) or scale <= 0:
            raise BlenderAdapterError(f"Asset {entity_id!r} scale must be positive.")

    hdri = environment.get("hdri")
    if isinstance(hdri, dict) and asset_entry_enabled(hdri):
        hdri_path = catalog_asset_path(resolved, hdri.get("file"), "environment.hdri.file")
        if not hdri_path.is_file():
            raise BlenderAdapterError(f"Environment HDRI was not found: {hdri_path}")
    ground = environment.get("groundMaterial")
    if isinstance(ground, dict) and asset_entry_enabled(ground):
        texture_keys = ("baseColor", "roughness", "normal", "height", "metallic")
        supplied = 0
        for key in texture_keys:
            value = ground.get(key)
            if value is None:
                continue
            supplied += 1
            texture_path = catalog_asset_path(
                resolved, value, f"environment.groundMaterial.{key}"
            )
            if not texture_path.is_file():
                raise BlenderAdapterError(
                    f"Ground material texture {key!r} was not found: {texture_path}"
                )
        if supplied == 0:
            raise BlenderAdapterError(
                "Enabled environment.groundMaterial must provide at least one texture map."
            )
    return catalog, resolved


def asset_catalog_fingerprint(
    catalog: Mapping[str, Any],
    catalog_path: Optional[Path],
) -> str:
    digest = hashlib.sha256(
        json.dumps(catalog, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    if catalog_path is None:
        return digest.hexdigest()
    referenced_paths: List[Path] = []
    entities = catalog.get("entities", {})
    if isinstance(entities, dict):
        for entry in entities.values():
            if isinstance(entry, dict) and asset_entry_enabled(entry):
                referenced_paths.append(
                    catalog_asset_path(catalog_path, asset_entry_file(entry), "asset file")
                )
    environment = catalog.get("environment", {})
    if isinstance(environment, dict):
        hdri = environment.get("hdri")
        if isinstance(hdri, dict) and asset_entry_enabled(hdri):
            referenced_paths.append(
                catalog_asset_path(catalog_path, hdri.get("file"), "environment.hdri.file")
            )
        ground = environment.get("groundMaterial")
        if isinstance(ground, dict) and asset_entry_enabled(ground):
            for key in ("baseColor", "roughness", "normal", "height", "metallic"):
                if ground.get(key) is not None:
                    referenced_paths.append(
                        catalog_asset_path(
                            catalog_path,
                            ground.get(key),
                            f"environment.groundMaterial.{key}",
                        )
                    )
    for asset_path in sorted(set(referenced_paths), key=lambda item: str(item)):
        stat = asset_path.stat()
        digest.update(str(asset_path).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def asset_source_records(catalog: Mapping[str, Any]) -> List[Dict[str, str]]:
    """Return enabled third-party source metadata for the execution record."""
    records: List[Dict[str, str]] = []

    def add(role: str, name: str, entry: Mapping[str, Any]) -> None:
        if not asset_entry_enabled(entry):
            return
        record = {
            "role": role,
            "name": name,
            "provider": str(entry.get("provider", "local")),
        }
        for source_key, output_key in (
            ("assetId", "assetId"),
            ("assetPage", "assetPage"),
            ("license", "license"),
        ):
            value = entry.get(source_key)
            if value is not None:
                record[output_key] = str(value)
        records.append(record)

    entities = catalog.get("entities", {})
    if isinstance(entities, dict):
        for entity_id, entry in sorted(entities.items()):
            if isinstance(entry, dict):
                add("entity", str(entity_id), entry)
    environment = catalog.get("environment", {})
    if isinstance(environment, dict):
        for name in ("hdri", "groundMaterial"):
            entry = environment.get(name)
            if isinstance(entry, dict):
                add("environment", name, entry)
    return records


def choose_scenes(
    storyboard: Mapping[str, Any],
    requested_scene: Optional[str],
) -> List[Mapping[str, Any]]:
    scenes = storyboard.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise BlenderAdapterError("Storyboard contains no scenes.")
    selected = [require_mapping(scene, "scene") for scene in scenes]
    if requested_scene:
        selected = [scene for scene in selected if scene.get("sceneId") == requested_scene]
        if not selected:
            available = ", ".join(str(scene.get("sceneId")) for scene in scenes)
            raise BlenderAdapterError(
                f"Unknown scene {requested_scene!r}. Available scenes: {available}"
            )
    return selected


def quality_for(
    storyboard: Mapping[str, Any],
    name: str,
    fps_override: Optional[int],
) -> Quality:
    project = require_mapping(storyboard.get("project"), "project")
    frame = require_mapping(project.get("frame"), "project.frame")
    width = int(frame.get("width", 1280))
    height = int(frame.get("height", 720))
    source_fps = int(frame.get("framesPerSecond", 24))
    if name == "draft":
        quality = Quality(
            name="draft",
            width=max(320, width // 2),
            height=max(180, height // 2),
            fps=12,
            samples=16,
            description="fast half-resolution Eevee animatic",
        )
    elif name == "preview":
        quality = Quality(
            name="preview",
            width=width,
            height=height,
            fps=source_fps,
            samples=64,
            description="full-resolution cinematic Eevee preview",
        )
    else:
        scale = max(1.0, 1920.0 / max(width, height))
        final_width = max(2, int(round(width * scale / 2.0)) * 2)
        final_height = max(2, int(round(height * scale / 2.0)) * 2)
        quality = Quality(
            name="final",
            width=final_width,
            height=final_height,
            fps=source_fps,
            samples=128,
            description="high-resolution cinematic Eevee render",
        )
    if fps_override is not None:
        if not 6 <= fps_override <= 60:
            raise BlenderAdapterError("--fps must be between 6 and 60.")
        quality = Quality(
            name=quality.name,
            width=quality.width,
            height=quality.height,
            fps=fps_override,
            samples=quality.samples,
            description=quality.description + f" at {fps_override} fps",
        )
    return quality


def normalize_verb(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def action_strategy(verb: str, aliases: Mapping[str, Any]) -> str:
    normalized = normalize_verb(verb)
    alias = aliases.get(normalized)
    if isinstance(alias, str) and alias.strip():
        normalized = normalize_verb(alias)
    words = set(normalized.split("_"))
    if words & SLIP_WORDS:
        return "slip-and-recover"
    if words & CLIMB_WORDS:
        return "climb"
    if words & LOWER_WORDS:
        return "lower"
    if words & RAISE_WORDS:
        return "raise"
    if words & TOUCH_WORDS:
        return "touch"
    if words & TWITCH_WORDS:
        return "twitch"
    if words & APPEAR_WORDS:
        return "appear"
    if words & DISAPPEAR_WORDS:
        return "disappear"
    if words & TURN_WORDS:
        return "turn-or-search"
    if words & MOVE_WORDS:
        return "move"
    return "generic-motion"


def default_output_dir(storyboard_path: Path) -> Path:
    return storyboard_path.resolve().parent / "outputs" / "blender"


def safe_slug(value: Any) -> str:
    text = value if isinstance(value, str) else "animind-story"
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return slug[:80].rstrip("-") or "animind-story"


def entity_catalog(storyboard: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    for value in storyboard.get("entities", []):
        if isinstance(value, dict) and isinstance(value.get("entityId"), str):
            result[value["entityId"]] = value
    return result


def synthetic_scene_entity(scene: Mapping[str, Any]) -> Dict[str, Any]:
    """Create a deterministic fallback when a planner supplied no entity catalog."""
    scene_id = require_text(scene.get("sceneId"), "scene.sceneId")
    visual = require_mapping(scene.get("visual"), f"{scene_id}.visual")
    narration = require_mapping(scene.get("narration"), f"{scene_id}.narration")
    description = (
        f"{visual.get('summary', '')} {narration.get('text', '')}"
    ).strip()
    lowered = description.lower()
    if any(word in lowered for word in ("insect", "lanternfly", "nymph", "animal", "bird", "dog", "cat", "fish", "creature")):
        entity_type = "creature"
    elif any(word in lowered for word in ("car", "truck", "bus", "train", "boat", "plane", "vehicle")):
        entity_type = "vehicle"
    elif re.search(r"\b(i|he|she|they|person|woman|man|child|girl|boy)\b", lowered):
        entity_type = "person"
    else:
        entity_type = "object"
    return {
        "entityId": f"{scene_id}_subject",
        "name": "Story Subject",
        "entityType": entity_type,
        "visualDescription": description or "The central subject of this story beat",
        "continuityNotes": "Procedural fallback created because the neutral plan supplied no entity catalog.",
    }


def scene_entity_ids(scene: Mapping[str, Any]) -> List[str]:
    visual = require_mapping(scene.get("visual"), "scene.visual")
    values = [str(value) for value in visual.get("entityIds", [])]
    for action in visual.get("actions", []):
        if not isinstance(action, dict):
            continue
        for key in ("actorId", "targetId"):
            reference = action.get(key)
            if isinstance(reference, str) and reference not in values:
                values.append(reference)
    return values or [str(synthetic_scene_entity(scene)["entityId"])]


def print_plan(
    storyboard: Mapping[str, Any],
    scenes: Sequence[Mapping[str, Any]],
    quality: Quality,
    asset_catalog: Mapping[str, Any],
    output_dir: Path,
    camera_style: str = "auto",
    motion_blur: bool = True,
    depth_of_field: bool = True,
    strict_assets: bool = False,
) -> None:
    project = require_mapping(storyboard.get("project"), "project")
    entities = entity_catalog(storyboard)
    overrides = require_mapping(asset_catalog.get("entities", {}), "asset catalog entities")
    aliases = require_mapping(asset_catalog.get("actionAliases", {}), "action aliases")
    seconds = sum(float(scene.get("durationSeconds", 0)) for scene in scenes)
    frames = sum(max(1, int(round(float(scene.get("durationSeconds", 0)) * quality.fps))) for scene in scenes)
    used_ids = {entity_id for scene in scenes for entity_id in scene_entity_ids(scene)}
    for scene in scenes:
        synthetic = synthetic_scene_entity(scene)
        if synthetic["entityId"] in used_ids and synthetic["entityId"] not in entities:
            entities[synthetic["entityId"]] = synthetic
    if strict_assets or asset_catalog.get("fallbackPolicy") == "error":
        missing = sorted(
            entity_id
            for entity_id in used_ids
            if not (
                isinstance(overrides.get(entity_id), dict)
                and asset_entry_enabled(overrides[entity_id])
            )
        )
        if missing:
            raise BlenderAdapterError(
                "Strict asset validation found no enabled production mapping for: "
                + ", ".join(missing)
            )

    print(f"Project: {project.get('title') or 'Untitled'}")
    print(
        f"Blender plan: {len(scenes)} scene(s), {seconds:g} seconds, "
        f"approximately {frames} frames"
    )
    print(
        f"Quality: {quality.name} - {quality.width}x{quality.height}, "
        f"{quality.fps} fps, {quality.samples} samples"
    )
    print(f"Output: {output_dir}")
    print("Assets:")
    for entity_id in sorted(used_ids):
        entity = entities.get(entity_id, {})
        entry = overrides.get(entity_id)
        if isinstance(entry, dict) and asset_entry_enabled(entry):
            source = asset_source_label(entry)
        elif isinstance(entry, dict):
            source = "procedural fallback (catalog entry disabled)"
        else:
            source = "procedural fallback"
        print(
            f"  {entity_id}: {entity.get('entityType', 'object')} - {source}"
        )
    environment = require_mapping(asset_catalog.get("environment", {}), "asset catalog environment")
    hdri = environment.get("hdri")
    ground = environment.get("groundMaterial")
    hdri_source = (
        asset_source_label(hdri)
        if isinstance(hdri, dict) and asset_entry_enabled(hdri)
        else "procedural world lighting"
    )
    ground_source = (
        str(ground.get("provider", "local PBR")) + " PBR"
        if isinstance(ground, dict) and asset_entry_enabled(ground)
        else "procedural material"
    )
    print(f"Environment: {hdri_source}; ground: {ground_source}")
    print(
        f"Camera: {camera_style}; depth of field: {'on' if depth_of_field else 'off'}; "
        f"motion blur: {'on' if motion_blur else 'off'}"
    )
    print("Actions:")
    for scene in scenes:
        visual = require_mapping(scene.get("visual"), f"{scene.get('sceneId')}.visual")
        actions = visual.get("actions", [])
        mappings = []
        for action in actions:
            if isinstance(action, dict):
                mappings.append(
                    f"{action.get('verb')} -> {action_strategy(str(action.get('verb', '')), aliases)}"
                )
        print(f"  {scene.get('sceneId')}: {', '.join(mappings)}")


def find_blender(requested: Optional[Path]) -> Path:
    if requested:
        resolved = requested.expanduser().resolve()
        if not resolved.is_file():
            raise BlenderAdapterError(f"Blender executable not found: {resolved}")
        return resolved
    on_path = shutil.which("blender")
    if on_path:
        return Path(on_path).resolve()
    candidates = (
        Path("/Applications/Blender.app/Contents/MacOS/Blender"),
        Path("/Applications/Blender 4.5.app/Contents/MacOS/Blender"),
        Path("/Applications/Blender 4.4.app/Contents/MacOS/Blender"),
        Path("/Applications/Blender 4.3.app/Contents/MacOS/Blender"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise BlenderAdapterError(
        "Blender was not found. Install it in Applications or provide "
        "--blender /path/to/Blender."
    )


def build_blender_command(
    blender: Path,
    args: argparse.Namespace,
    storyboard_path: Path,
    output_dir: Path,
) -> List[str]:
    command = [
        str(blender),
        "--background",
        "--factory-startup",
        "--python",
        str(Path(__file__).resolve()),
        "--",
        str(storyboard_path),
        "--inside-blender",
        "--quality",
        args.quality,
        "--output-dir",
        str(output_dir),
        "--narration",
        "off",
    ]
    if args.scene:
        command.extend(["--scene", args.scene])
    if args.asset_catalog:
        command.extend(["--asset-catalog", str(args.asset_catalog.expanduser().resolve())])
    if args.strict_assets:
        command.append("--strict-assets")
    command.extend(["--camera-style", args.camera_style])
    if args.no_motion_blur:
        command.append("--no-motion-blur")
    if args.no_depth_of_field:
        command.append("--no-depth-of-field")
    if args.fps is not None:
        command.extend(["--fps", str(args.fps)])
    if args.overwrite:
        command.append("--overwrite")
    if args.no_save_blend:
        command.append("--no-save-blend")
    command.append("--no-assemble")
    return command


def _load_bpy() -> Tuple[Any, Any, Any]:
    try:
        import bpy
        from mathutils import Vector
        from mathutils import Euler
    except ImportError as exc:
        raise BlenderAdapterError(
            "The internal render stage must run inside Blender, not ordinary Python."
        ) from exc
    return bpy, Vector, Euler


def clear_blender_scene(bpy: Any) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        if collection.users == 0:
            bpy.data.collections.remove(collection)
    for obj in list(bpy.data.objects):
        if obj.users == 0:
            bpy.data.objects.remove(obj)
    for block_group in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.materials,
    ):
        for block in list(block_group):
            if block.users == 0:
                block_group.remove(block)


def color_for(identifier: str, saturation: float = 0.48, value: float = 0.72) -> Tuple[float, float, float, float]:
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:2], "big") / 65535.0
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return red, green, blue, 1.0


def create_material(
    bpy: Any,
    name: str,
    color: Tuple[float, float, float, float],
    roughness: float = 0.65,
    metallic: float = 0.0,
) -> Any:
    material = bpy.data.materials.new(name=name)
    material.diffuse_color = color
    material.use_nodes = True
    node = material.node_tree.nodes.get("Principled BSDF")
    if node:
        if "Base Color" in node.inputs:
            node.inputs["Base Color"].default_value = color
        if "Roughness" in node.inputs:
            node.inputs["Roughness"].default_value = roughness
        if "Metallic" in node.inputs:
            node.inputs["Metallic"].default_value = metallic
    return material


def load_blender_image(bpy: Any, path: Path, non_color: bool = False) -> Any:
    try:
        image = bpy.data.images.load(str(path), check_existing=True)
    except Exception as exc:
        raise BlenderAdapterError(f"Blender could not load texture {path}: {exc}") from exc
    if non_color:
        try:
            image.colorspace_settings.name = "Non-Color"
        except (TypeError, ValueError):
            pass
    return image


def create_pbr_material(
    bpy: Any,
    name: str,
    specification: Mapping[str, Any],
    catalog_path: Path,
    fallback_color: Tuple[float, float, float, float],
) -> Any:
    """Build a Principled material from local Poly Haven-style texture maps."""
    material = bpy.data.materials.new(name=name)
    material.diffuse_color = fallback_color
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (700, 0)
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.location = (420, 0)
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    if "Base Color" in shader.inputs:
        shader.inputs["Base Color"].default_value = fallback_color
    if "Roughness" in shader.inputs:
        shader.inputs["Roughness"].default_value = 0.86

    coordinates = nodes.new("ShaderNodeTexCoord")
    coordinates.location = (-900, 0)
    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (-700, 0)
    texture_scale = float(specification.get("textureScale", 3.0))
    mapping.inputs["Scale"].default_value = (texture_scale,) * 3
    links.new(coordinates.outputs["Generated"], mapping.inputs["Vector"])

    positions = {
        "baseColor": (-430, 240),
        "roughness": (-430, 70),
        "metallic": (-430, -80),
        "normal": (-430, -250),
        "height": (-430, -430),
    }
    texture_nodes: Dict[str, Any] = {}
    for key, location in positions.items():
        value = specification.get(key)
        if value is None:
            continue
        texture_path = catalog_asset_path(
            catalog_path, value, f"environment.groundMaterial.{key}"
        )
        texture = nodes.new("ShaderNodeTexImage")
        texture.name = f"{name}_{key}"
        texture.label = key
        texture.location = location
        texture.extension = "REPEAT"
        texture.image = load_blender_image(bpy, texture_path, non_color=key != "baseColor")
        links.new(mapping.outputs["Vector"], texture.inputs["Vector"])
        texture_nodes[key] = texture

    if "baseColor" in texture_nodes and "Base Color" in shader.inputs:
        links.new(texture_nodes["baseColor"].outputs["Color"], shader.inputs["Base Color"])
    if "roughness" in texture_nodes and "Roughness" in shader.inputs:
        links.new(texture_nodes["roughness"].outputs["Color"], shader.inputs["Roughness"])
    if "metallic" in texture_nodes and "Metallic" in shader.inputs:
        links.new(texture_nodes["metallic"].outputs["Color"], shader.inputs["Metallic"])

    normal_output = None
    if "normal" in texture_nodes:
        normal_map = nodes.new("ShaderNodeNormalMap")
        normal_map.location = (160, -240)
        normal_map.inputs["Strength"].default_value = float(
            specification.get("normalStrength", 0.65)
        )
        links.new(texture_nodes["normal"].outputs["Color"], normal_map.inputs["Color"])
        normal_output = normal_map.outputs["Normal"]
    if "height" in texture_nodes:
        bump = nodes.new("ShaderNodeBump")
        bump.location = (180, -420)
        bump.inputs["Strength"].default_value = float(
            specification.get("heightStrength", 0.28)
        )
        bump.inputs["Distance"].default_value = float(
            specification.get("heightDistance", 0.08)
        )
        links.new(texture_nodes["height"].outputs["Color"], bump.inputs["Height"])
        if normal_output is not None:
            links.new(normal_output, bump.inputs["Normal"])
        normal_output = bump.outputs["Normal"]
    if normal_output is not None and "Normal" in shader.inputs:
        links.new(normal_output, shader.inputs["Normal"])
    return material


def configure_hdri_world(
    bpy: Any,
    specification: Mapping[str, Any],
    catalog_path: Path,
) -> bool:
    if not asset_entry_enabled(specification):
        return False
    hdri_path = catalog_asset_path(catalog_path, specification.get("file"), "environment.hdri.file")
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputWorld")
    output.location = (520, 0)
    background = nodes.new("ShaderNodeBackground")
    background.location = (280, 0)
    background.inputs["Strength"].default_value = float(specification.get("strength", 0.65))
    environment = nodes.new("ShaderNodeTexEnvironment")
    environment.location = (-100, 0)
    environment.image = load_blender_image(bpy, hdri_path)
    coordinates = nodes.new("ShaderNodeTexCoord")
    coordinates.location = (-650, 0)
    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (-420, 0)
    rotation = math.radians(float(specification.get("rotationDegrees", 0.0)))
    mapping.inputs["Rotation"].default_value[2] = rotation
    links.new(coordinates.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], environment.inputs["Vector"])
    links.new(environment.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])
    return True


def assign_material(obj: Any, material: Any) -> None:
    if hasattr(obj.data, "materials"):
        obj.data.materials.append(material)


def add_empty(bpy: Any, name: str) -> Any:
    root = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(root)
    root.empty_display_type = "PLAIN_AXES"
    root.empty_display_size = 0.35
    return root


def parent_local(obj: Any, root: Any) -> None:
    obj.parent = root


def add_uv_sphere(
    bpy: Any,
    name: str,
    location: Tuple[float, float, float],
    scale: Tuple[float, float, float],
    material: Any,
    root: Any,
) -> Any:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    assign_material(obj, material)
    parent_local(obj, root)
    return obj


def add_ico_sphere(
    bpy: Any,
    name: str,
    location: Tuple[float, float, float],
    scale: Tuple[float, float, float],
    material: Any,
    root: Any,
) -> Any:
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    assign_material(obj, material)
    parent_local(obj, root)
    return obj


def add_cube(
    bpy: Any,
    name: str,
    location: Tuple[float, float, float],
    scale: Tuple[float, float, float],
    material: Any,
    root: Any,
    bevel: float = 0.05,
) -> Any:
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    assign_material(obj, material)
    parent_local(obj, root)
    if bevel > 0:
        modifier = obj.modifiers.new(name="Soft edges", type="BEVEL")
        modifier.width = bevel
        modifier.segments = 2
    return obj


def add_cylinder_between(
    bpy: Any,
    Vector: Any,
    name: str,
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    radius: float,
    material: Any,
    root: Any,
) -> Any:
    start_vector = Vector(start)
    end_vector = Vector(end)
    direction = end_vector - start_vector
    midpoint = (start_vector + end_vector) / 2.0
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=16,
        radius=radius,
        depth=max(0.001, direction.length),
        location=midpoint,
    )
    obj = bpy.context.object
    obj.name = name
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = direction.to_track_quat("Z", "Y")
    assign_material(obj, material)
    parent_local(obj, root)
    return obj


def create_insect(bpy: Any, Vector: Any, entity: Mapping[str, Any]) -> Any:
    entity_id = str(entity["entityId"])
    root = add_empty(bpy, entity_id)
    shell = create_material(bpy, f"{entity_id}_shell", (0.16, 0.10, 0.07, 1.0), 0.82)
    cream = create_material(bpy, f"{entity_id}_markings", (0.72, 0.65, 0.48, 1.0), 0.75)
    add_uv_sphere(bpy, f"{entity_id}_body", (0, 0, 0.28), (0.72, 0.42, 0.30), shell, root)
    add_uv_sphere(bpy, f"{entity_id}_head", (0.68, 0, 0.31), (0.30, 0.31, 0.27), shell, root)
    for index, x_value in enumerate((-0.40, 0.0, 0.38), start=1):
        for side in (-1, 1):
            start = (x_value, side * 0.22, 0.25)
            knee = (x_value + 0.08, side * 0.58, 0.12)
            foot = (x_value + 0.20, side * 0.82, 0.04)
            add_cylinder_between(bpy, Vector, f"{entity_id}_leg_{index}_{side}_a", start, knee, 0.035, shell, root)
            add_cylinder_between(bpy, Vector, f"{entity_id}_leg_{index}_{side}_b", knee, foot, 0.028, shell, root)
    for side in (-1, 1):
        add_cylinder_between(
            bpy,
            Vector,
            f"{entity_id}_antenna_{side}",
            (0.86, side * 0.13, 0.42),
            (1.35, side * 0.36, 0.60),
            0.018,
            shell,
            root,
        )
    for index, location in enumerate(((-0.28, -0.32, 0.49), (0.14, 0.32, 0.51), (0.42, -0.23, 0.48))):
        add_uv_sphere(bpy, f"{entity_id}_spot_{index}", location, (0.06, 0.035, 0.025), cream, root)
    root.scale = (0.48, 0.48, 0.48)
    return root


def create_generic_creature(bpy: Any, Vector: Any, entity: Mapping[str, Any]) -> Any:
    entity_id = str(entity["entityId"])
    root = add_empty(bpy, entity_id)
    fur = create_material(bpy, f"{entity_id}_body_material", color_for(entity_id), 0.78)
    add_uv_sphere(bpy, f"{entity_id}_body", (0, 0, 0.8), (0.9, 0.48, 0.58), fur, root)
    add_uv_sphere(bpy, f"{entity_id}_head", (0.85, 0, 1.12), (0.42, 0.38, 0.40), fur, root)
    for x_value in (-0.48, 0.48):
        for side in (-1, 1):
            add_cylinder_between(
                bpy,
                Vector,
                f"{entity_id}_leg_{x_value}_{side}",
                (x_value, side * 0.24, 0.58),
                (x_value, side * 0.28, 0.05),
                0.10,
                fur,
                root,
            )
    return root


def create_person(bpy: Any, Vector: Any, entity: Mapping[str, Any]) -> Any:
    entity_id = str(entity["entityId"])
    description = f"{entity.get('name', '')} {entity.get('visualDescription', '')}".lower()
    root = add_empty(bpy, entity_id)
    skin = create_material(bpy, f"{entity_id}_skin", (0.42, 0.23, 0.13, 1.0), 0.72)
    cloth_color = (0.11, 0.34, 0.36, 1.0) if "teal" in description else color_for(entity_id)
    cloth = create_material(bpy, f"{entity_id}_cloth", cloth_color, 0.86)
    if "only" in description and ("hand" in description or "forearm" in description):
        add_cylinder_between(
            bpy,
            Vector,
            f"{entity_id}_sleeve",
            (-0.95, 0, 0.62),
            (0.05, 0, 0.62),
            0.20,
            cloth,
            root,
        )
        add_cylinder_between(
            bpy,
            Vector,
            f"{entity_id}_forearm",
            (0.02, 0, 0.62),
            (0.62, 0, 0.56),
            0.14,
            skin,
            root,
        )
        add_uv_sphere(bpy, f"{entity_id}_hand", (0.78, 0, 0.54), (0.30, 0.19, 0.12), skin, root)
        return root

    add_cube(bpy, f"{entity_id}_torso", (0, 0, 1.45), (0.38, 0.24, 0.62), cloth, root, 0.12)
    add_uv_sphere(bpy, f"{entity_id}_head", (0, 0, 2.30), (0.31, 0.29, 0.36), skin, root)
    for side in (-1, 1):
        add_cylinder_between(
            bpy,
            Vector,
            f"{entity_id}_arm_{side}",
            (0, side * 0.42, 1.74),
            (0.05, side * 0.58, 0.92),
            0.11,
            cloth,
            root,
        )
        add_cylinder_between(
            bpy,
            Vector,
            f"{entity_id}_leg_{side}",
            (0, side * 0.20, 0.90),
            (0, side * 0.22, 0.08),
            0.14,
            cloth,
            root,
        )
    return root


def create_prop(bpy: Any, Vector: Any, entity: Mapping[str, Any]) -> Any:
    entity_id = str(entity["entityId"])
    description = f"{entity.get('name', '')} {entity.get('visualDescription', '')}".lower()
    root = add_empty(bpy, entity_id)
    base = create_material(bpy, f"{entity_id}_material", color_for(entity_id, 0.35, 0.62), 0.66)
    dark = create_material(bpy, f"{entity_id}_dark", (0.035, 0.045, 0.055, 1.0), 0.38, 0.18)
    glow = create_material(bpy, f"{entity_id}_screen", (0.26, 0.58, 0.72, 1.0), 0.28)
    if "laptop" in description:
        add_cube(bpy, f"{entity_id}_base", (0, 0, 0.08), (0.78, 0.54, 0.06), dark, root, 0.03)
        add_cube(bpy, f"{entity_id}_screen_case", (0, 0.50, 0.57), (0.78, 0.055, 0.50), dark, root, 0.04)
        add_cube(bpy, f"{entity_id}_screen", (0, 0.435, 0.58), (0.68, 0.012, 0.40), glow, root, 0.01)
    elif "nozzle" in description or "vacuum" in description:
        add_cylinder_between(
            bpy,
            Vector,
            f"{entity_id}_tube",
            (-0.95, 0, 0.75),
            (0.38, 0, 0.30),
            0.19,
            dark,
            root,
        )
        add_cube(bpy, f"{entity_id}_mouth", (0.55, 0, 0.22), (0.28, 0.36, 0.12), dark, root, 0.06)
    elif "pebble" in description or "rock" in description or "stone" in description:
        add_ico_sphere(bpy, f"{entity_id}_stone", (0, 0, 0.23), (0.55, 0.42, 0.28), base, root)
        root.rotation_euler[2] = 0.35
    elif "seam" in description or "ridge" in description:
        add_cube(bpy, f"{entity_id}_ridge", (0, 0, 0.07), (1.50, 0.17, 0.08), base, root, 0.04)
    else:
        add_cube(bpy, f"{entity_id}_prop", (0, 0, 0.45), (0.48, 0.48, 0.45), base, root, 0.10)
    return root


def create_vehicle(bpy: Any, Vector: Any, entity: Mapping[str, Any]) -> Any:
    entity_id = str(entity["entityId"])
    root = add_empty(bpy, entity_id)
    body = create_material(bpy, f"{entity_id}_body", color_for(entity_id, 0.65, 0.78), 0.42, 0.25)
    tire = create_material(bpy, f"{entity_id}_tire", (0.025, 0.025, 0.025, 1.0), 0.88)
    add_cube(bpy, f"{entity_id}_chassis", (0, 0, 0.62), (1.30, 0.66, 0.34), body, root, 0.15)
    add_cube(bpy, f"{entity_id}_cabin", (0.10, 0, 1.12), (0.68, 0.58, 0.36), body, root, 0.14)
    for x_value in (-0.76, 0.76):
        for side in (-1, 1):
            bpy.ops.mesh.primitive_cylinder_add(vertices=20, radius=0.30, depth=0.18, location=(x_value, side * 0.70, 0.32), rotation=(math.pi / 2, 0, 0))
            wheel = bpy.context.object
            wheel.name = f"{entity_id}_wheel_{x_value}_{side}"
            assign_material(wheel, tire)
            parent_local(wheel, root)
    return root


def create_environment_feature(bpy: Any, Vector: Any, entity: Mapping[str, Any]) -> Any:
    return create_prop(bpy, Vector, entity)


def load_custom_collection(
    bpy: Any,
    entity_id: str,
    entry: Mapping[str, Any],
    catalog_path: Path,
) -> Any:
    blend_path = catalog_asset_path(
        catalog_path, asset_entry_file(entry), f"assets.{entity_id}.file"
    )
    collection_name = str(entry["collectionName"])
    with bpy.data.libraries.load(str(blend_path), link=False) as (available, requested):
        if collection_name not in available.collections:
            raise BlenderAdapterError(
                f"Collection {collection_name!r} was not found in {blend_path}."
            )
        requested.collections = [collection_name]
    loaded = requested.collections[0]
    root = add_empty(bpy, entity_id)
    root.instance_type = "COLLECTION"
    root.instance_collection = loaded
    apply_asset_transform(root, entry)
    annotate_asset_root(root, entry)
    configure_imported_animation(root, entry)
    return root


def apply_asset_transform(root: Any, entry: Mapping[str, Any]) -> None:
    scale = float(entry.get("scale", 1.0))
    root.scale = (scale,) * 3
    rotation = entry.get("rotationDegrees", [0, 0, 0])
    if isinstance(rotation, list) and len(rotation) == 3:
        root.rotation_euler = tuple(math.radians(float(item)) for item in rotation)
    offset = entry.get("locationOffset", [0, 0, 0])
    if isinstance(offset, list) and len(offset) == 3:
        root["animind_location_offset"] = [float(item) for item in offset]


def annotate_asset_root(root: Any, entry: Mapping[str, Any]) -> None:
    root["animind_asset_provider"] = str(entry.get("provider", "local"))
    root["animind_asset_format"] = asset_entry_format(entry)
    if entry.get("assetId") is not None:
        root["animind_asset_id"] = str(entry.get("assetId"))
    if entry.get("license") is not None:
        root["animind_asset_license"] = str(entry.get("license"))


def configure_imported_animation(root: Any, entry: Mapping[str, Any]) -> None:
    animation = entry.get("animation", {})
    if not isinstance(animation, dict):
        return
    loop = animation.get("loop", False) is True
    objects = [root, *_descendants(root)]
    instance_collection = getattr(root, "instance_collection", None)
    if instance_collection is not None:
        objects.extend(list(getattr(instance_collection, "all_objects", [])))
    for obj in objects:
        animation_data = getattr(obj, "animation_data", None)
        action = getattr(animation_data, "action", None)
        if action is None:
            continue
        if loop:
            for curve in getattr(action, "fcurves", []):
                if not any(modifier.type == "CYCLES" for modifier in curve.modifiers):
                    curve.modifiers.new(type="CYCLES")
        obj["animind_imported_animation"] = True


def import_external_asset(
    bpy: Any,
    entity_id: str,
    entry: Mapping[str, Any],
    catalog_path: Path,
) -> Any:
    asset_path = catalog_asset_path(
        catalog_path, asset_entry_file(entry), f"assets.{entity_id}.file"
    )
    asset_format = asset_entry_format(entry)
    before = set(bpy.data.objects)
    try:
        if asset_format in {"gltf", "glb"}:
            bpy.ops.import_scene.gltf(filepath=str(asset_path))
        elif asset_format == "fbx":
            imported = False
            try:
                bpy.ops.wm.fbx_import(filepath=str(asset_path))
                imported = True
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
            if not imported:
                bpy.ops.import_scene.fbx(filepath=str(asset_path))
        elif asset_format == "obj":
            imported = False
            try:
                bpy.ops.wm.obj_import(filepath=str(asset_path))
                imported = True
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
            if not imported:
                bpy.ops.import_scene.obj(filepath=str(asset_path))
        else:
            raise BlenderAdapterError(
                f"Unsupported external asset format {asset_format!r} for {entity_id!r}."
            )
    except Exception as exc:
        raise BlenderAdapterError(
            f"Blender could not import {asset_path.name} for {entity_id!r}: {exc}"
        ) from exc

    imported_objects = [obj for obj in bpy.data.objects if obj not in before]
    keep_scene_objects = entry.get("keepImportedLightsAndCameras", False) is True
    if not keep_scene_objects:
        for obj in list(imported_objects):
            if getattr(obj, "type", None) in {"LIGHT", "CAMERA"}:
                bpy.data.objects.remove(obj, do_unlink=True)
                imported_objects.remove(obj)
    if not imported_objects:
        raise BlenderAdapterError(
            f"Importing {asset_path.name} created no usable objects for {entity_id!r}."
        )

    root = add_empty(bpy, entity_id)
    imported_set = set(imported_objects)
    for obj in imported_objects:
        if obj.parent not in imported_set:
            world_matrix = obj.matrix_world.copy()
            obj.parent = root
            obj.matrix_world = world_matrix
    apply_asset_transform(root, entry)
    annotate_asset_root(root, entry)
    configure_imported_animation(root, entry)
    return root


def load_catalog_asset(
    bpy: Any,
    entity_id: str,
    entry: Mapping[str, Any],
    catalog_path: Path,
) -> Any:
    if asset_entry_format(entry) == "blend":
        return load_custom_collection(bpy, entity_id, entry, catalog_path)
    return import_external_asset(bpy, entity_id, entry, catalog_path)


def create_entity_object(
    bpy: Any,
    Vector: Any,
    entity: Mapping[str, Any],
    catalog: Mapping[str, Any],
    catalog_path: Optional[Path],
    strict_assets: bool = False,
) -> Any:
    entity_id = str(entity["entityId"])
    entries = require_mapping(catalog.get("entities", {}), "asset catalog entities")
    entry = entries.get(entity_id)
    if (
        isinstance(entry, dict)
        and asset_entry_enabled(entry)
        and catalog_path is not None
    ):
        return load_catalog_asset(bpy, entity_id, entry, catalog_path)
    fallback_policy = str(catalog.get("fallbackPolicy", "warn"))
    if strict_assets or fallback_policy == "error":
        raise BlenderAdapterError(
            f"No enabled production asset is mapped for {entity_id!r}. "
            "Add it to --asset-catalog or disable --strict-assets."
        )
    if fallback_policy == "warn":
        print(f"Warning: using procedural fallback for {entity_id}")
    entity_type = str(entity.get("entityType", "object")).lower()
    description = f"{entity.get('name', '')} {entity.get('visualDescription', '')}".lower()
    if entity_type in {"person", "human", "character"}:
        return create_person(bpy, Vector, entity)
    if entity_type in {"animal", "creature", "insect", "bird", "fish"}:
        if any(word in description for word in ("insect", "lanternfly", "nymph", "six legs", "antenna")):
            return create_insect(bpy, Vector, entity)
        return create_generic_creature(bpy, Vector, entity)
    if entity_type in {"vehicle", "car", "truck", "boat", "aircraft"}:
        return create_vehicle(bpy, Vector, entity)
    if "environment" in entity_type:
        return create_environment_feature(bpy, Vector, entity)
    return create_prop(bpy, Vector, entity)


def scene_positions(
    scene: Mapping[str, Any],
    entities: Mapping[str, Mapping[str, Any]],
    entity_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Tuple[float, float, float]]:
    visual = require_mapping(scene.get("visual"), "scene.visual")
    ids = list(entity_ids) if entity_ids is not None else scene_entity_ids(scene)
    actions = [value for value in visual.get("actions", []) if isinstance(value, dict)]
    priority: List[str] = list(ids)
    for action in actions:
        for key in ("actorId", "targetId"):
            value = action.get(key)
            if isinstance(value, str) and value in entities and value not in priority:
                priority.append(value)
    positions: Dict[str, Tuple[float, float, float]] = {}
    slots = (
        (-1.25, 0.0, 0.0),
        (0.65, 0.0, 0.0),
        (1.75, -0.75, 0.0),
        (1.50, 1.05, 0.0),
        (-0.30, 1.20, 0.0),
        (-1.60, -1.10, 0.0),
    )
    for index, entity_id in enumerate(priority):
        positions[entity_id] = slots[index] if index < len(slots) else (
            -1.5 + (index % 4),
            1.5 + (index // 4) * 0.8,
            0.0,
        )

    # Manipulated props look clearer when placed between the actor and subject.
    for action in actions:
        strategy = action_strategy(str(action.get("verb", "")), {})
        target_id = action.get("targetId")
        if strategy in {"lower", "raise"} and isinstance(target_id, str) and target_id in positions:
            positions[target_id] = (0.45, 0.0, 1.25)
    for entity_id, entity in entities.items():
        if entity_id not in positions:
            continue
        description = f"{entity.get('name', '')} {entity.get('visualDescription', '')}".lower()
        if "laptop" in description:
            positions[entity_id] = (0.45, 0.0, 0.78)
        elif "pebble" in description:
            positions[entity_id] = (0.55, 0.0, 0.0)
        elif "seam" in description:
            positions[entity_id] = (0.30, 0.0, 0.0)
    if any("laptop" in f"{entities.get(item, {}).get('name', '')}".lower() for item in ids):
        for entity_id in ids:
            entity = entities.get(entity_id, {})
            if str(entity.get("entityType", "")).lower() in {"person", "human", "character"}:
                x_value, y_value, _z_value = positions[entity_id]
                positions[entity_id] = (x_value, y_value, 0.15)
    return positions


def create_location(
    bpy: Any,
    Vector: Any,
    storyboard: Mapping[str, Any],
    scene: Mapping[str, Any],
    catalog: Mapping[str, Any],
    catalog_path: Optional[Path],
) -> None:
    visual = require_mapping(scene.get("visual"), "scene.visual")
    description = str(visual.get("settingDescription", "")).lower()
    location_id = visual.get("locationId")
    for location in storyboard.get("locations", []):
        if isinstance(location, dict) and location.get("locationId") == location_id:
            description += " " + str(location.get("visualDescription", "")).lower()
            break
    root = add_empty(bpy, "environment")
    if any(word in description for word in ("cement", "concrete", "terrace", "urban")):
        ground_color = (0.36, 0.38, 0.39, 1.0)
    elif any(word in description for word in ("grass", "forest", "garden", "moss")):
        ground_color = (0.18, 0.32, 0.16, 1.0)
    elif any(word in description for word in ("sand", "desert", "beach")):
        ground_color = (0.62, 0.48, 0.30, 1.0)
    else:
        ground_color = (0.24, 0.27, 0.31, 1.0)
    environment = require_mapping(catalog.get("environment", {}), "asset catalog environment")
    ground_specification = environment.get("groundMaterial")
    if (
        isinstance(ground_specification, dict)
        and asset_entry_enabled(ground_specification)
        and catalog_path is not None
    ):
        ground = create_pbr_material(
            bpy,
            "ground_material",
            ground_specification,
            catalog_path,
            ground_color,
        )
    else:
        ground = create_material(bpy, "ground_material", ground_color, 0.92)
    ground_object = add_cube(
        bpy, "ground", (0, 0, -0.14), (8.0, 8.0, 0.12), ground, root, 0.0
    )
    ground_object["animind_ground_surface_z"] = GROUND_SURFACE_Z

    if "terrace" in description or "planter" in description:
        planter = create_material(bpy, "planter_material", (0.36, 0.22, 0.13, 1.0), 0.78)
        foliage = create_material(bpy, "foliage_material", (0.12, 0.30, 0.10, 1.0), 0.85)
        for index, y_value in enumerate((-2.6, 0.0, 2.6)):
            bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.62, depth=0.80, location=(4.4, y_value, 0.28))
            pot = bpy.context.object
            pot.name = f"planter_{index}"
            assign_material(pot, planter)
            parent_local(pot, root)
            add_ico_sphere(bpy, f"plant_{index}", (4.4, y_value, 1.05), (0.85, 0.78, 0.72), foliage, root)
    if any(word in description for word in ("inside", "interior", "table", "door")):
        wood = create_material(bpy, "table_material", (0.31, 0.18, 0.09, 1.0), 0.74)
        add_cube(bpy, "table_top", (0.5, 0.0, 0.68), (2.0, 1.4, 0.08), wood, root, 0.04)
        for x_value in (-1.20, 2.20):
            for y_value in (-1.00, 1.00):
                add_cube(bpy, f"table_leg_{x_value}_{y_value}", (x_value, y_value, 0.30), (0.08, 0.08, 0.36), wood, root, 0.02)
        wall = create_material(bpy, "wall_material", (0.58, 0.59, 0.58, 1.0), 0.95)
        add_cube(bpy, "back_wall", (3.8, 0, 2.4), (0.10, 6.0, 2.5), wall, root, 0.0)


def configure_world_and_lights(
    bpy: Any,
    scene: Mapping[str, Any],
    focus: Tuple[float, float, float],
    catalog: Mapping[str, Any],
    catalog_path: Optional[Path],
) -> None:
    visual = require_mapping(scene.get("visual"), "scene.visual")
    look = require_mapping(visual.get("look"), "scene.visual.look")
    text = f"{look.get('lighting', '')} {look.get('mood', '')}".lower()
    if any(word in text for word in ("warm", "sun", "hope", "morning")):
        key_color = (1.0, 0.73, 0.48)
        world_color = (0.055, 0.045, 0.035, 1.0)
    elif any(word in text for word in ("cool", "tense", "dark", "shadow", "night")):
        key_color = (0.48, 0.62, 1.0)
        world_color = (0.018, 0.026, 0.050, 1.0)
    else:
        key_color = (0.92, 0.95, 1.0)
        world_color = (0.035, 0.045, 0.060, 1.0)
    environment = require_mapping(catalog.get("environment", {}), "asset catalog environment")
    hdri_specification = environment.get("hdri")
    hdri_enabled = (
        isinstance(hdri_specification, dict)
        and asset_entry_enabled(hdri_specification)
        and catalog_path is not None
        and configure_hdri_world(bpy, hdri_specification, catalog_path)
    )
    if not hdri_enabled:
        world = bpy.context.scene.world
        world.use_nodes = True
        background = world.node_tree.nodes.get("Background")
        if background:
            background.inputs["Color"].default_value = world_color
            background.inputs["Strength"].default_value = 0.38

    energy_scale = 0.48 if hdri_enabled else 1.0

    bpy.ops.object.light_add(type="AREA", location=(focus[0] - 3.0, focus[1] - 4.0, focus[2] + 6.0))
    key = bpy.context.object
    key.name = "Key light"
    key.data.energy = 950 * energy_scale
    key.data.shape = "DISK"
    key.data.size = 5.0
    key.data.color = key_color

    bpy.ops.object.light_add(type="AREA", location=(focus[0] + 4.0, focus[1] + 2.5, focus[2] + 3.0))
    fill = bpy.context.object
    fill.name = "Fill light"
    fill.data.energy = 420 * energy_scale
    fill.data.size = 4.0
    fill.data.color = (0.48, 0.64, 1.0)

    bpy.ops.object.light_add(type="AREA", location=(focus[0] + 1.0, focus[1] + 4.0, focus[2] + 5.0))
    rim = bpy.context.object
    rim.name = "Rim light"
    rim.data.energy = 650 * energy_scale
    rim.data.size = 3.0
    rim.data.color = (1.0, 0.58, 0.34)


def camera_profile(camera_plan: Mapping[str, Any], override: str = "auto") -> CameraProfile:
    description = " ".join(
        str(camera_plan.get(key, ""))
        for key in ("shotType", "angle", "movement", "description")
    ).lower()
    if override != "auto":
        style = override
    elif any(word in description for word in ("overhead", "top-down", "bird's-eye", "birds-eye")):
        style = "overhead"
    elif "macro" in description or "extreme close" in description:
        style = "macro"
    elif "close" in description:
        style = "close"
    elif "medium" in description:
        style = "medium"
    elif "handheld" in description:
        style = "handheld"
    else:
        style = "wide"
    profiles = {
        "macro": CameraProfile("macro", 105.0, 2.45, 3.2, 0.20, 0.01, 0.12),
        "close": CameraProfile("close", 70.0, 4.20, 3.5, 0.55, 0.02, 0.30),
        "medium": CameraProfile("medium", 50.0, 6.70, 4.5, 0.90, 0.04, 0.46),
        "wide": CameraProfile("wide", 32.0, 10.20, 5.6, 1.00, 0.08, 0.64),
        "overhead": CameraProfile("overhead", 45.0, 7.20, 5.0, 0.55, 0.04, 1.00),
        "handheld": CameraProfile("handheld", 55.0, 4.80, 3.8, 0.75, 0.02, 0.36),
    }
    return profiles[style]


def add_handheld_camera_motion(camera_rig: Any) -> None:
    animation_data = getattr(camera_rig, "animation_data", None)
    action = getattr(animation_data, "action", None)
    if action is None:
        return
    curves = getattr(action, "fcurves", None)
    if curves is None:
        return
    for index in range(3):
        curve = curves.find("location", index=index)
        if curve is None:
            continue
        modifier = curve.modifiers.new(type="NOISE")
        modifier.scale = 11.0
        modifier.strength = 0.022 if index < 2 else 0.012
        modifier.phase = float(index * 17)


def create_camera(
    bpy: Any,
    Vector: Any,
    scene_value: Mapping[str, Any],
    roots: Mapping[str, Any],
    fps: int,
    camera_style: str = "auto",
    use_depth_of_field: bool = True,
) -> Any:
    visual = require_mapping(scene_value.get("visual"), "scene.visual")
    camera_plan = require_mapping(visual.get("camera"), "scene.visual.camera")
    action_values = [item for item in visual.get("actions", []) if isinstance(item, dict)]
    declared_ids = [str(value) for value in visual.get("entityIds", []) if str(value) in roots]
    subject_id = declared_ids[0] if declared_ids else next(
        (
            item.get("actorId")
            for item in action_values
            if isinstance(item.get("actorId"), str) and item.get("actorId") in roots
        ),
        None,
    )
    if subject_id is None:
        subject_id = next(iter(roots), None)
    subject = roots.get(subject_id) if subject_id else None
    base = subject.location.copy() if subject is not None else Vector((0, 0, 0))
    profile = camera_profile(camera_plan, camera_style)
    focus = add_empty(bpy, "camera_focus")
    if subject is not None:
        focus.parent = subject
        if hasattr(focus, "inherit_scale"):
            focus.inherit_scale = "NONE"
        focus.location = (0, 0, profile.focus_height)
    else:
        focus.location = (base.x, base.y, profile.focus_height)

    angle = str(camera_plan.get("angle", "eye-level")).lower()
    height_factor = profile.height_factor
    if "low" in angle or "ground" in angle:
        height_factor = min(height_factor, 0.16)
    elif "high" in angle:
        height_factor = max(height_factor, 0.72)
    if profile.name == "overhead":
        local_camera_position = Vector((0.0, -profile.distance * 0.12, profile.distance))
    else:
        local_camera_position = Vector(
            (-profile.distance * 0.72, -profile.distance * 0.78, profile.distance * height_factor)
        )

    camera_rig = add_empty(bpy, "camera_rig")
    camera_rig.location = base
    bpy.ops.object.camera_add(location=(0, 0, 0))
    camera = bpy.context.object
    camera.name = "Story camera"
    camera.parent = camera_rig
    camera.location = local_camera_position
    bpy.context.scene.camera = camera
    camera.data.lens = profile.lens_mm
    camera.data.sensor_width = 36.0
    camera.data.clip_start = profile.clip_start
    camera.data.clip_end = 1000.0
    camera.data.dof.use_dof = use_depth_of_field
    camera.data.dof.focus_object = focus
    camera.data.dof.aperture_fstop = profile.aperture_fstop
    if hasattr(camera.data.dof, "aperture_blades"):
        camera.data.dof.aperture_blades = 7
    constraint = camera.constraints.new(type="TRACK_TO")
    constraint.target = focus
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"

    start_frame = 1
    end_frame = max(2, int(round(float(scene_value.get("durationSeconds", 1)) * fps)))
    movement = str(camera_plan.get("movement", "static")).lower()
    camera_rig.keyframe_insert(data_path="location", frame=start_frame)
    camera_rig.keyframe_insert(data_path="rotation_euler", frame=start_frame)
    camera.keyframe_insert(data_path="location", frame=start_frame)
    initial = camera.location.copy()
    if "push" in movement:
        camera.location = initial * 0.76
    elif "track" in movement:
        camera_rig.location.x += 1.6
    elif "orbit" in movement:
        camera_rig.rotation_euler[2] = math.radians(-14)
        camera_rig.keyframe_insert(data_path="rotation_euler", frame=start_frame)
        camera_rig.rotation_euler[2] = math.radians(14)
    elif "lateral" in movement or "dolly" in movement:
        camera_rig.location.x += 1.8
    elif "lower" in movement:
        camera.location.z = max(profile.clip_start * 8.0, camera.location.z - 1.2)
    camera_rig.keyframe_insert(data_path="location", frame=end_frame)
    camera_rig.keyframe_insert(data_path="rotation_euler", frame=end_frame)
    camera.keyframe_insert(data_path="location", frame=end_frame)
    set_smooth_interpolation((camera_rig, camera))
    if profile.name == "handheld" or "handheld" in movement:
        add_handheld_camera_motion(camera_rig)
    camera["animind_camera_profile"] = profile.name
    return camera


def keyframe_transform(obj: Any, frame: int) -> None:
    obj.keyframe_insert(data_path="location", frame=frame)
    obj.keyframe_insert(data_path="rotation_euler", frame=frame)
    obj.keyframe_insert(data_path="scale", frame=frame)


def _descendants(root: Any) -> List[Any]:
    result: List[Any] = []
    pending = list(getattr(root, "children", []))
    while pending:
        child = pending.pop()
        result.append(child)
        pending.extend(list(getattr(child, "children", [])))
    return result


def object_world_min_z(bpy: Any, Vector: Any, root: Any) -> float:
    """Return the lowest rendered point belonging to a procedural/custom root."""
    bpy.context.view_layer.update()
    values: List[float] = []
    for obj in [root, *_descendants(root)]:
        if getattr(obj, "type", None) not in {"MESH", "CURVE", "SURFACE", "FONT", "META"}:
            continue
        for corner in getattr(obj, "bound_box", []):
            values.append(float((obj.matrix_world @ Vector(corner)).z))

    instance_collection = getattr(root, "instance_collection", None)
    if instance_collection is not None:
        for obj in getattr(instance_collection, "all_objects", []):
            if getattr(obj, "type", None) not in {"MESH", "CURVE", "SURFACE", "FONT", "META"}:
                continue
            matrix = root.matrix_world @ obj.matrix_world
            for corner in getattr(obj, "bound_box", []):
                values.append(float((matrix @ Vector(corner)).z))
    return min(values) if values else float(root.location.z)


def register_ground_clearance(bpy: Any, Vector: Any, root: Any) -> None:
    """Lift intersecting geometry and add a non-penetrating world-Z constraint."""
    lowest = object_world_min_z(bpy, Vector, root)
    relative_lowest = lowest - float(root.location.z)
    minimum_root_z = GROUND_SURFACE_Z + GROUND_CLEARANCE - relative_lowest
    if float(root.location.z) < minimum_root_z:
        root.location.z = minimum_root_z
        bpy.context.view_layer.update()
    root["animind_min_root_z"] = float(minimum_root_z)
    root["animind_rest_z"] = float(root.location.z)
    constraint = root.constraints.new(type="LIMIT_LOCATION")
    constraint.name = "AniMind ground contact"
    constraint.use_min_z = True
    constraint.min_z = float(minimum_root_z)
    constraint.owner_space = "WORLD"
    if hasattr(constraint, "use_transform_limit"):
        constraint.use_transform_limit = True


def minimum_root_z(obj: Any) -> float:
    try:
        return float(obj.get("animind_min_root_z", GROUND_SURFACE_Z + GROUND_CLEARANCE))
    except (AttributeError, TypeError, ValueError):
        return GROUND_SURFACE_Z + GROUND_CLEARANCE


def animate_action(
    action: Mapping[str, Any],
    roots: Mapping[str, Any],
    strategy: str,
    fps: int,
    scene_end_frame: int,
) -> None:
    actor_id = action.get("actorId")
    target_id = action.get("targetId")
    actor = roots.get(actor_id) if isinstance(actor_id, str) else None
    target = roots.get(target_id) if isinstance(target_id, str) else None
    subject = target if strategy in {"lower", "raise"} and target is not None else actor
    if subject is None:
        subject = next(iter(roots.values()), None)
    if subject is None:
        return
    start = max(1, 1 + int(round(float(action.get("startOffsetSeconds", 0)) * fps)))
    end = min(
        scene_end_frame,
        max(start + 1, start + int(round(float(action.get("durationSeconds", 1)) * fps))),
    )
    midpoint = max(start + 1, (start + end) // 2)
    keyframe_transform(subject, start)

    floor_z = minimum_root_z(subject)

    if strategy == "move":
        original = subject.location.copy()
        if target is not None and target is not subject:
            delta = target.location - subject.location
            destination = subject.location + delta * 0.70
        else:
            destination = subject.location.copy()
            destination.x += 1.8
        subject.location = original + (destination - original) * 0.50
        subject.location.z = max(floor_z, original.z) + 0.035
        keyframe_transform(subject, midpoint)
        subject.location = destination
        subject.location.z = max(floor_z, original.z)
        keyframe_transform(subject, end)
    elif strategy == "climb":
        if target is not None and target is not subject:
            subject.location.x = target.location.x - 0.15
            subject.location.y = target.location.y
            subject.location.z = target.location.z + 0.65
        else:
            subject.location.x += 0.9
            subject.location.z += 0.85
        keyframe_transform(subject, end)
    elif strategy == "lower":
        final_z = max(floor_z, subject.location.z - 1.0)
        subject.location.z += 1.0
        keyframe_transform(subject, start)
        subject.location.z = final_z
        keyframe_transform(subject, end)
    elif strategy == "raise":
        subject.location.z += 1.2
        keyframe_transform(subject, end)
    elif strategy == "touch":
        original_z = float(subject.location.z)
        subject.location.z = max(floor_z, original_z - 0.16)
        subject.rotation_euler[1] += 0.12
        keyframe_transform(subject, midpoint)
        subject.location.z = original_z
        subject.rotation_euler[1] -= 0.12
        keyframe_transform(subject, end)
    elif strategy == "turn-or-search":
        subject.rotation_euler[2] -= 0.28
        keyframe_transform(subject, start)
        subject.rotation_euler[2] += 0.56
        keyframe_transform(subject, midpoint)
        subject.rotation_euler[2] -= 0.28
        keyframe_transform(subject, end)
    elif strategy == "twitch":
        original = subject.rotation_euler[2]
        for index, frame in enumerate(range(start, end + 1, max(1, fps // 6))):
            subject.rotation_euler[2] = original + (0.10 if index % 2 else -0.10)
            keyframe_transform(subject, frame)
        subject.rotation_euler[2] = original
        keyframe_transform(subject, end)
    elif strategy == "slip-and-recover":
        original_z = float(subject.location.z)
        original_y = float(subject.location.y)
        original_yaw = float(subject.rotation_euler[2])
        # A slip is expressed sideways. Pitching and sinking the whole root
        # made small creatures visibly intersect the ground plane.
        subject.location.z = max(floor_z + 0.01, original_z - 0.035)
        subject.location.y = original_y + 0.10
        subject.rotation_euler[2] = original_yaw + 0.16
        keyframe_transform(subject, midpoint)
        subject.location.z = original_z
        subject.location.y = original_y
        subject.rotation_euler[2] = original_yaw
        subject.location.x += 0.35
        keyframe_transform(subject, end)
    elif strategy == "appear":
        final_scale = subject.scale.copy()
        subject.scale = (0.001, 0.001, 0.001)
        keyframe_transform(subject, start)
        subject.scale = final_scale
        keyframe_transform(subject, end)
    elif strategy == "disappear":
        subject.scale = (0.001, 0.001, 0.001)
        keyframe_transform(subject, end)
    else:
        subject.location.x += 0.28
        subject.location.z += 0.08
        subject.rotation_euler[2] += 0.10
        keyframe_transform(subject, midpoint)
        subject.location.z -= 0.08
        subject.rotation_euler[2] -= 0.10
        keyframe_transform(subject, end)


def set_smooth_interpolation(roots: Iterable[Any]) -> None:
    visited = set()
    for root in roots:
        objects = [root, *_descendants(root)]
        instance_collection = getattr(root, "instance_collection", None)
        if instance_collection is not None:
            objects.extend(list(getattr(instance_collection, "all_objects", [])))
        for obj in objects:
            pointer = obj.as_pointer() if hasattr(obj, "as_pointer") else id(obj)
            if pointer in visited:
                continue
            visited.add(pointer)
            animation_data = getattr(obj, "animation_data", None)
            action = getattr(animation_data, "action", None)
            if action is None:
                continue
            for curve in getattr(action, "fcurves", []):
                for point in curve.keyframe_points:
                    point.interpolation = "BEZIER"
                    if hasattr(point, "handle_left_type"):
                        point.handle_left_type = "AUTO_CLAMPED"
                    if hasattr(point, "handle_right_type"):
                        point.handle_right_type = "AUTO_CLAMPED"


def set_supported_property(owner: Any, names: Sequence[str], value: Any) -> bool:
    if owner is None:
        return False
    for name in names:
        if not hasattr(owner, name):
            continue
        try:
            setattr(owner, name, value)
            return True
        except (AttributeError, TypeError, ValueError):
            continue
    return False


def configure_render(
    bpy: Any,
    quality: Quality,
    output_path: Path,
    frame_end: int,
    use_motion_blur: bool = True,
) -> None:
    scene = bpy.context.scene
    engine_set = False
    for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            scene.render.engine = engine
            engine_set = True
            break
        except (TypeError, ValueError):
            continue
    if not engine_set:
        raise BlenderAdapterError("This Blender build does not provide the Eevee render engine.")
    scene.render.resolution_x = quality.width
    scene.render.resolution_y = quality.height
    scene.render.resolution_percentage = 100
    scene.render.fps = quality.fps
    scene.frame_start = 1
    scene.frame_end = frame_end
    scene.render.filepath = str(output_path)
    scene.render.use_file_extension = True
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    preferred_rate = "PERC_LOSSLESS" if quality.name == "final" else "HIGH"
    try:
        scene.render.ffmpeg.constant_rate_factor = preferred_rate
    except (TypeError, ValueError):
        scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.film_transparent = False
    if hasattr(scene.render, "use_persistent_data"):
        scene.render.use_persistent_data = True
    eevee = getattr(scene, "eevee", None)
    set_supported_property(eevee, ("taa_render_samples", "taa_samples"), quality.samples)
    set_supported_property(eevee, ("use_gtao",), True)
    set_supported_property(eevee, ("use_raytracing",), quality.name != "draft")

    motion_owners = (scene.render, eevee)
    motion_enabled = False
    for owner in motion_owners:
        motion_enabled = set_supported_property(owner, ("use_motion_blur",), use_motion_blur) or motion_enabled
        set_supported_property(owner, ("motion_blur_shutter",), 0.36)
        set_supported_property(owner, ("motion_blur_position",), "CENTER")
        set_supported_property(
            owner,
            ("motion_blur_steps", "motion_blur_samples"),
            2 if quality.name == "draft" else 4 if quality.name == "preview" else 8,
        )
        set_supported_property(owner, ("motion_blur_max",), 24)
    if use_motion_blur and not motion_enabled:
        print("Warning: this Blender build did not expose an Eevee motion-blur switch.")
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except (TypeError, ValueError):
        pass


def write_state(path: Path, state: MutableMapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def render_inside_blender(
    args: argparse.Namespace,
    storyboard: Mapping[str, Any],
    scenes: Sequence[Mapping[str, Any]],
    quality: Quality,
    catalog: Mapping[str, Any],
    catalog_path: Optional[Path],
    output_dir: Path,
) -> None:
    bpy, Vector, _Euler = _load_bpy()
    entities = entity_catalog(storyboard)
    aliases = require_mapping(catalog.get("actionAliases", {}), "action aliases")
    state_path = output_dir / "blender-execution.json"
    storyboard_hash = hashlib.sha256(
        json.dumps(storyboard, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    state: MutableMapping[str, Any] = {
        "stateVersion": STATE_VERSION,
        "adapterVersion": ADAPTER_VERSION,
        "storyboardSha256": storyboard_hash,
        "assetCatalogSha256": asset_catalog_fingerprint(catalog, catalog_path),
        "assetSources": asset_source_records(catalog),
        "quality": {
            "name": quality.name,
            "width": quality.width,
            "height": quality.height,
            "fps": quality.fps,
            "samples": quality.samples,
        },
        "renderFeatures": {
            "cameraStyle": args.camera_style,
            "depthOfField": not args.no_depth_of_field,
            "motionBlur": not args.no_motion_blur,
            "strictAssets": bool(args.strict_assets),
        },
        "scenes": {},
    }
    if state_path.exists():
        try:
            existing = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BlenderAdapterError(f"Could not read existing Blender state: {exc}") from exc
        state_mismatch = (
            existing.get("storyboardSha256") != storyboard_hash
            or existing.get("adapterVersion") != ADAPTER_VERSION
            or existing.get("assetCatalogSha256") != state["assetCatalogSha256"]
            or existing.get("quality") != state["quality"]
            or existing.get("renderFeatures") != state["renderFeatures"]
        )
        if state_mismatch:
            if not args.overwrite:
                raise BlenderAdapterError(
                    f"{state_path} belongs to a different storyboard, asset catalog, "
                    "adapter version, camera setup, or quality profile. "
                    "Choose another --output-dir or use --overwrite deliberately."
                )
        else:
            state = existing

    scenes_dir = output_dir / "scenes"
    blend_dir = output_dir / "blend"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_save_blend:
        blend_dir.mkdir(parents=True, exist_ok=True)

    for neutral_scene in scenes:
        scene_id = require_text(neutral_scene.get("sceneId"), "scene.sceneId")
        video_path = scenes_dir / f"{scene_id}.mp4"
        if video_path.exists() and not args.overwrite:
            entry = require_mapping(state.get("scenes", {}).get(scene_id, {}), f"state.{scene_id}")
            if entry.get("status") == "SUCCEEDED" and video_path.stat().st_size > 0:
                print(f"Preserved completed Blender scene: {scene_id}")
                continue
            raise BlenderAdapterError(
                f"Scene output already exists without matching completed state: {video_path}. "
                "Use another output directory or --overwrite."
            )

        clear_blender_scene(bpy)
        create_location(
            bpy,
            Vector,
            storyboard,
            neutral_scene,
            catalog,
            catalog_path,
        )
        visual = require_mapping(neutral_scene.get("visual"), f"{scene_id}.visual")
        active_entities: Dict[str, Mapping[str, Any]] = dict(entities)
        active_ids = scene_entity_ids(neutral_scene)
        if active_ids[0] not in active_entities and not visual.get("entityIds"):
            synthetic = synthetic_scene_entity(neutral_scene)
            active_entities[str(synthetic["entityId"])] = synthetic
        positions = scene_positions(neutral_scene, active_entities, active_ids)
        roots: Dict[str, Any] = {}
        for entity_id in active_ids:
            entity = active_entities.get(str(entity_id))
            if entity is None:
                raise BlenderAdapterError(f"{scene_id} references unknown entity {entity_id!r}.")
            root = create_entity_object(
                bpy,
                Vector,
                entity,
                catalog,
                catalog_path,
                strict_assets=args.strict_assets,
            )
            root.location = positions[str(entity_id)]
            offset = root.get("animind_location_offset", [0.0, 0.0, 0.0])
            try:
                if len(offset) == 3:
                    root.location = root.location + Vector(tuple(float(item) for item in offset))
            except (TypeError, ValueError):
                pass
            register_ground_clearance(bpy, Vector, root)
            roots[str(entity_id)] = root

        focus_tuple = (0.0, 0.0, 0.5)
        if roots:
            first = next(iter(roots.values()))
            focus_tuple = (float(first.location.x), float(first.location.y), float(first.location.z + 0.5))
        configure_world_and_lights(
            bpy,
            neutral_scene,
            focus_tuple,
            catalog,
            catalog_path,
        )
        create_camera(
            bpy,
            Vector,
            neutral_scene,
            roots,
            quality.fps,
            camera_style=args.camera_style,
            use_depth_of_field=not args.no_depth_of_field,
        )
        frame_end = max(2, int(round(float(neutral_scene.get("durationSeconds", 1)) * quality.fps)))
        for action_value in visual.get("actions", []):
            action = require_mapping(action_value, f"{scene_id}.action")
            strategy = action_strategy(str(action.get("verb", "")), aliases)
            animate_action(action, roots, strategy, quality.fps, frame_end)
        set_smooth_interpolation(roots.values())
        configure_render(
            bpy,
            quality,
            video_path,
            frame_end,
            use_motion_blur=not args.no_motion_blur,
        )

        state.setdefault("scenes", {})[scene_id] = {
            "status": "RENDERING",
            "video": str(video_path.relative_to(output_dir)),
            "blend": None,
        }
        write_state(state_path, state)
        if not args.no_save_blend:
            blend_path = blend_dir / f"{scene_id}.blend"
            bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
            state["scenes"][scene_id]["blend"] = str(blend_path.relative_to(output_dir))
            write_state(state_path, state)

        print(f"Rendering Blender scene: {scene_id} ({frame_end} frames)")
        bpy.ops.render.render(animation=True)
        if not video_path.is_file():
            # Blender versions vary in how they append an extension or frame
            # range to FFmpeg output paths. Normalize the result for assembly.
            candidates = sorted(
                scenes_dir.glob(f"{scene_id}*.mp4"),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                os.replace(candidates[0], video_path)
        if not video_path.is_file() or video_path.stat().st_size == 0:
            raise BlenderAdapterError(f"Blender did not create the expected video: {video_path}")
        state["scenes"][scene_id]["status"] = "SUCCEEDED"
        write_state(state_path, state)
        print(f"Completed Blender scene: {video_path}")


def ffprobe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def create_narration(
    storyboard: Mapping[str, Any],
    scenes: Sequence[Mapping[str, Any]],
    output_dir: Path,
    mode: str,
    voice: str,
    overwrite: bool,
) -> Dict[str, Path]:
    if mode == "off":
        return {}
    say = shutil.which("say")
    if say is None:
        if mode == "on":
            raise BlenderAdapterError(
                "macOS 'say' was not found, so narration could not be generated."
            )
        print("Narration skipped: macOS 'say' is not available on this computer.")
        return {}
    project = require_mapping(storyboard.get("project"), "project")
    rate = int(project.get("narrationWordsPerMinute", 145))
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Path] = {}
    for scene in scenes:
        scene_id = require_text(scene.get("sceneId"), "scene.sceneId")
        path = audio_dir / f"{scene_id}.aiff"
        if path.exists() and not overwrite:
            results[scene_id] = path
            continue
        narration = require_mapping(scene.get("narration"), f"{scene_id}.narration")
        text = require_text(narration.get("text"), f"{scene_id}.narration.text")
        try:
            subprocess.run(
                [say, "-v", voice, "-r", str(rate), "-o", str(path)],
                input=text,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            if mode == "auto":
                print(f"Narration skipped after macOS voice generation failed: {exc}")
                return {}
            raise BlenderAdapterError(f"macOS narration failed for {scene_id}.") from exc
        if not path.is_file() or path.stat().st_size == 0:
            raise BlenderAdapterError(f"Narration file was not created: {path}")
        results[scene_id] = path
    return results


def run_ffmpeg(command: Sequence[str], label: str) -> None:
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise BlenderAdapterError(f"FFmpeg failed while {label}.") from exc


def mux_narration(
    video: Path,
    audio: Path,
    output: Path,
    narration_offset: float,
    overwrite: bool,
) -> Path:
    if output.exists() and not overwrite:
        return output
    try:
        video_duration = ffprobe_duration(video)
        audio_duration = ffprobe_duration(audio)
    except (subprocess.CalledProcessError, ValueError) as exc:
        raise BlenderAdapterError(f"Could not inspect media duration for {video}.") from exc
    final_duration = max(video_duration, narration_offset + audio_duration)
    padding = max(0.0, final_duration - video_duration)
    delay_ms = max(0, int(round(narration_offset * 1000)))
    filter_graph = (
        f"[0:v]tpad=stop_mode=clone:stop_duration={padding:.3f},"
        f"trim=duration={final_duration:.3f},setpts=PTS-STARTPTS[v];"
        f"[1:a]asetpts=PTS-STARTPTS,adelay={delay_ms}:all=1,apad,"
        f"atrim=duration={final_duration:.3f}[a]"
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y" if overwrite else "-n",
        "-i",
        str(video),
        "-i",
        str(audio),
        "-filter_complex",
        filter_graph,
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        "-t",
        f"{final_duration:.3f}",
        str(output),
    ]
    run_ffmpeg(command, f"adding narration to {video.name}")
    return output


def ffconcat_quote(path: Path) -> str:
    return str(path.resolve()).replace("'", "'\\''")


def assemble_video(
    storyboard: Mapping[str, Any],
    scenes: Sequence[Mapping[str, Any]],
    output_dir: Path,
    narration_files: Mapping[str, Path],
    overwrite: bool,
    selected_scene: Optional[str],
) -> Path:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise BlenderAdapterError(
            "FFmpeg and ffprobe are required for assembly. Install them with: brew install ffmpeg"
        )
    scenes_dir = output_dir / "scenes"
    muxed_dir = output_dir / "muxed"
    muxed_dir.mkdir(parents=True, exist_ok=True)
    clips: List[Path] = []
    for scene in scenes:
        scene_id = require_text(scene.get("sceneId"), "scene.sceneId")
        video = scenes_dir / f"{scene_id}.mp4"
        if not video.is_file() or video.stat().st_size == 0:
            raise BlenderAdapterError(
                f"Rendered scene is missing: {video}. Run with --render first."
            )
        audio = narration_files.get(scene_id)
        if audio:
            narration = require_mapping(scene.get("narration"), f"{scene_id}.narration")
            offset = float(narration.get("startOffsetSeconds", 0.25))
            clips.append(
                mux_narration(
                    video,
                    audio,
                    muxed_dir / f"{scene_id}.mp4",
                    offset,
                    overwrite,
                )
            )
        else:
            clips.append(video)

    project = require_mapping(storyboard.get("project"), "project")
    suffix = f"-{selected_scene}" if selected_scene else ""
    final_path = output_dir / f"{safe_slug(project.get('title'))}{suffix}.mp4"
    if final_path.exists() and not overwrite:
        print(f"Final video already exists and was preserved: {final_path}")
        return final_path
    concat_file = output_dir / "ffmpeg-concat.txt"
    concat_file.write_text(
        "".join(f"file '{ffconcat_quote(clip)}'\n" for clip in clips),
        encoding="utf-8",
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y" if overwrite else "-n",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(final_path),
    ]
    run_ffmpeg(command, "assembling the final Blender video")
    print(f"Created final Blender video: {final_path}")
    return final_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(_program_arguments() if argv is None else argv)
    try:
        storyboard_path = args.storyboard.expanduser().resolve()
        storyboard = read_json(storyboard_path, "Storyboard")
        validate_storyboard(storyboard)
        catalog, catalog_path = read_asset_catalog(args.asset_catalog)
        scenes = choose_scenes(storyboard, args.scene)
        quality = quality_for(storyboard, args.quality, args.fps)
        output_dir = (
            args.output_dir or default_output_dir(storyboard_path)
        ).expanduser().resolve()

        if args.inside_blender:
            render_inside_blender(
                args,
                storyboard,
                scenes,
                quality,
                catalog,
                catalog_path,
                output_dir,
            )
            return 0

        print_plan(
            storyboard,
            scenes,
            quality,
            catalog,
            output_dir,
            camera_style=args.camera_style,
            motion_blur=not args.no_motion_blur,
            depth_of_field=not args.no_depth_of_field,
            strict_assets=args.strict_assets,
        )
        if not args.render and not args.assemble_only:
            scene_option = f" --scene {args.scene}" if args.scene else " --scene scene_001"
            catalog_option = (
                f" --asset-catalog {args.asset_catalog.name}"
                if args.asset_catalog is not None
                else ""
            )
            camera_option = (
                f" --camera-style {args.camera_style}"
                if args.camera_style != "auto"
                else ""
            )
            print("\nDry run only: Blender was not launched and no frames were rendered.")
            print(
                "Recommended first render:\n"
                f"  python3 {Path(__file__).name} {storyboard_path.name} "
                f"--render{scene_option} --quality draft"
                f"{catalog_option}{camera_option}"
            )
            return 0

        if args.render:
            blender = find_blender(args.blender)
            command = build_blender_command(
                blender,
                args,
                storyboard_path,
                output_dir,
            )
            print(f"Launching Blender: {blender}")
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as exc:
                raise BlenderAdapterError(
                    f"Blender stopped with exit code {exc.returncode}."
                ) from exc

        if args.no_assemble:
            print(f"Individual Blender scenes are in: {output_dir / 'scenes'}")
            return 0
        narration = create_narration(
            storyboard,
            scenes,
            output_dir,
            args.narration,
            args.voice,
            args.overwrite,
        )
        assemble_video(
            storyboard,
            scenes,
            output_dir,
            narration,
            args.overwrite,
            args.scene,
        )
        return 0
    except BlenderAdapterError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nStopped. Completed scene videos were preserved.", file=sys.stderr)
        return 130
    except OSError as exc:
        print(f"Error reading, writing, or launching a program: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
