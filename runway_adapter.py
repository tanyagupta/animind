#!/usr/bin/env python3
"""Adapt a renderer-neutral AniMind storyboard to the Runway API.

The companion ``story_board_maker.py`` creates a provider-independent JSON
storyboard.  This adapter converts that storyboard to Runway requests in
memory and handles the optional paid stage:

* validates the neutral storyboard and compiles Runway jobs in memory;
* optionally exports the compiled Runway JSON for inspection;
* validates the jobs and estimates their cost;
* performs a no-charge dry run by default;
* requires both ``--submit`` and a user-supplied spending ceiling;
* checks the Runway credit balance before submitting new work;
* saves every task ID immediately so interrupted runs can resume without
  submitting the same paid job twice;
* downloads completed outputs before their temporary URLs expire; and
* assembles the video and narration with FFmpeg when FFmpeg is installed.

Examples
--------
Preview jobs and estimated cost without making an API request::

    python3 runway_adapter.py renderer_neutral_sb.json

Submit, allowing no more than five dollars of estimated work::

    python3 runway_adapter.py renderer_neutral_sb.json --submit --max-cost-usd 5

If the command is interrupted, run the same submit command again.  Existing
task IDs are resumed from the state file rather than submitted again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from urllib.parse import urlparse


STATE_VERSION = 1
EXPECTED_STORYBOARD_KIND = "animind.renderer-neutral-storyboard"
EXPECTED_STORYBOARD_VERSION = "1.0"
EXPECTED_MANIFEST_KIND = "runway.story-animation-manifest"
EXPECTED_MANIFEST_VERSION = "1.0"
RUNWAY_KEY_ENV = "RUNWAYML_API_SECRET"
RUNWAY_API_VERSION = "2024-11-06"

# Runway API prices checked on 2026-08-07.  One credit costs USD 0.01.
# Refuse paid submission when a manifest uses a model absent from these tables;
# an unknown price should never be treated as zero.
VIDEO_CREDITS_PER_SECOND: Mapping[str, float] = {
    "gen4.5": 12.0,
}
VOICE_CREDITS_PER_50_CHARACTERS: Mapping[str, float] = {
    "eleven_multilingual_v2": 1.0,
}

DEFAULT_VIDEO_MODEL = "gen4.5"
DEFAULT_VOICE_MODEL = "eleven_multilingual_v2"
DEFAULT_VOICE = "Maya"
SUPPORTED_RATIOS = {"1280:720", "720:1280"}
VOICE_PRESETS = {
    "Maya", "Arjun", "Serene", "Bernard", "Billy", "Mark", "Clint",
    "Mabel", "Chad", "Leslie", "Eleanor", "Elias", "Elliot", "Grungle",
    "Brodie", "Sandra", "Kirk", "Kylie", "Lara", "Lisa", "Malachi",
    "Marlene", "Martin", "Miriam", "Monster", "Paula", "Pip", "Rusty",
    "Ragnar", "Xylar", "Maggie", "Jack", "Katie", "Noah", "James",
    "Rina", "Ella", "Mariah", "Frank", "Claudia", "Niki", "Vincent",
    "Kendrick", "Myrna", "Tom", "Wanda", "Benjamin", "Kiana", "Rachel",
}

VIDEO_ENDPOINT = "/v1/text_to_video"
VOICE_ENDPOINT = "/v1/text_to_speech"
TERMINAL_STATUSES = {"SUCCEEDED", "FAILED", "CANCELLED"}
ACTIVE_STATUSES = {"PENDING", "RUNNING", "THROTTLED"}


class RenderError(RuntimeError):
    """Raised for a safe, user-facing rendering failure."""


@dataclass(frozen=True)
class CostLine:
    job_id: str
    kind: str
    model: str
    credits: int


@dataclass(frozen=True)
class CostEstimate:
    lines: Tuple[CostLine, ...]

    @property
    def credits(self) -> int:
        return sum(line.credits for line in self.lines)

    @property
    def usd(self) -> float:
        return self.credits * 0.01


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adapt a renderer-neutral storyboard to Runway, then price, "
            "optionally submit, download, and assemble it. Dry run is the default."
        )
    )
    parser.add_argument(
        "storyboard",
        type=Path,
        help="Renderer-neutral JSON created by story_board_maker.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: outputs/runway beside the storyboard)",
    )
    parser.add_argument(
        "--export-manifest",
        type=Path,
        help=(
            "Optional diagnostic path for the compiled Runway JSON; it is not "
            "needed for submission"
        ),
    )
    parser.add_argument(
        "--video-model",
        default=DEFAULT_VIDEO_MODEL,
        help="Runway text-to-video model (default: gen4.5)",
    )
    parser.add_argument(
        "--voice-model",
        default=DEFAULT_VOICE_MODEL,
        help="Runway text-to-speech model (default: eleven_multilingual_v2)",
    )
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        choices=sorted(VOICE_PRESETS),
        help="Runway preset voice (default: Maya)",
    )
    parser.add_argument(
        "--ratio",
        choices=sorted(SUPPORTED_RATIOS),
        help="Override automatic landscape/portrait mapping",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Allow paid Runway jobs after validation, balance check, and confirmation",
    )
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        help="Required spending ceiling for --submit; submission is refused above it",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive RUN confirmation (the spending ceiling still applies)",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=2,
        help="Maximum simultaneous Runway tasks (default: 2)",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=8.0,
        help="Seconds between task-status checks (default: 8)",
    )
    parser.add_argument(
        "--timeout-minutes",
        type=float,
        default=60.0,
        help="Stop waiting after this many minutes; rerun to resume (default: 60)",
    )
    parser.add_argument(
        "--skip-assembly",
        action="store_true",
        help="Download assets but do not create the final MP4",
    )
    parser.add_argument(
        "--overwrite-final",
        action="store_true",
        help="Permit replacement of an existing assembled MP4",
    )
    return parser.parse_args(argv)


def read_storyboard(path: Path) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RenderError(f"Storyboard not found: {path}")
    try:
        storyboard = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise RenderError("Storyboard must be UTF-8 JSON.") from exc
    except json.JSONDecodeError as exc:
        raise RenderError(f"Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
    if not isinstance(storyboard, dict):
        raise RenderError("Storyboard root must be a JSON object.")
    return storyboard


def utf16_length(value: str) -> int:
    return len(value.encode("utf-16-le")) // 2


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RenderError(f"{label} must be a JSON object.")
    return value


def require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RenderError(f"{label} must be a non-empty string.")
    return value


def validate_neutral_storyboard(storyboard: Dict[str, Any]) -> None:
    if storyboard.get("kind") != EXPECTED_STORYBOARD_KIND:
        raise RenderError(
            f"Unsupported storyboard kind {storyboard.get('kind')!r}; expected "
            f"{EXPECTED_STORYBOARD_KIND!r}. This adapter does not accept the old "
            "Runway-specific JSON."
        )
    if storyboard.get("schemaVersion") != EXPECTED_STORYBOARD_VERSION:
        raise RenderError(
            f"Unsupported storyboard schemaVersion {storyboard.get('schemaVersion')!r}; "
            f"expected {EXPECTED_STORYBOARD_VERSION!r}."
        )
    try:
        from story_board_maker import StoryboardError, validate_storyboard
    except ImportError as exc:
        raise RenderError(
            "story_board_maker.py must be in the same folder as runway_adapter.py."
        ) from exc
    try:
        validate_storyboard(storyboard)
    except StoryboardError as exc:
        raise RenderError(f"Invalid renderer-neutral storyboard: {exc}") from exc


def truncate_utf16(value: str, limit: int) -> str:
    if utf16_length(value) <= limit:
        return value
    suffix = "..."
    result: List[str] = []
    used = 0
    for char in value:
        size = utf16_length(char)
        if used + size + len(suffix) > limit:
            break
        result.append(char)
        used += size
    return "".join(result).rstrip(" ,;:-") + suffix


def runway_ratio(storyboard: Mapping[str, Any], override: Optional[str]) -> str:
    if override:
        return override
    project = require_mapping(storyboard.get("project"), "project")
    frame = require_mapping(project.get("frame"), "project.frame")
    width = frame.get("width")
    height = frame.get("height")
    if not isinstance(width, int) or not isinstance(height, int):
        raise RenderError("project.frame width and height must be integers.")
    return "1280:720" if width >= height else "720:1280"


def split_runway_duration(duration: float) -> List[int]:
    """Map a neutral duration to one or more integer Runway clips of 2-10s."""
    total = max(2, int(math.ceil(duration)))
    parts: List[int] = []
    remaining = total
    while remaining > 10:
        part = 9 if remaining - 10 == 1 else 10
        parts.append(part)
        remaining -= part
    if remaining == 1 and parts:
        parts[-1] -= 1
        remaining = 2
    parts.append(remaining)
    if any(part < 2 or part > 10 for part in parts):
        raise RenderError(f"Could not split a {duration:g}-second scene into Runway clips.")
    return parts


def segment_seed(scene_seed: int, segment_number: int) -> int:
    if segment_number == 1:
        return scene_seed
    digest = hashlib.sha256(
        f"{scene_seed}:runway-segment:{segment_number}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big")


def build_runway_prompt(
    storyboard: Mapping[str, Any],
    scene: Mapping[str, Any],
    segment_number: int,
    segment_count: int,
) -> str:
    project = require_mapping(storyboard.get("project"), "project")
    visual = require_mapping(scene.get("visual"), f"{scene.get('sceneId')}.visual")
    camera = require_mapping(visual.get("camera"), "visual.camera")
    look = require_mapping(visual.get("look"), "visual.look")

    entities_by_id = {
        str(item.get("entityId")): item
        for item in storyboard.get("entities", [])
        if isinstance(item, dict)
    }
    locations_by_id = {
        str(item.get("locationId")): item
        for item in storyboard.get("locations", [])
        if isinstance(item, dict)
    }
    entity_descriptions = []
    for entity_id in visual.get("entityIds", []):
        entity = entities_by_id.get(str(entity_id))
        if entity:
            entity_descriptions.append(
                f"{entity.get('name')}: {entity.get('visualDescription')}"
            )
    location_id = visual.get("locationId")
    location = locations_by_id.get(str(location_id)) if location_id else None
    location_text = (
        str(location.get("visualDescription"))
        if location
        else str(visual.get("settingDescription", ""))
    )
    actions = visual.get("actions")
    if not isinstance(actions, list) or not actions:
        raise RenderError(f"{scene.get('sceneId')} has no visual actions.")
    action_text = " Then ".join(
        str(action.get("description"))
        for action in actions
        if isinstance(action, dict) and action.get("description")
    )
    continuity_parts = [str(visual.get("continuityNotes", ""))]
    for entity_id in visual.get("entityIds", []):
        entity = entities_by_id.get(str(entity_id))
        if entity and entity.get("continuityNotes"):
            continuity_parts.append(str(entity["continuityNotes"]))

    segment_note = ""
    if segment_count > 1:
        segment_note = (
            f"This is continuous clip {segment_number} of {segment_count} for the same "
            "scene; preserve exact subject, setting, screen direction, and motion continuity."
        )
    sections = (
        f"Animated film scene. {visual.get('summary', '')}.",
        f"Setting: {location_text}.",
        f"Visible entities: {'; '.join(entity_descriptions)}." if entity_descriptions else "",
        f"Action: {action_text}.",
        (
            "Camera: "
            f"{camera.get('shotType', '')} shot, {camera.get('angle', '')} angle, "
            f"{camera.get('movement', '')}; {camera.get('description', '')}."
        ),
        f"Lighting: {look.get('lighting', '')}.",
        f"Mood: {look.get('mood', '')}.",
        f"Continuity: {' '.join(part for part in continuity_parts if part)}.",
        f"Visual style: {project.get('visualStyle', '')}.",
        segment_note,
        "Continuous natural motion, one coherent shot, no text, no subtitles, no logos.",
    )
    prompt = " ".join(section.strip() for section in sections if section.strip())
    return truncate_utf16(prompt, 1000)


def compile_runway_manifest(
    storyboard: Mapping[str, Any],
    *,
    video_model: str,
    voice_model: str,
    voice: str,
    ratio_override: Optional[str],
) -> Dict[str, Any]:
    """Translate neutral semantics into a deterministic Runway execution plan."""
    if video_model not in VIDEO_CREDITS_PER_SECOND:
        raise RenderError(
            f"No verified price is configured for video model {video_model!r}; refusing paid use."
        )
    if voice_model not in VOICE_CREDITS_PER_50_CHARACTERS:
        raise RenderError(
            f"No verified price is configured for voice model {voice_model!r}; refusing paid use."
        )
    if voice not in VOICE_PRESETS:
        raise RenderError(f"Unsupported Runway voice preset: {voice!r}.")

    ratio = runway_ratio(storyboard, ratio_override)
    neutral_scenes = storyboard.get("scenes")
    if not isinstance(neutral_scenes, list) or not neutral_scenes:
        raise RenderError("Storyboard must contain at least one scene.")
    project = require_mapping(storyboard.get("project"), "project")

    scenes: List[Dict[str, Any]] = []
    jobs: List[Dict[str, Any]] = []
    video_track: List[Dict[str, Any]] = []
    voice_track: List[Dict[str, Any]] = []
    render_cursor = 0.0

    for scene_value in neutral_scenes:
        scene = require_mapping(scene_value, "scene")
        scene_id = require_nonempty_string(scene.get("sceneId"), "scene.sceneId")
        duration_value = scene.get("durationSeconds")
        if not isinstance(duration_value, (int, float)) or isinstance(duration_value, bool):
            raise RenderError(f"{scene_id}.durationSeconds must be numeric.")
        durations = split_runway_duration(float(duration_value))
        animation_ids: List[str] = []
        scene_start = render_cursor
        for segment_index, duration in enumerate(durations, start=1):
            if len(durations) == 1:
                job_id = f"{scene_id}_animation"
            else:
                job_id = f"{scene_id}_animation_{segment_index:02d}"
            animation_ids.append(job_id)
            seed_value = scene.get("seed")
            if not isinstance(seed_value, int) or isinstance(seed_value, bool):
                raise RenderError(f"{scene_id}.seed must be an integer.")
            jobs.append(
                {
                    "jobId": job_id,
                    "sceneId": scene_id,
                    "kind": "animation",
                    "method": "POST",
                    "endpoint": VIDEO_ENDPOINT,
                    "body": {
                        "model": video_model,
                        "promptText": build_runway_prompt(
                            storyboard,
                            scene,
                            segment_index,
                            len(durations),
                        ),
                        "ratio": ratio,
                        "duration": duration,
                        "seed": segment_seed(seed_value, segment_index),
                    },
                }
            )
            video_track.append(
                {
                    "sceneId": scene_id,
                    "jobId": job_id,
                    "startSeconds": round(render_cursor, 3),
                    "durationSeconds": duration,
                }
            )
            render_cursor += duration

        narration = require_mapping(scene.get("narration"), f"{scene_id}.narration")
        narration_text = require_nonempty_string(
            narration.get("text"), f"{scene_id}.narration.text"
        )
        if utf16_length(narration_text) > 1000:
            raise RenderError(
                f"{scene_id} narration exceeds Runway's 1000 UTF-16-unit limit. "
                "Shorten or split the neutral scene."
            )
        voice_job_id = f"{scene_id}_voice"
        jobs.append(
            {
                "jobId": voice_job_id,
                "sceneId": scene_id,
                "kind": "voice",
                "method": "POST",
                "endpoint": VOICE_ENDPOINT,
                "body": {
                    "model": voice_model,
                    "promptText": narration_text,
                    "voice": {"type": "runway-preset", "presetId": voice},
                },
            }
        )
        narration_offset = narration.get("startOffsetSeconds", 0.25)
        if not isinstance(narration_offset, (int, float)):
            raise RenderError(f"{scene_id}.narration.startOffsetSeconds must be numeric.")
        voice_track.append(
            {
                "sceneId": scene_id,
                "jobId": voice_job_id,
                "startSeconds": round(scene_start + float(narration_offset), 3),
                "gainDb": 0,
            }
        )
        scenes.append(
            {
                "sceneId": scene_id,
                "sequence": scene.get("sequence"),
                "startSeconds": round(scene_start, 3),
                "endSeconds": round(render_cursor, 3),
                "durationSeconds": round(render_cursor - scene_start, 3),
                "animationJobIds": animation_ids,
                "voiceJobId": voice_job_id,
                "transitionIn": scene.get("transitionIn"),
            }
        )

    manifest: Dict[str, Any] = {
        "schemaVersion": EXPECTED_MANIFEST_VERSION,
        "kind": EXPECTED_MANIFEST_KIND,
        "generatedAt": storyboard.get("generatedAt"),
        "source": {
            "storyboardKind": storyboard.get("kind"),
            "storyboardSchemaVersion": storyboard.get("schemaVersion"),
            "storyPath": require_mapping(storyboard.get("source"), "source").get("storyPath"),
        },
        "project": {
            "title": project.get("title"),
            "visualStyle": project.get("visualStyle"),
            "ratio": ratio,
            "estimatedDurationSeconds": round(render_cursor, 3),
            "sceneCount": len(scenes),
        },
        "runway": {
            "baseUrl": "https://api.dev.runwayml.com",
            "apiVersion": RUNWAY_API_VERSION,
            "authorizationEnvironmentVariable": RUNWAY_KEY_ENV,
            "videoModel": video_model,
            "voiceModel": voice_model,
            "voicePreset": voice,
        },
        "scenes": scenes,
        "jobs": jobs,
        "assembly": {
            "videoTrack": video_track,
            "voiceTrack": voice_track,
            "finalDurationSeconds": round(render_cursor, 3),
        },
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    if manifest.get("kind") != EXPECTED_MANIFEST_KIND:
        raise RenderError(
            f"Unsupported manifest kind {manifest.get('kind')!r}; expected {EXPECTED_MANIFEST_KIND!r}."
        )
    if manifest.get("schemaVersion") != EXPECTED_MANIFEST_VERSION:
        raise RenderError(
            f"Unsupported schemaVersion {manifest.get('schemaVersion')!r}; "
            f"expected {EXPECTED_MANIFEST_VERSION!r}."
        )

    jobs_value = manifest.get("jobs")
    if not isinstance(jobs_value, list) or not jobs_value:
        raise RenderError("Manifest must contain a non-empty jobs array.")

    jobs: List[Mapping[str, Any]] = []
    seen_ids = set()
    for index, value in enumerate(jobs_value):
        job = require_mapping(value, f"jobs[{index}]")
        job_id = require_nonempty_string(job.get("jobId"), f"jobs[{index}].jobId")
        if job_id in seen_ids:
            raise RenderError(f"Duplicate jobId: {job_id}")
        seen_ids.add(job_id)
        if job.get("method") != "POST":
            raise RenderError(f"{job_id}: only POST jobs are supported.")

        kind = job.get("kind")
        endpoint = job.get("endpoint")
        body = require_mapping(job.get("body"), f"{job_id}.body")
        model = require_nonempty_string(body.get("model"), f"{job_id}.body.model")
        prompt = require_nonempty_string(body.get("promptText"), f"{job_id}.body.promptText")

        if kind == "animation":
            if endpoint != VIDEO_ENDPOINT:
                raise RenderError(f"{job_id}: animation endpoint must be {VIDEO_ENDPOINT}.")
            allowed = {"model", "promptText", "ratio", "duration", "seed"}
            unexpected = set(body) - allowed
            if unexpected:
                raise RenderError(f"{job_id}: unsupported animation fields: {sorted(unexpected)}")
            duration = body.get("duration")
            if not isinstance(duration, int) or isinstance(duration, bool) or not 2 <= duration <= 10:
                raise RenderError(f"{job_id}: animation duration must be an integer from 2 to 10.")
            if body.get("ratio") not in {"1280:720", "720:1280"}:
                raise RenderError(f"{job_id}: unsupported gen4.5 ratio {body.get('ratio')!r}.")
            if utf16_length(prompt) > 1000:
                raise RenderError(f"{job_id}: animation prompt exceeds 1000 UTF-16 code units.")
            seed = body.get("seed")
            if seed is not None and (
                not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 4_294_967_295
            ):
                raise RenderError(f"{job_id}: seed must be an integer from 0 to 4294967295.")
            if model not in VIDEO_CREDITS_PER_SECOND:
                raise RenderError(
                    f"{job_id}: no verified price is configured for video model {model!r}; refusing paid use."
                )
        elif kind == "voice":
            if endpoint != VOICE_ENDPOINT:
                raise RenderError(f"{job_id}: voice endpoint must be {VOICE_ENDPOINT}.")
            allowed = {"model", "promptText", "voice"}
            unexpected = set(body) - allowed
            if unexpected:
                raise RenderError(f"{job_id}: unsupported voice fields: {sorted(unexpected)}")
            if utf16_length(prompt) > 1000:
                raise RenderError(f"{job_id}: narration exceeds 1000 UTF-16 code units.")
            voice = require_mapping(body.get("voice"), f"{job_id}.body.voice")
            if voice.get("type") != "runway-preset":
                raise RenderError(f"{job_id}: only runway-preset voices are supported.")
            require_nonempty_string(voice.get("presetId"), f"{job_id}.body.voice.presetId")
            if model not in VOICE_CREDITS_PER_50_CHARACTERS:
                raise RenderError(
                    f"{job_id}: no verified price is configured for voice model {model!r}; refusing paid use."
                )
        else:
            raise RenderError(f"{job_id}: unsupported job kind {kind!r}.")
        jobs.append(job)

    scenes = manifest.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise RenderError("Manifest must contain a non-empty scenes array.")
    scene_job_ids = set()
    for index, scene_value in enumerate(scenes):
        scene = require_mapping(scene_value, f"scenes[{index}]")
        animation_references = scene.get("animationJobIds")
        if not isinstance(animation_references, list) or not animation_references:
            raise RenderError(f"scenes[{index}].animationJobIds must be a non-empty array.")
        references = [*animation_references, scene.get("voiceJobId")]
        for reference_index, reference_value in enumerate(references):
            reference = require_nonempty_string(
                reference_value,
                f"scenes[{index}] job reference {reference_index}",
            )
            if reference not in seen_ids:
                raise RenderError(f"scenes[{index}] references missing job {reference!r}.")
            scene_job_ids.add(reference)
    if scene_job_ids != seen_ids:
        extra = sorted(seen_ids - scene_job_ids)
        raise RenderError(f"Jobs not referenced by any scene: {extra}")
    return jobs


def estimate_cost(jobs: Iterable[Mapping[str, Any]]) -> CostEstimate:
    lines: List[CostLine] = []
    for job in jobs:
        body = require_mapping(job.get("body"), f"{job.get('jobId')}.body")
        job_id = str(job["jobId"])
        kind = str(job["kind"])
        model = str(body["model"])
        if kind == "animation":
            credits = math.ceil(float(body["duration"]) * VIDEO_CREDITS_PER_SECOND[model])
        else:
            character_count = utf16_length(str(body["promptText"]))
            unit_price = VOICE_CREDITS_PER_50_CHARACTERS[model]
            credits = max(1, math.ceil(character_count / 50.0 * unit_price))
        lines.append(CostLine(job_id=job_id, kind=kind, model=model, credits=credits))
    return CostEstimate(tuple(lines))


def default_output_dir(storyboard_path: Path) -> Path:
    return storyboard_path.resolve().parent / "outputs" / "runway"


def export_manifest(path: Path, manifest: Mapping[str, Any]) -> Path:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return resolved


def safe_title(value: Any) -> str:
    text = value if isinstance(value, str) else "runway-story"
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return (slug[:80].rstrip("-") or "runway-story") + ".mp4"


def print_estimate(manifest: Mapping[str, Any], estimate: CostEstimate) -> None:
    project = manifest.get("project") if isinstance(manifest.get("project"), dict) else {}
    title = project.get("title") or "Untitled"
    animations = [line for line in estimate.lines if line.kind == "animation"]
    voices = [line for line in estimate.lines if line.kind == "voice"]
    video_credits = sum(line.credits for line in animations)
    voice_credits = sum(line.credits for line in voices)
    print(f"Project: {title}")
    print(f"Jobs: {len(animations)} animation + {len(voices)} narration = {len(estimate.lines)} total")
    print(f"Estimated video cost: {video_credits} credits (${video_credits * 0.01:.2f})")
    print(f"Estimated voice cost: {voice_credits} credits (${voice_credits * 0.01:.2f})")
    print(f"Estimated maximum total: {estimate.credits} credits (${estimate.usd:.2f})")


def validate_key() -> None:
    key = os.environ.get(RUNWAY_KEY_ENV, "")
    if not key:
        raise RenderError(f"{RUNWAY_KEY_ENV} is not set in this Terminal window.")
    if not key.startswith("key_") or key.count("key_") != 1:
        raise RenderError(f"{RUNWAY_KEY_ENV} does not have the expected single key_ prefix.")
    if any(char.isspace() for char in key):
        raise RenderError(f"{RUNWAY_KEY_ENV} contains whitespace.")


def initial_state(manifest_sha256: str, jobs: Sequence[Mapping[str, Any]], estimate: CostEstimate) -> Dict[str, Any]:
    return {
        "stateVersion": STATE_VERSION,
        "manifestSha256": manifest_sha256,
        "createdAt": utc_now(),
        "updatedAt": utc_now(),
        "estimatedCredits": estimate.credits,
        "estimatedUsd": round(estimate.usd, 2),
        "jobs": {
            str(job["jobId"]): {
                "taskId": None,
                "status": "NOT_SUBMITTED",
                "submittedAt": None,
                "completedAt": None,
                "failure": None,
                "files": [],
            }
            for job in jobs
        },
    }


def load_or_create_state(
    state_path: Path,
    manifest_sha256: str,
    jobs: Sequence[Mapping[str, Any]],
    estimate: CostEstimate,
) -> Dict[str, Any]:
    if not state_path.exists():
        return initial_state(manifest_sha256, jobs, estimate)
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RenderError(f"Cannot read existing state file {state_path}: {exc}") from exc
    if not isinstance(state, dict) or state.get("stateVersion") != STATE_VERSION:
        raise RenderError(f"Unsupported or invalid state file: {state_path}")
    if state.get("manifestSha256") != manifest_sha256:
        raise RenderError(
            f"{state_path} belongs to a different manifest. Move it aside or choose another --output-dir."
        )
    job_states = state.get("jobs")
    if not isinstance(job_states, dict) or set(job_states) != {str(job["jobId"]) for job in jobs}:
        raise RenderError(f"The job list in {state_path} does not match this manifest.")
    return state


def write_state(path: Path, state: MutableMapping[str, Any]) -> None:
    state["updatedAt"] = utc_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_path, path)


def import_runway() -> Tuple[Any, Any, Any]:
    try:
        from runwayml import RunwayML
        from runwayml.lib.polling import TaskFailedError, TaskTimeoutError
    except ImportError as exc:
        raise RenderError(
            "The Runway SDK is not installed. Run: python3 -m pip install runwayml"
        ) from exc
    return RunwayML, TaskFailedError, TaskTimeoutError


def submit_job(client: Any, job: Mapping[str, Any]) -> str:
    body = require_mapping(job["body"], f"{job['jobId']}.body")
    if job["kind"] == "animation":
        kwargs: Dict[str, Any] = {
            "model": body["model"],
            "prompt_text": body["promptText"],
            "ratio": body["ratio"],
            "duration": body["duration"],
        }
        if body.get("seed") is not None:
            kwargs["seed"] = body["seed"]
        response = client.text_to_video.create(**kwargs)
    elif job["kind"] == "voice":
        voice = require_mapping(body["voice"], f"{job['jobId']}.body.voice")
        response = client.text_to_speech.create(
            model=body["model"],
            prompt_text=body["promptText"],
            voice={"type": voice["type"], "preset_id": voice["presetId"]},
        )
    else:
        raise RenderError(f"Unsupported job kind {job['kind']!r}.")
    task_id = getattr(response, "id", None)
    if not isinstance(task_id, str) or not task_id:
        raise RenderError(f"Runway returned no task ID for {job['jobId']}.")
    return task_id


def status_payload(task: Any) -> Tuple[str, Optional[str], List[str]]:
    status = str(getattr(task, "status", "UNKNOWN"))
    failure = getattr(task, "failure", None)
    output = getattr(task, "output", None)
    urls = [str(value) for value in output] if isinstance(output, list) else []
    return status, str(failure) if failure else None, urls


def guess_extension(url: str, content_type: Optional[str], kind: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".mp4", ".mov", ".webm", ".mp3", ".wav", ".ogg", ".opus", ".m4a"}:
        return suffix
    normalized = (content_type or "").split(";", 1)[0].strip().lower()
    by_type = {
        "video/mp4": ".mp4",
        "video/quicktime": ".mov",
        "video/webm": ".webm",
        "audio/mpeg": ".mp3",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/ogg": ".ogg",
        "audio/opus": ".opus",
        "audio/mp4": ".m4a",
    }
    return by_type.get(normalized, ".mp4" if kind == "animation" else ".mp3")


def download_outputs(
    urls: Sequence[str],
    job: Mapping[str, Any],
    output_dir: Path,
) -> List[str]:
    try:
        import certifi
        import httpx
    except ImportError as exc:
        raise RenderError(
            "Secure downloads require httpx and certifi. "
            "Install them with: python3 -m pip install httpx certifi"
        ) from exc
    if not urls:
        raise RenderError(f"{job['jobId']} succeeded but returned no output URL.")
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    with httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(180.0, connect=30.0),
        verify=certifi.where(),
        headers={"User-Agent": "story-to-runway-renderer/1.1"},
    ) as download_client:
        for index, url in enumerate(urls, start=1):
            partial: Optional[Path] = None
            try:
                with download_client.stream("GET", url) as response:
                    response.raise_for_status()
                    extension = guess_extension(
                        url,
                        response.headers.get("Content-Type"),
                        str(job["kind"]),
                    )
                    suffix = "" if len(urls) == 1 else f"_{index}"
                    target = assets_dir / f"{job['jobId']}{suffix}{extension}"
                    partial = target.with_suffix(target.suffix + ".part")
                    with partial.open("wb") as handle:
                        for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                            handle.write(chunk)
                if not partial.exists() or partial.stat().st_size == 0:
                    raise RenderError(f"Downloaded an empty file for {job['jobId']}.")
                os.replace(partial, target)
            except RenderError:
                if partial and partial.exists():
                    partial.unlink()
                raise
            except Exception as exc:
                if partial and partial.exists():
                    partial.unlink()
                raise RenderError(f"Could not download output for {job['jobId']}: {exc}") from exc
            saved.append(str(target.relative_to(output_dir)))
    return saved


def local_files_present(job_state: Mapping[str, Any], output_dir: Path) -> bool:
    files = job_state.get("files")
    return bool(
        isinstance(files, list)
        and files
        and all((output_dir / str(item)).is_file() and (output_dir / str(item)).stat().st_size > 0 for item in files)
    )


def confirm_submission(
    estimate: CostEstimate,
    new_job_count: int,
    remaining_credits: int,
    balance: int,
    assume_yes: bool,
) -> None:
    print(f"Runway credit balance: {balance} credits (${balance * 0.01:.2f})")
    print(f"New paid jobs to submit: {new_job_count}")
    if new_job_count:
        print(
            f"Estimated cost of jobs not yet submitted: {remaining_credits} credits "
            f"(${remaining_credits * 0.01:.2f})"
        )
    if balance < remaining_credits:
        raise RenderError(
            f"The remaining unsubmitted jobs are estimated at {remaining_credits} credits, "
            f"but the organization balance is {balance}."
        )
    if new_job_count == 0 or assume_yes:
        return
    if not sys.stdin.isatty():
        raise RenderError("Interactive confirmation is unavailable; rerun with --yes after reviewing the estimate.")
    answer = input(
        f"Type RUN to authorize up to ${estimate.usd:.2f} for this manifest, or press Return to cancel: "
    ).strip()
    if answer != "RUN":
        raise RenderError("Cancelled; no new paid jobs were submitted.")


def process_jobs(
    client: Any,
    jobs: Sequence[Mapping[str, Any]],
    state: MutableMapping[str, Any],
    state_path: Path,
    output_dir: Path,
    max_in_flight: int,
    poll_seconds: float,
    timeout_minutes: float,
) -> bool:
    jobs_by_id = {str(job["jobId"]): job for job in jobs}
    ordered_ids = [str(job["jobId"]) for job in jobs]
    started = time.monotonic()
    submission_blocked = False
    consecutive_poll_errors = 0

    while True:
        inflight_ids: List[str] = []
        unresolved_ids: List[str] = []
        failed_ids: List[str] = []
        uncertain_ids: List[str] = []
        for job_id in ordered_ids:
            entry = state["jobs"][job_id]
            status = entry.get("status")
            if status == "NOT_SUBMITTED":
                unresolved_ids.append(job_id)
            elif status in ACTIVE_STATUSES or (entry.get("taskId") and status not in TERMINAL_STATUSES):
                inflight_ids.append(job_id)
            elif status in {"FAILED", "CANCELLED"}:
                failed_ids.append(job_id)
            elif status == "SUBMISSION_UNCERTAIN":
                uncertain_ids.append(job_id)
            elif status == "SUCCEEDED" and not local_files_present(entry, output_dir):
                inflight_ids.append(job_id)

        if uncertain_ids:
            raise RenderError(
                "These jobs have an uncertain submission result and will not be retried automatically: "
                f"{', '.join(uncertain_ids)}. Check the Runway dashboard before taking further action."
            )

        if failed_ids:
            submission_blocked = True

        # Poll submitted tasks first.  Never infer success from a previous run
        # unless the downloaded local file is still present.
        for job_id in list(inflight_ids):
            entry = state["jobs"][job_id]
            task_id = entry.get("taskId")
            if not task_id:
                raise RenderError(f"State for {job_id} has no Runway task ID.")
            try:
                task = client.tasks.retrieve(task_id)
                consecutive_poll_errors = 0
            except Exception as exc:
                consecutive_poll_errors += 1
                print(f"Temporary status-check error for {job_id}: {exc}", file=sys.stderr)
                if consecutive_poll_errors >= 5:
                    raise RenderError(
                        "Runway status checks failed five times. Rerun the same command to resume safely."
                    ) from exc
                continue

            status, failure, urls = status_payload(task)
            entry["status"] = status
            entry["failure"] = failure
            if status == "SUCCEEDED":
                if not local_files_present(entry, output_dir):
                    entry["files"] = download_outputs(urls, jobs_by_id[job_id], output_dir)
                entry["completedAt"] = utc_now()
                print(f"Completed and downloaded: {job_id}")
            elif status in {"FAILED", "CANCELLED"}:
                entry["completedAt"] = utc_now()
                submission_blocked = True
                print(f"Runway task {job_id} {status.lower()}: {failure or 'no reason supplied'}", file=sys.stderr)
            else:
                progress = getattr(task, "progress", None)
                detail = f" ({float(progress) * 100:.0f}%)" if isinstance(progress, (int, float)) else ""
                print(f"Waiting: {job_id} — {status}{detail}")
            write_state(state_path, state)

        inflight_ids = [
            job_id
            for job_id in ordered_ids
            if state["jobs"][job_id].get("status") in ACTIVE_STATUSES
        ]
        unresolved_ids = [
            job_id
            for job_id in ordered_ids
            if state["jobs"][job_id].get("status") == "NOT_SUBMITTED"
        ]

        while not submission_blocked and unresolved_ids and len(inflight_ids) < max_in_flight:
            job_id = unresolved_ids.pop(0)
            print(f"Submitting paid job: {job_id}")
            try:
                task_id = submit_job(client, jobs_by_id[job_id])
            except Exception as exc:
                # A transport failure can be ambiguous: the server may have
                # accepted the paid task without returning its ID. Do not retry
                # automatically because that could create a duplicate charge.
                entry = state["jobs"][job_id]
                entry["status"] = "SUBMISSION_UNCERTAIN"
                entry["submittedAt"] = utc_now()
                entry["failure"] = str(exc)
                write_state(state_path, state)
                raise RenderError(
                    f"Submission of {job_id} did not return safely: {exc}. "
                    "Do not retry this job automatically; check the Runway dashboard first."
                ) from exc
            entry = state["jobs"][job_id]
            entry["taskId"] = task_id
            entry["status"] = "PENDING"
            entry["submittedAt"] = utc_now()
            write_state(state_path, state)
            inflight_ids.append(job_id)

        statuses = [state["jobs"][job_id].get("status") for job_id in ordered_ids]
        if all(status == "SUCCEEDED" for status in statuses):
            return True
        if submission_blocked and not any(status in ACTIVE_STATUSES for status in statuses):
            return False
        if time.monotonic() - started > timeout_minutes * 60:
            raise RenderError(
                f"Stopped waiting after {timeout_minutes:g} minutes. "
                "The task IDs are saved; rerun the same command to resume without duplicate submissions."
            )
        time.sleep(poll_seconds)


def ffprobe_duration(path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return float(result.stdout.strip())


def first_asset(state: Mapping[str, Any], output_dir: Path, job_id: str) -> Path:
    entry = require_mapping(require_mapping(state.get("jobs"), "state.jobs").get(job_id), f"state.{job_id}")
    files = entry.get("files")
    if not isinstance(files, list) or not files:
        raise RenderError(f"No downloaded asset recorded for {job_id}.")
    path = output_dir / str(files[0])
    if not path.is_file():
        raise RenderError(f"Downloaded asset is missing: {path}")
    return path


def assemble_final_video(
    manifest: Mapping[str, Any],
    state: Mapping[str, Any],
    output_dir: Path,
    final_path: Path,
    overwrite: bool,
) -> Optional[Path]:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("FFmpeg was not found, so the individual assets were downloaded but not assembled.")
        print("Install it with 'brew install ffmpeg', then rerun the same command.")
        return None
    if final_path.exists() and not overwrite:
        print(f"Final video already exists and was preserved: {final_path}")
        print("Use --overwrite-final only if you intend to replace it.")
        return final_path

    assembly = require_mapping(manifest.get("assembly"), "assembly")
    video_track = assembly.get("videoTrack")
    voice_track = assembly.get("voiceTrack")
    if not isinstance(video_track, list) or not video_track:
        raise RenderError("assembly.videoTrack must be a non-empty array.")
    if not isinstance(voice_track, list) or not voice_track:
        raise RenderError("assembly.voiceTrack must be a non-empty array.")

    video_inputs: List[Tuple[Path, float]] = []
    for item_value in video_track:
        item = require_mapping(item_value, "assembly.videoTrack item")
        job_id = require_nonempty_string(item.get("jobId"), "videoTrack.jobId")
        duration = float(item.get("durationSeconds"))
        video_inputs.append((first_asset(state, output_dir, job_id), duration))

    voice_inputs: List[Tuple[Path, float, float]] = []
    for item_value in voice_track:
        item = require_mapping(item_value, "assembly.voiceTrack item")
        job_id = require_nonempty_string(item.get("jobId"), "voiceTrack.jobId")
        start = float(item.get("startSeconds", 0.0))
        gain_db = float(item.get("gainDb", 0.0))
        voice_inputs.append((first_asset(state, output_dir, job_id), start, gain_db))

    ratio = require_mapping(manifest.get("project"), "project").get("ratio")
    if ratio == "1280:720":
        width, height = 1280, 720
    elif ratio == "720:1280":
        width, height = 720, 1280
    else:
        raise RenderError(f"Unsupported assembly ratio: {ratio!r}")

    declared_final = float(assembly.get("finalDurationSeconds", sum(duration for _, duration in video_inputs)))
    audio_ends = []
    for path, start, _gain in voice_inputs:
        try:
            audio_ends.append(start + ffprobe_duration(path))
        except (subprocess.CalledProcessError, ValueError) as exc:
            raise RenderError(f"Could not inspect narration duration for {path}: {exc}") from exc
    final_duration = max([declared_final, *audio_ends])
    video_duration = sum(duration for _, duration in video_inputs)
    padding = max(0.0, final_duration - video_duration)

    command: List[str] = ["ffmpeg", "-hide_banner", "-loglevel", "warning"]
    command.append("-y" if overwrite else "-n")
    for path, _duration in video_inputs:
        command.extend(["-i", str(path)])
    for path, _start, _gain in voice_inputs:
        command.extend(["-i", str(path)])

    filters: List[str] = []
    video_labels = []
    for index, (_path, duration) in enumerate(video_inputs):
        label = f"v{index}"
        video_labels.append(f"[{label}]")
        filters.append(
            f"[{index}:v]setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={duration:.3f},"
            f"trim=duration={duration:.3f},"
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,format=yuv420p[{label}]"
        )
    filters.append(f"{''.join(video_labels)}concat=n={len(video_inputs)}:v=1:a=0[vcat]")
    if padding > 0.001:
        filters.append(
            f"[vcat]tpad=stop_mode=clone:stop_duration={padding:.3f},"
            f"trim=duration={final_duration:.3f}[vout]"
        )
    else:
        filters.append(f"[vcat]trim=duration={final_duration:.3f}[vout]")

    audio_labels = []
    first_audio_input = len(video_inputs)
    for offset, (_path, start, gain_db) in enumerate(voice_inputs):
        input_index = first_audio_input + offset
        delay_ms = max(0, int(round(start * 1000)))
        label = f"a{offset}"
        audio_labels.append(f"[{label}]")
        filters.append(
            f"[{input_index}:a]asetpts=PTS-STARTPTS,adelay={delay_ms}:all=1,"
            f"volume={gain_db:.2f}dB[{label}]"
        )
    filters.append(
        f"{''.join(audio_labels)}amix=inputs={len(voice_inputs)}:duration=longest:normalize=0,"
        f"alimiter=limit=0.95,atrim=duration={final_duration:.3f}[aout]"
    )

    final_path.parent.mkdir(parents=True, exist_ok=True)
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[vout]",
            "-map",
            "[aout]",
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
            str(final_path),
        ]
    )
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RenderError(f"FFmpeg assembly failed with exit code {exc.returncode}.") from exc
    print(f"Created final video: {final_path}")
    return final_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if not 1 <= args.max_in_flight <= 5:
            raise RenderError("--max-in-flight must be between 1 and 5.")
        if args.poll_seconds < 3:
            raise RenderError("--poll-seconds must be at least 3.")
        if args.timeout_minutes <= 0:
            raise RenderError("--timeout-minutes must be greater than zero.")

        storyboard_path = args.storyboard.expanduser().resolve()
        storyboard = read_storyboard(storyboard_path)
        validate_neutral_storyboard(storyboard)
        manifest = compile_runway_manifest(
            storyboard,
            video_model=args.video_model,
            voice_model=args.voice_model,
            voice=args.voice,
            ratio_override=args.ratio,
        )
        jobs = validate_manifest(manifest)
        estimate = estimate_cost(jobs)
        print_estimate(manifest, estimate)

        if args.export_manifest:
            exported = export_manifest(args.export_manifest, manifest)
            print(f"Compiled Runway manifest exported for inspection: {exported}")

        output_dir = (
            args.output_dir or default_output_dir(storyboard_path)
        ).expanduser().resolve()
        if not args.submit:
            print("\nDry run only: no Runway API request was made and no credits were spent.")
            print(
                "To submit after reviewing the estimate:\n"
                f"  python3 {Path(__file__).name} {storyboard_path.name} "
                f"--submit --max-cost-usd {math.ceil(estimate.usd)}"
            )
            return 0

        if args.max_cost_usd is None:
            raise RenderError("--max-cost-usd is required with --submit.")
        if args.max_cost_usd <= 0:
            raise RenderError("--max-cost-usd must be greater than zero.")
        if estimate.usd > args.max_cost_usd + 1e-9:
            raise RenderError(
                f"Estimated cost ${estimate.usd:.2f} exceeds your ${args.max_cost_usd:.2f} spending ceiling."
            )

        validate_key()
        RunwayML, _TaskFailedError, _TaskTimeoutError = import_runway()
        client = RunwayML()
        try:
            compiled = json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            manifest_hash = hashlib.sha256(compiled).hexdigest()
            state_path = output_dir / "runway-execution.json"
            state = load_or_create_state(state_path, manifest_hash, jobs, estimate)
            new_job_count = sum(
                1 for entry in state["jobs"].values() if entry.get("status") == "NOT_SUBMITTED"
            )
            remaining_job_ids = {
                job_id
                for job_id, entry in state["jobs"].items()
                if entry.get("status") == "NOT_SUBMITTED"
            }
            remaining_credits = sum(
                line.credits for line in estimate.lines if line.job_id in remaining_job_ids
            )
            try:
                organization = client.organization.retrieve()
                balance = int(organization.credit_balance)
            except Exception as exc:
                raise RenderError(f"Could not verify the Runway organization credit balance: {exc}") from exc
            confirm_submission(
                estimate,
                new_job_count,
                remaining_credits,
                balance,
                args.yes,
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            write_state(state_path, state)
            complete = process_jobs(
                client,
                jobs,
                state,
                state_path,
                output_dir,
                args.max_in_flight,
                args.poll_seconds,
                args.timeout_minutes,
            )
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()

        if not complete:
            raise RenderError(
                f"One or more tasks failed. Successful assets were preserved in {output_dir}. "
                "Failed jobs are not automatically retried because retries can incur additional charges."
            )

        if not args.skip_assembly:
            project = require_mapping(manifest.get("project"), "project")
            final_path = output_dir / safe_title(project.get("title"))
            assemble_final_video(manifest, state, output_dir, final_path, args.overwrite_final)
        else:
            print(f"All Runway assets downloaded to: {output_dir / 'assets'}")
        return 0
    except RenderError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nStopped. Saved task IDs can be resumed with the same command.", file=sys.stderr)
        return 130
    except OSError as exc:
        print(f"Error reading or writing a file: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
