#!/usr/bin/env python3
"""Convert a natural-language story into a renderer-neutral storyboard.

The output describes story structure, entities, locations, actions, camera
direction, narration, timing, and continuity.  It deliberately contains no
Runway, Blender, API, pricing, job, endpoint, or vendor-specific fields.

Examples
--------
Create a storyboard with the OpenAI planner when OPENAI_API_KEY is set::

    python3 story_board_maker.py lanternfly.txt -o renderer_neutral_sb.json

Use the dependency-free local planner::

    python3 story_board_maker.py lanternfly.txt -o renderer_neutral_sb.json --planner local

Read a story from the clipboard on macOS::

    pbpaste | python3 story_board_maker.py - -o renderer_neutral_sb.json

The OpenAI planner is optional and requires::

    python3 -m pip install --upgrade openai "pydantic>=2"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


SCHEMA_VERSION = "1.0"
STORYBOARD_KIND = "animind.renderer-neutral-storyboard"
DEFAULT_STYLE = (
    "cinematic animated film, expressive characters, coherent production "
    "design, polished lighting, and natural motion"
)
FRAME_PRESETS = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
}
FORBIDDEN_PROVIDER_KEYS = {
    "animationJobId",
    "authorizationEnvironmentVariable",
    "baseUrl",
    "body",
    "creditCost",
    "endpoint",
    "jobId",
    "jobs",
    "method",
    "presetId",
    "promptText",
    "runway",
    "videoModel",
    "voiceJobId",
    "voiceModel",
    "voicePreset",
}


class StoryboardError(RuntimeError):
    """Raised when a story cannot be converted into a valid storyboard."""


@dataclass(frozen=True)
class SceneChunk:
    scene_id: str
    narration: str
    word_count: int
    duration_seconds: int


@dataclass(frozen=True)
class Entity:
    entity_id: str
    name: str
    entity_type: str
    visual_description: str
    continuity_notes: str


@dataclass(frozen=True)
class Location:
    location_id: str
    name: str
    visual_description: str
    continuity_notes: str


@dataclass(frozen=True)
class Action:
    actor_id: Optional[str]
    verb: str
    target_id: Optional[str]
    manner: str
    description: str


@dataclass(frozen=True)
class SceneVisual:
    scene_id: str
    visual_summary: str
    location_id: Optional[str]
    setting: str
    entity_ids: List[str]
    actions: List[Action]
    camera_shot: str
    camera_angle: str
    camera_movement: str
    camera_subject: str
    camera_description: str
    lighting: str
    mood: str
    continuity_notes: str
    transition: str


@dataclass(frozen=True)
class VisualPlan:
    title: str
    style_bible: str
    entities: List[Entity]
    locations: List[Location]
    scenes: List[SceneVisual]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a one-page story into a renderer-neutral JSON storyboard "
            "for use by adapters such as Runway or Blender."
        )
    )
    parser.add_argument(
        "story",
        help="UTF-8 text file containing the story, or '-' to read stdin",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output path (default: <story-name>.storyboard.json)",
    )
    parser.add_argument("--title", help="Override the inferred story title")
    parser.add_argument(
        "--style",
        default=DEFAULT_STYLE,
        help="Renderer-independent visual style used throughout the story",
    )
    parser.add_argument(
        "--planner",
        choices=("auto", "openai", "local"),
        default="auto",
        help=(
            "Storyboard planner: auto uses OpenAI when configured and otherwise "
            "uses the dependency-free local planner"
        ),
    )
    parser.add_argument(
        "--planner-model",
        default="gpt-5.6",
        help="OpenAI model used only for storyboard planning",
    )
    parser.add_argument(
        "--aspect-ratio",
        choices=tuple(FRAME_PRESETS),
        default="16:9",
        help="Storyboard frame shape (default: 16:9)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Intended timeline frame rate (default: 24)",
    )
    parser.add_argument(
        "--narration-wpm",
        type=int,
        default=145,
        help="Estimated narration speed used for timing (default: 145)",
    )
    parser.add_argument(
        "--target-scene-seconds",
        type=int,
        default=8,
        help="Preferred scene duration (default: 8)",
    )
    parser.add_argument(
        "--max-scene-seconds",
        type=int,
        default=20,
        help="Maximum storyboard scene duration before further chunking (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Base seed for repeatability; otherwise derived from the story",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation (default: 2)",
    )
    return parser.parse_args(argv)


def normalize_story(text: str) -> str:
    """Normalize whitespace while preserving paragraphs and sentences."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    paragraphs = [
        re.sub(r"\s+", " ", part).strip()
        for part in re.split(r"\n\s*\n", text)
    ]
    normalized = "\n\n".join(part for part in paragraphs if part)
    if not normalized:
        raise StoryboardError("The story is empty.")
    if word_count(normalized) < 3:
        raise StoryboardError(
            "The story is too short; provide at least one complete sentence."
        )
    return normalized


def read_story(value: str) -> Tuple[str, str]:
    if value == "-":
        return normalize_story(sys.stdin.read()), "stdin"
    path = Path(value).expanduser()
    if not path.is_file():
        raise StoryboardError(f"Story file not found: {path}")
    try:
        return normalize_story(path.read_text(encoding="utf-8")), str(path.resolve())
    except UnicodeDecodeError as exc:
        raise StoryboardError(f"Story file must be UTF-8 text: {path}") from exc


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE))


def infer_title(story: str) -> str:
    first_sentence = re.split(
        r"(?<=[.!?])\s+", story.replace("\n", " "), maxsplit=1
    )[0]
    words = re.findall(r"[\w'’-]+", first_sentence, flags=re.UNICODE)[:8]
    if not words:
        return "Untitled Story"
    title = " ".join(words)
    return title if len(title) <= 72 else title[:69].rstrip() + "..."


def _split_sentences(story: str) -> List[str]:
    sentences: List[str] = []
    for paragraph in story.split("\n\n"):
        parts = re.split(
            r"(?<=[.!?])(?:[\"'”’])?\s+(?=[\"'“‘(]*[A-Z0-9])",
            paragraph,
        )
        sentences.extend(part.strip() for part in parts if part.strip())
    return sentences


def _split_long_unit(text: str, max_words: int) -> List[str]:
    if word_count(text) <= max_words:
        return [text.strip()]
    clauses = [
        part.strip()
        for part in re.split(r"(?<=[,;:—–-])\s+", text)
        if part.strip()
    ]
    if len(clauses) > 1 and all(word_count(part) <= max_words for part in clauses):
        result: List[str] = []
        current = ""
        for clause in clauses:
            proposed = f"{current} {clause}".strip()
            if current and word_count(proposed) > max_words:
                result.append(current)
                current = clause
            else:
                current = proposed
        if current:
            result.append(current)
        return result
    tokens = text.split()
    return [
        " ".join(tokens[index:index + max_words])
        for index in range(0, len(tokens), max_words)
    ]


def _estimate_duration(text: str, narration_wpm: int) -> int:
    spoken = word_count(text) * 60.0 / narration_wpm
    punctuation_pause = 0.12 * len(re.findall(r"[,;:—–]", text))
    sentence_pause = 0.22 * len(re.findall(r"[.!?]", text))
    return max(2, int(math.ceil(spoken + punctuation_pause + sentence_pause + 0.5)))


def chunk_story(
    story: str,
    narration_wpm: int,
    target_scene_seconds: int,
    max_scene_seconds: int,
) -> List[SceneChunk]:
    if not 80 <= narration_wpm <= 220:
        raise StoryboardError("--narration-wpm must be between 80 and 220.")
    if not 2 <= target_scene_seconds <= 30:
        raise StoryboardError("--target-scene-seconds must be between 2 and 30.")
    if not 2 <= max_scene_seconds <= 120:
        raise StoryboardError("--max-scene-seconds must be between 2 and 120.")
    if target_scene_seconds > max_scene_seconds:
        raise StoryboardError(
            "--target-scene-seconds cannot exceed --max-scene-seconds."
        )

    words_per_second = narration_wpm / 60.0
    target_words = max(4, int((target_scene_seconds - 1.0) * words_per_second))
    max_words = max(target_words, int((max_scene_seconds - 1.0) * words_per_second))

    units: List[str] = []
    for sentence in _split_sentences(story):
        units.extend(_split_long_unit(sentence, max_words))

    grouped: List[str] = []
    current = ""
    for unit in units:
        proposed = f"{current} {unit}".strip()
        if current and word_count(proposed) > target_words:
            grouped.append(current)
            current = unit
        else:
            current = proposed
    if current:
        grouped.append(current)

    if len(grouped) > 1 and word_count(grouped[-1]) <= 4:
        proposed = f"{grouped[-2]} {grouped[-1]}"
        if (
            word_count(proposed) <= max_words
            and _estimate_duration(proposed, narration_wpm) <= max_scene_seconds
        ):
            grouped[-2:] = [proposed]

    chunks = [
        SceneChunk(
            scene_id=f"scene_{index:03d}",
            narration=text,
            word_count=word_count(text),
            duration_seconds=_estimate_duration(text, narration_wpm),
        )
        for index, text in enumerate(grouped, start=1)
    ]
    if not chunks:
        raise StoryboardError("No scenes could be extracted from the story.")
    return chunks


def _local_visual_plan(
    title: str,
    style: str,
    chunks: Sequence[SceneChunk],
) -> VisualPlan:
    camera_cycle = (
        ("wide", "eye-level", "slow push-in"),
        ("medium", "eye-level", "tracking"),
        ("close-up", "eye-level", "subtle parallax"),
        ("medium", "three-quarter", "gentle orbit"),
        ("wide", "high-angle", "lower toward eye-level"),
    )
    transitions = ("fade", "cut", "dissolve", "cut", "match-cut")
    scenes: List[SceneVisual] = []
    for index, chunk in enumerate(chunks):
        excerpt = chunk.narration.rstrip(".!?")
        shot, angle, movement = camera_cycle[index % len(camera_cycle)]
        action = Action(
            actor_id=None,
            verb="perform_story_beat",
            target_id=None,
            manner="as described by the narration",
            description=(
                f"Show the central subject visibly acting out this story beat: {excerpt}"
            ),
        )
        scenes.append(
            SceneVisual(
                scene_id=chunk.scene_id,
                visual_summary=f"An animated interpretation of: {excerpt}",
                location_id=None,
                setting="The location and period described or implied by the narration",
                entity_ids=[],
                actions=[action],
                camera_shot=shot,
                camera_angle=angle,
                camera_movement=movement,
                camera_subject="the central subject",
                camera_description=f"{shot} {movement} following the central action",
                lighting=(
                    "motivated cinematic lighting appropriate to the time, place, and mood"
                ),
                mood="emotionally faithful to this story beat",
                continuity_notes=(
                    "Preserve recurring characters, clothing, props, geography, and color palette"
                ),
                transition="fade-in" if index == 0 else transitions[index % len(transitions)],
            )
        )
    return VisualPlan(
        title=title,
        style_bible=style,
        entities=[],
        locations=[],
        scenes=scenes,
    )


def _openai_visual_plan(
    story: str,
    title: str,
    style: str,
    chunks: Sequence[SceneChunk],
    model: str,
) -> VisualPlan:
    try:
        from openai import OpenAI
        from pydantic import BaseModel, ConfigDict, Field
    except ImportError as exc:
        raise StoryboardError(
            "The OpenAI planner needs the 'openai' and 'pydantic' packages. "
            "Install them with: python3 -m pip install --upgrade openai 'pydantic>=2'"
        ) from exc

    if not os.environ.get("OPENAI_API_KEY"):
        raise StoryboardError("OPENAI_API_KEY is not set.")

    class EntityPlan(BaseModel):
        model_config = ConfigDict(extra="forbid")
        entity_id: str = Field(description="Stable lowercase identifier")
        name: str
        entity_type: str = Field(
            description="Plain category such as person, animal, creature, vehicle, or prop"
        )
        visual_description: str
        continuity_notes: str

    class LocationPlan(BaseModel):
        model_config = ConfigDict(extra="forbid")
        location_id: str = Field(description="Stable lowercase identifier")
        name: str
        visual_description: str
        continuity_notes: str

    class ActionPlan(BaseModel):
        model_config = ConfigDict(extra="forbid")
        actor_id: Optional[str] = Field(
            description="Existing entity_id, or null when the actor is not explicit"
        )
        verb: str = Field(description="Short physical action verb or verb phrase")
        target_id: Optional[str] = Field(
            description="Existing entity_id affected by the action, or null"
        )
        manner: str
        description: str = Field(description="Concrete visible action")

    class ScenePlan(BaseModel):
        model_config = ConfigDict(extra="forbid")
        scene_id: str
        visual_summary: str
        location_id: Optional[str]
        setting: str
        entity_ids: List[str]
        actions: List[ActionPlan]
        camera_shot: str
        camera_angle: str
        camera_movement: str
        camera_subject: str
        camera_description: str
        lighting: str
        mood: str
        continuity_notes: str
        transition: str

    class PlannerOutput(BaseModel):
        model_config = ConfigDict(extra="forbid")
        title: str
        style_bible: str
        entities: List[EntityPlan]
        locations: List[LocationPlan]
        scenes: List[ScenePlan]

    PlannerOutput.model_rebuild(
        _types_namespace={
            "List": List,
            "Optional": Optional,
            "EntityPlan": EntityPlan,
            "LocationPlan": LocationPlan,
            "ActionPlan": ActionPlan,
            "ScenePlan": ScenePlan,
        }
    )

    fixed_scenes = [
        {
            "scene_id": chunk.scene_id,
            "duration_seconds": chunk.duration_seconds,
            "narration": chunk.narration,
        }
        for chunk in chunks
    ]
    instructions = (
        "You are an animation storyboard planner. Produce a renderer-neutral plan, "
        "not instructions for any particular video service or 3D package. Return "
        "exactly one scene for every supplied scene_id in the same order. Catalog "
        "recurring people, creatures, vehicles, and important props as entities, "
        "and recurring places as locations. Use stable lowercase IDs. Every ID "
        "referenced by a scene or action must exist in those catalogs. Give each "
        "scene one or two concrete visible actions that fit its duration. Separate "
        "camera shot, angle, movement, subject, lighting, mood, and continuity. "
        "Do not rewrite or quote narration. Avoid captions, subtitles, logos, "
        "readable interface text, split screens, montage lists, sound-generation "
        "instructions, renderer names, model names, APIs, endpoints, or job fields. "
        "Keep every field concise and physically depictable."
    )
    payload = {
        "requested_title": title,
        "requested_visual_style": style,
        "full_story_for_context": story,
        "fixed_scenes": fixed_scenes,
    }
    try:
        response = OpenAI(timeout=120.0).responses.parse(
            model=model,
            instructions=instructions,
            input=json.dumps(payload, ensure_ascii=False),
            text_format=PlannerOutput,
            store=False,
        )
    except Exception as exc:
        raise StoryboardError(f"OpenAI storyboard planning failed: {exc}") from exc

    parsed = response.output_parsed
    if parsed is None:
        raise StoryboardError("OpenAI returned no structured storyboard plan.")
    expected_ids = [chunk.scene_id for chunk in chunks]
    returned_ids = [scene.scene_id for scene in parsed.scenes]
    if returned_ids != expected_ids:
        raise StoryboardError(
            "OpenAI returned scenes that do not match the fixed scene IDs."
        )

    return VisualPlan(
        title=parsed.title.strip() or title,
        style_bible=parsed.style_bible.strip() or style,
        entities=[
            Entity(
                entity_id=item.entity_id,
                name=item.name,
                entity_type=item.entity_type,
                visual_description=item.visual_description,
                continuity_notes=item.continuity_notes,
            )
            for item in parsed.entities
        ],
        locations=[
            Location(
                location_id=item.location_id,
                name=item.name,
                visual_description=item.visual_description,
                continuity_notes=item.continuity_notes,
            )
            for item in parsed.locations
        ],
        scenes=[
            SceneVisual(
                scene_id=item.scene_id,
                visual_summary=item.visual_summary,
                location_id=item.location_id,
                setting=item.setting,
                entity_ids=list(item.entity_ids),
                actions=[
                    Action(
                        actor_id=action.actor_id,
                        verb=action.verb,
                        target_id=action.target_id,
                        manner=action.manner,
                        description=action.description,
                    )
                    for action in item.actions
                ],
                camera_shot=item.camera_shot,
                camera_angle=item.camera_angle,
                camera_movement=item.camera_movement,
                camera_subject=item.camera_subject,
                camera_description=item.camera_description,
                lighting=item.lighting,
                mood=item.mood,
                continuity_notes=item.continuity_notes,
                transition=item.transition,
            )
            for item in parsed.scenes
        ],
    )


def choose_visual_plan(
    planner: str,
    story: str,
    title: str,
    style: str,
    chunks: Sequence[SceneChunk],
    planner_model: str,
) -> Tuple[VisualPlan, str, Optional[str]]:
    if planner == "local":
        return _local_visual_plan(title, style, chunks), "local", None
    if planner == "openai":
        return (
            _openai_visual_plan(story, title, style, chunks, planner_model),
            "openai",
            None,
        )
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return (
                _openai_visual_plan(story, title, style, chunks, planner_model),
                "openai",
                None,
            )
        except StoryboardError as exc:
            return (
                _local_visual_plan(title, style, chunks),
                "local",
                f"OpenAI planner unavailable; used local planner instead: {exc}",
            )
    return (
        _local_visual_plan(title, style, chunks),
        "local",
        "OPENAI_API_KEY was not set; used the local storyboard planner.",
    )


def derive_base_seed(story: str, requested_seed: Optional[int]) -> int:
    if requested_seed is not None:
        if not 0 <= requested_seed <= 4_294_967_295:
            raise StoryboardError("--seed must be between 0 and 4294967295.")
        return requested_seed
    return int.from_bytes(hashlib.sha256(story.encode("utf-8")).digest()[:4], "big")


def stable_seed(base_seed: int, scene_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{scene_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _transition_object(description: str, first: bool) -> Dict[str, str]:
    text = description.strip() or ("fade-in" if first else "cut")
    lowered = text.lower()
    if "match" in lowered and "cut" in lowered:
        transition_type = "match-cut"
    elif "dissolve" in lowered:
        transition_type = "dissolve"
    elif "fade" in lowered:
        transition_type = "fade"
    elif "cut" in lowered:
        transition_type = "cut"
    else:
        transition_type = "custom"
    return {"type": transition_type, "description": text}


def build_storyboard(
    *,
    story: str,
    source: str,
    title: str,
    style: str,
    chunks: Sequence[SceneChunk],
    visual_plan: VisualPlan,
    planner_used: str,
    planner_model: str,
    aspect_ratio: str,
    fps: int,
    narration_wpm: int,
    base_seed: int,
) -> Dict[str, Any]:
    visuals = {scene.scene_id: scene for scene in visual_plan.scenes}
    if set(visuals) != {chunk.scene_id for chunk in chunks}:
        raise StoryboardError(
            "The visual plan and narration chunks do not have matching scene IDs."
        )
    if not 12 <= fps <= 120:
        raise StoryboardError("--fps must be between 12 and 120.")
    width, height = FRAME_PRESETS[aspect_ratio]
    scenes: List[Dict[str, Any]] = []
    narration_track: List[Dict[str, Any]] = []
    cursor = 0.0

    for index, chunk in enumerate(chunks):
        visual = visuals[chunk.scene_id]
        start = round(cursor, 3)
        end = round(cursor + chunk.duration_seconds, 3)
        action_duration = chunk.duration_seconds / max(1, len(visual.actions))
        actions: List[Dict[str, Any]] = []
        action_cursor = 0.0
        for action_index, action in enumerate(visual.actions, start=1):
            remaining = chunk.duration_seconds - action_cursor
            duration = remaining if action_index == len(visual.actions) else action_duration
            actions.append(
                {
                    "actionId": f"{chunk.scene_id}_action_{action_index:02d}",
                    "actorId": action.actor_id,
                    "verb": action.verb,
                    "targetId": action.target_id,
                    "manner": action.manner,
                    "description": action.description,
                    "startOffsetSeconds": round(action_cursor, 3),
                    "durationSeconds": round(duration, 3),
                }
            )
            action_cursor += duration

        scenes.append(
            {
                "sceneId": chunk.scene_id,
                "sequence": index + 1,
                "startSeconds": start,
                "endSeconds": end,
                "durationSeconds": chunk.duration_seconds,
                "seed": stable_seed(base_seed, chunk.scene_id),
                "narration": {
                    "text": chunk.narration,
                    "wordCount": chunk.word_count,
                    "startOffsetSeconds": 0.25,
                },
                "visual": {
                    "summary": visual.visual_summary,
                    "locationId": visual.location_id,
                    "settingDescription": visual.setting,
                    "entityIds": visual.entity_ids,
                    "actions": actions,
                    "camera": {
                        "shotType": visual.camera_shot,
                        "angle": visual.camera_angle,
                        "movement": visual.camera_movement,
                        "subject": visual.camera_subject,
                        "description": visual.camera_description,
                    },
                    "look": {
                        "lighting": visual.lighting,
                        "mood": visual.mood,
                    },
                    "continuityNotes": visual.continuity_notes,
                },
                "transitionIn": _transition_object(
                    visual.transition,
                    first=index == 0,
                ),
            }
        )
        narration_track.append(
            {
                "sceneId": chunk.scene_id,
                "startSeconds": round(start + 0.25, 3),
                "gainDb": 0,
            }
        )
        cursor = end

    storyboard: Dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": STORYBOARD_KIND,
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": {
            "storyPath": source,
            "wordCount": word_count(story),
            "narrationMode": "faithful-chunking",
            "storyText": story,
        },
        "project": {
            "title": visual_plan.title or title,
            "visualStyle": visual_plan.style_bible or style,
            "frame": {
                "width": width,
                "height": height,
                "aspectRatio": aspect_ratio,
                "framesPerSecond": fps,
            },
            "estimatedDurationSeconds": round(cursor, 3),
            "sceneCount": len(chunks),
            "narrationWordsPerMinute": narration_wpm,
            "baseSeed": base_seed,
        },
        "planner": {
            "type": planner_used,
            "model": planner_model if planner_used == "openai" else None,
        },
        "entities": [
            {
                "entityId": item.entity_id,
                "name": item.name,
                "entityType": item.entity_type,
                "visualDescription": item.visual_description,
                "continuityNotes": item.continuity_notes,
            }
            for item in visual_plan.entities
        ],
        "locations": [
            {
                "locationId": item.location_id,
                "name": item.name,
                "visualDescription": item.visual_description,
                "continuityNotes": item.continuity_notes,
            }
            for item in visual_plan.locations
        ],
        "scenes": scenes,
        "timeline": {
            "sceneOrder": [chunk.scene_id for chunk in chunks],
            "narrationTrack": narration_track,
            "finalDurationSeconds": round(cursor, 3),
        },
    }
    validate_storyboard(storyboard)
    return storyboard


def _walk_keys(value: Any, path: str = "root") -> List[Tuple[str, str]]:
    found: List[Tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            found.append((key, f"{path}.{key}"))
            found.extend(_walk_keys(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk_keys(child, f"{path}[{index}]"))
    return found


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StoryboardError(f"{label} must be a non-empty string.")
    return value


def validate_storyboard(storyboard: Dict[str, Any]) -> None:
    if storyboard.get("kind") != STORYBOARD_KIND:
        raise StoryboardError(f"kind must be {STORYBOARD_KIND!r}.")
    if storyboard.get("schemaVersion") != SCHEMA_VERSION:
        raise StoryboardError(f"schemaVersion must be {SCHEMA_VERSION!r}.")

    forbidden = [
        path for key, path in _walk_keys(storyboard) if key in FORBIDDEN_PROVIDER_KEYS
    ]
    if forbidden:
        raise StoryboardError(
            "Renderer-neutral storyboard contains provider-specific fields: "
            + ", ".join(forbidden[:5])
        )

    project = storyboard.get("project")
    if not isinstance(project, dict):
        raise StoryboardError("project must be an object.")
    _require_text(project.get("title"), "project.title")
    _require_text(project.get("visualStyle"), "project.visualStyle")
    frame = project.get("frame")
    if not isinstance(frame, dict):
        raise StoryboardError("project.frame must be an object.")
    for key in ("width", "height", "framesPerSecond"):
        if not isinstance(frame.get(key), int) or isinstance(frame.get(key), bool):
            raise StoryboardError(f"project.frame.{key} must be an integer.")

    entities = storyboard.get("entities")
    locations = storyboard.get("locations")
    if not isinstance(entities, list) or not isinstance(locations, list):
        raise StoryboardError("entities and locations must be arrays.")
    entity_ids = set()
    for index, entity in enumerate(entities):
        if not isinstance(entity, dict):
            raise StoryboardError(f"entities[{index}] must be an object.")
        entity_id = _require_text(entity.get("entityId"), f"entities[{index}].entityId")
        if entity_id in entity_ids:
            raise StoryboardError(f"Duplicate entityId: {entity_id}")
        entity_ids.add(entity_id)
    location_ids = set()
    for index, location in enumerate(locations):
        if not isinstance(location, dict):
            raise StoryboardError(f"locations[{index}] must be an object.")
        location_id = _require_text(
            location.get("locationId"), f"locations[{index}].locationId"
        )
        if location_id in location_ids:
            raise StoryboardError(f"Duplicate locationId: {location_id}")
        location_ids.add(location_id)

    scenes = storyboard.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise StoryboardError("scenes must be a non-empty array.")
    scene_ids = set()
    expected_start = 0.0
    for index, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            raise StoryboardError(f"scenes[{index}] must be an object.")
        scene_id = _require_text(scene.get("sceneId"), f"scenes[{index}].sceneId")
        if scene_id in scene_ids:
            raise StoryboardError(f"Duplicate sceneId: {scene_id}")
        scene_ids.add(scene_id)
        if scene.get("sequence") != index + 1:
            raise StoryboardError("Scene sequence values must start at 1 and be consecutive.")
        duration = scene.get("durationSeconds")
        start = scene.get("startSeconds")
        end = scene.get("endSeconds")
        if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration <= 0:
            raise StoryboardError(f"{scene_id}.durationSeconds must be positive.")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise StoryboardError(f"{scene_id} must have numeric start and end times.")
        if abs(float(start) - expected_start) > 0.001:
            raise StoryboardError(f"{scene_id} does not begin at the preceding scene's end.")
        if abs(float(end) - (float(start) + float(duration))) > 0.001:
            raise StoryboardError(f"{scene_id} has inconsistent duration and end time.")
        expected_start = float(end)

        narration = scene.get("narration")
        visual = scene.get("visual")
        if not isinstance(narration, dict) or not isinstance(visual, dict):
            raise StoryboardError(f"{scene_id} must contain narration and visual objects.")
        _require_text(narration.get("text"), f"{scene_id}.narration.text")
        location_id = visual.get("locationId")
        if location_id is not None and location_id not in location_ids:
            raise StoryboardError(f"{scene_id} references unknown locationId {location_id!r}.")
        referenced_entities = visual.get("entityIds")
        if not isinstance(referenced_entities, list):
            raise StoryboardError(f"{scene_id}.visual.entityIds must be an array.")
        missing = sorted(set(referenced_entities) - entity_ids)
        if missing:
            raise StoryboardError(f"{scene_id} references unknown entities: {missing}")
        actions = visual.get("actions")
        if not isinstance(actions, list) or not actions:
            raise StoryboardError(f"{scene_id} must contain at least one action.")
        action_ids = set()
        for action_index, action in enumerate(actions):
            if not isinstance(action, dict):
                raise StoryboardError(
                    f"{scene_id}.visual.actions[{action_index}] must be an object."
                )
            action_id = _require_text(
                action.get("actionId"),
                f"{scene_id}.visual.actions[{action_index}].actionId",
            )
            if action_id in action_ids:
                raise StoryboardError(f"Duplicate actionId in {scene_id}: {action_id}")
            action_ids.add(action_id)
            _require_text(action.get("verb"), f"{action_id}.verb")
            _require_text(action.get("description"), f"{action_id}.description")
            for reference_key in ("actorId", "targetId"):
                reference = action.get(reference_key)
                if reference is not None and reference not in entity_ids:
                    raise StoryboardError(
                        f"{action_id}.{reference_key} references unknown entity {reference!r}."
                    )

    timeline = storyboard.get("timeline")
    if not isinstance(timeline, dict):
        raise StoryboardError("timeline must be an object.")
    if timeline.get("sceneOrder") != [scene["sceneId"] for scene in scenes]:
        raise StoryboardError("timeline.sceneOrder must match the scene array.")
    final_duration = timeline.get("finalDurationSeconds")
    if not isinstance(final_duration, (int, float)) or abs(float(final_duration) - expected_start) > 0.001:
        raise StoryboardError("timeline.finalDurationSeconds is inconsistent with the scenes.")


def default_output_path(story_arg: str) -> Path:
    if story_arg == "-":
        return Path("story.storyboard.json")
    source = Path(story_arg)
    return source.with_name(f"{source.stem}.storyboard.json")


def write_json(path: Path, storyboard: Dict[str, Any], indent: int) -> None:
    if not 0 <= indent <= 8:
        raise StoryboardError("--indent must be between 0 and 8.")
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            storyboard,
            ensure_ascii=False,
            indent=indent if indent else None,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        story, source = read_story(args.story)
        title = (args.title or infer_title(story)).strip()
        if not title:
            raise StoryboardError("The title cannot be empty.")
        chunks = chunk_story(
            story,
            args.narration_wpm,
            args.target_scene_seconds,
            args.max_scene_seconds,
        )
        visual_plan, planner_used, warning = choose_visual_plan(
            args.planner,
            story,
            title,
            args.style,
            chunks,
            args.planner_model,
        )
        storyboard = build_storyboard(
            story=story,
            source=source,
            title=title,
            style=args.style,
            chunks=chunks,
            visual_plan=visual_plan,
            planner_used=planner_used,
            planner_model=args.planner_model,
            aspect_ratio=args.aspect_ratio,
            fps=args.fps,
            narration_wpm=args.narration_wpm,
            base_seed=derive_base_seed(story, args.seed),
        )
        output = (args.output or default_output_path(args.story)).resolve()
        write_json(output, storyboard, args.indent)
    except StoryboardError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"Error reading or writing a file: {exc}", file=sys.stderr)
        return 2

    if warning:
        print(f"Warning: {warning}", file=sys.stderr)
    print(
        f"Created {output} with {len(chunks)} scenes and an estimated duration "
        f"of {storyboard['project']['estimatedDurationSeconds']} seconds."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
