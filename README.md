# AniMind story-to-animation pipeline

AniMind now separates the story from the rendering service:

```text
lanternfly.txt
      |
      v
story_board_maker.py
      |
      v
renderer_neutral_sb.json
      |
      +------------------------+
      |                        |
      v                        v
runway_adapter.py       blender_adapter.py (next phase)
      |
      v
Runway assets + FFmpeg final MP4
```

## Files

- `story_board_maker.py` reads a natural-language story and creates the
  renderer-neutral storyboard.
- `renderer_neutral_sb.json` is the current lanternfly example. It contains
  entities, locations, actions, camera direction, narration, timing, and
  continuity. It contains no Runway jobs or API fields.
- `storyboard.schema.json` documents the neutral JSON contract.
- `runway_adapter.py` reads the neutral storyboard, creates Runway requests in
  memory, estimates cost, optionally submits them, downloads the results, and
  asks FFmpeg to assemble the final MP4.
- `lanternfly.txt` is the example source story.

`runway_anim_sb.json` is no longer required. The adapter can export an
equivalent diagnostic file when you explicitly request one.

## 1. Create the neutral storyboard

Use the richer OpenAI planner:

```bash
python3 story_board_maker.py lanternfly.txt -o renderer_neutral_sb.json --planner openai
```

This needs `OPENAI_API_KEY` in the same Terminal window. It uses OpenAI only to
plan the storyboard; it does not contact Runway.

Or use the free local planner:

```bash
python3 story_board_maker.py lanternfly.txt -o renderer_neutral_sb.json --planner local
```

## 2. Preview the Runway work and cost

```bash
python3 runway_adapter.py renderer_neutral_sb.json
```

This is a dry run. It does not need a Runway key, make an API request, or spend
credits.

To inspect the generated Runway-specific JSON without making it part of the
normal workflow:

```bash
python3 runway_adapter.py renderer_neutral_sb.json --export-manifest runway_preview.json
```

## 3. Submit to Runway only after reviewing the estimate

```bash
python3 runway_adapter.py renderer_neutral_sb.json --submit --max-cost-usd 5
```

Submission requires `RUNWAYML_API_SECRET`. The adapter checks the credit
balance and asks for `RUN` confirmation before submitting new paid jobs. Use
`--yes` only when you deliberately want to skip that typed confirmation.

The default Runway output folder is `outputs/runway`. It contains downloaded
assets, the assembled MP4, and `runway-execution.json`. Keep that execution
file: it records task IDs so an interrupted run can resume without creating
duplicate paid jobs. If the folder already contains execution state for an
older storyboard, choose a new folder, for example:

```bash
python3 runway_adapter.py renderer_neutral_sb.json \
  --output-dir outputs/runway-neutral \
  --submit --max-cost-usd 5
```

## Optional packages

For OpenAI storyboard planning:

```bash
python3 -m pip install --upgrade openai "pydantic>=2"
```

For paid Runway submission and secure downloads:

```bash
python3 -m pip install --upgrade runwayml httpx certifi
```

For final assembly on macOS:

```bash
brew install ffmpeg
```

## Next phase

`blender_adapter.py` will read the same `renderer_neutral_sb.json`, map its
entities and actions to reusable Blender assets and behaviors, render each
scene locally, and assemble the output under `outputs/blender`.
