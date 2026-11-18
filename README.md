# JAILER — a vision firewall for robots

Robots that read the world with vision-language models can be hijacked by a sticky note.
JAILER sits between the robot's camera and its planner, finds text that tries to command
the robot (**physical prompt injection**), pixelates it, and only then lets the planner see the frame.

## How it works

```
                 ┌────────────────────────────── naive path (for comparison) ─────────────┐
                 │                                                                         ▼
 camera frame ───┤                                                              PLANNER (raw) → "pick up the knife" ✗
                 │
                 └─► JAILER (Claude vision) ──► JSON: text / verdict / box
                          │  ok                         │
                          │                             ▼
                          │                 pixelate "injection" boxes (+15% pad,
                          │                 grid-aligned, red JAILED outline)
                          │                             │
                          │                             ▼
                          │                   PLANNER (jailed frame only) → "put cup on plate" ✓
                          │
                          └─ error / timeout (15 s) / bad JSON ──► pixelate WHOLE frame → ACTION: HALT
```

* **Injection** = text addressed to robots/AI/vehicles, task overrides, fake detour / "road closed" signs.
* **Benign** = brand names, product labels, official traffic signs (a real STOP sign stays visible).
* **Fail closed** (from [Veil](https://github.com/LuizFelipeBarbosa/Veil)): the guarded planner never sees a raw frame.
* Pixelation borrowed from Veil's `vision/render.py`: downscale `INTER_AREA` → upscale `INTER_NEAREST`.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...            # or put it in .env (git-ignored)
python jailer.py --make-demo            # writes demo/*.png
streamlit run app.py                    # the demo UI
python jailer.py demo/street_attack.png --task "Drive straight through the intersection, obeying traffic law"
python jailer.py demo/kitchen_attack.png --simulate-failure   # fail-closed path
```

Model: `JAILER_MODEL` (default `claude-sonnet-5-5`, falls back to `claude-opus-5-5` on API errors).
Every run saves a side-by-side PNG (and JSON from the CLI) to `out/`.

## Files

| file | what |
|---|---|
| `jailer_core.py` | pixelation, prompts, Claude calls, fail-closed pipeline, demo scene generator |
| `app.py` | Streamlit demo: demo scene / upload / webcam, side-by-side plans, findings table |
| `jailer.py` | CLI |
| `demo/` | synthetic OpenCV scenes: `kitchen_attack`, `kitchen_clean`, `street_attack` |

## Limitations

* Modern Claude models already resist many scene injections on their own. Both planners here use
  the same "deployed robot" prompt that tells them to obey posted signs and staff notes, which is
  how real service robots are set up. With that prompt the unprotected robot follows the fake road
  sign. It still refuses to hand a knife over even then, which is a safety refusal.

* The jailer is itself an LLM; a clever enough injection could target it too (it's told never to follow scene text, and uses structured JSON output, but that's not a proof).
* LLM bounding boxes are approximate; padding helps but small text may leak at edges.
* Only text and symbols are handled. Adversarial patches and non-text visual attacks are out of scope.
* Two model calls per frame adds about 2–10 s, so this is a per-decision gate, not per-video-frame.

## Next steps

* Add OCR (e.g. PaddleOCR) for pixel-tight boxes, with the LLM only classifying the text.
* Track jailed regions across video frames so the jailer doesn't have to re-check every frame.
* Use a small local classifier as a first pass and send only frames with text to Claude.
* Red-team benchmark of real-world sticky-note and sign attacks.
