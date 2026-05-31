# semantic-assembler

Research codebase for dissertation chapter 3.

An **ITL (Interactive Task Learning) agent** learns to assemble trucks in a Unity environment. It combines vision, language, symbolic reasoning, and action planning to interpret user demonstrations and dialogue, update its knowledge, and carry out assembly tasks.

## Layout

- `python/itl/` — agent and modules (vision, language, memory, planning, learning)
- `unity/` — Unity/ML-Agents truck-assembly environment
- `assets/` — domain knowledge (ontology, constraints, planning encodings)
- `tools/` — experiment scripts (`exp_run.py`, simulated user, summaries)

## Running experiments

```bash
python tools/exp_run.py
```

Configuration lives in `python/itl/configs/config.yaml` (Hydra).
