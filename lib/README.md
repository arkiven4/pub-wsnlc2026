# Numerical implementation

See [architecture.md](../docs/architecture.md) for the function/module map.
`study_cleaning.py` fits and writes decisions; `study_evaluation.py` reads those
decisions and evaluates raw frozen features. `qwen_extractor.py` and
`hear_extractor.py` contain encoder code. `helper.py`, `confident_learning.py`
and `utils.py` retain the paper numerical algorithms.
