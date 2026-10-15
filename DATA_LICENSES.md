# Data and third-party licenses

The Apache License 2.0 in `LICENSE` covers the **source code** of this project only.
The data and libraries listed below are covered by their own licenses.

## Hurtlex (modified, included in this repository)

- Location: `src/assets/`
- Source: https://github.com/valeriobasile/hurtlex
- Authors: Elisa Bassignana, Valerio Basile, Viviana Patti
- License: **Creative Commons Attribution-NonCommercial-ShareAlike 4.0
  International (CC BY-NC-SA 4.0)**
  https://creativecommons.org/licenses/by-nc-sa/4.0/
- **Changes made:** the lexicon was trimmed and modified for this project
  (see `src/assets/README.md` for details).
- The modified files are distributed under the **same license (CC BY-NC-SA 4.0)**.
- **Non-commercial use only** applies to this data. Using this project with
  Hurtlex for commercial purposes is not permitted by the Hurtlex license.
  To use the project commercially, replace the lexicon with one you are
  allowed to use commercially.

Citation:

> Elisa Bassignana, Valerio Basile, Viviana Patti.
> "Hurtlex: A Multi-lingual Lexicon of Words to Hurt."
> Proceedings of the Fifth Italian Conference on Computational Linguistics
> (CLiC-it 2018).

## wordfreq (runtime dependency, not redistributed)

- Installed from PyPI via `requirements.txt`; no files are included here.
- Author: Robyn Speer.
- Code: Apache License 2.0. Data: CC BY-SA 4.0, plus the sources listed in
  wordfreq's own `NOTICE.md`.
- Upstream: https://github.com/rspeer/wordfreq
- If you use wordfreq in academic work, cite it as described in its README.

## Python dependencies

Other dependencies (FastAPI, Starlette, Uvicorn, Pydantic, OpenAI SDK, httpx,
NumPy, pandas, PyYAML, nvidia-ml-py, etc.) are installed from PyPI and are
distributed under their own licenses (mostly MIT, BSD or Apache 2.0).
PyInstaller (GPL-2.0 with a bootloader exception) is used only as a build tool.