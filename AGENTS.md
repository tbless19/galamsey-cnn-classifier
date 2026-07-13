# AGENTS.md

## Cursor Cloud specific instructions

This repo is a single-notebook Python data-science project: **Galamsey image
classification with a CNN (TensorFlow/Keras) + Apache Spark (PySpark)**. All
logic lives in `final.ipynb`. There is no web app, API, database, or secrets.

### Environment
- Dependencies are installed into the **system Python 3.12** with
  `pip install --break-system-packages ...` (PEP 668 is enforced, so the flag is
  required). The update script refreshes them; `python3` and the Jupyter kernel
  (`/usr/bin/python3`) both see them — no virtualenv activation needed.
- Installed stack: `numpy matplotlib pillow scikit-learn seaborn pyspark
  tensorflow-cpu jupyterlab notebook nbconvert nbclient ipykernel`.
- PySpark runs in local mode and needs a JDK. Java 21 is preinstalled and works
  with the installed PySpark 4.x. No Spark cluster is required.

### Running the application (Jupyter)
- The `jupyter` console scripts install to `~/.local/bin`, which is **not on
  PATH** by default. Launch via the module instead:
  `python3 -m jupyterlab --ip=0.0.0.0 --port=8888 --no-browser --ServerApp.token=""`
- Run the notebook headless with:
  `python3 -m nbconvert --to notebook --execute final.ipynb --output /tmp/executed.ipynb`

### Known caveat: the committed `final.ipynb` does not run as-is
These are pre-existing **application** bugs (not environment problems). Do not
"fix" them unless explicitly asked:
- Cell 2 `Config.create_directories()` has an empty `for` body → `IndentationError`.
- `Config.BASE_DIR = "galamsey"` makes it look for images under
  `galamsey/data/yes` and `galamsey/data/no`, but the real images live at
  `data/yes` and `data/no` (100 images each).

To smoke-test the toolchain end to end without touching the notebook, exercise
the same stages directly on the real data: PIL 128×128 patch extraction from
`data/yes` + `data/no` → `SparkSession` → the notebook's CNN architecture →
`model.fit` / evaluate. Model/plot outputs are written under `output/`
(git-ignored).
