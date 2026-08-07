# Importing mjlab triggers its one-time task auto-discovery, which loads
# javis.tasks -> javis.velocity_task -> javis.robot_constants. If that chain
# instead got triggered *from inside* javis.robot_constants's own top-level
# `import mjlab...` (e.g. `python -c "from javis.robot_constants import
# ..."`), robot_constants would be re-entered while only partially
# initialized -> ImportError for names defined later in that file. Importing
# mjlab here first, before any javis submodule runs, avoids the cycle: this
# runs to completion before Python proceeds to import whichever javis
# submodule was actually requested.
import mjlab  # noqa: F401
