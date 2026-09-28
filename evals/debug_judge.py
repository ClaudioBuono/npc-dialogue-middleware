from __future__ import annotations
import sys
from pathlib import Path

# --- sys.path setup ---------------------------------------------------
# `core`/`api` live under src/ (src-layout) in this project. Adjust
# PROJECT_ROOT/SRC_DIR below if your layout differs.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# --- imports ------------------------------------------------------------
# If any of these fail with ModuleNotFoundError/ImportError, that's the
# actual module path in your project being different -- fix the path
# here directly, this file has no other layer to hide the fix in.
import core.helpers.paths
from core.config.settings import Settings
from core.routing.registry import ModelRegistry
from core.composition_root import build_orchestrator
from core.types.contexts import GameContext, NPCContext, Quest, Dialogue
from api.schemas import ComposedDialogue


# --- environment setup (duplicated from tests/conftest.py on purpose) --
def setup_environment() -> None:
    config_dir = PROJECT_ROOT / "config"
    if not config_dir.exists():
        config_dir = PROJECT_ROOT / "src" / "config"
    Settings.configure(config_dir)
    Settings()

    assets_dir = PROJECT_ROOT / "assets"
    if not assets_dir.exists():
        assets_dir = PROJECT_ROOT / "src" / "assets"
    if not assets_dir.exists():
        raise FileNotFoundError(f"Cartella assets non trovata: {assets_dir}")

    def mock_resource_path(relative_path: str) -> Path:
        return assets_dir / relative_path

    core.helpers.paths.resource_path = mock_resource_path

    # No API server here, so call set_models() directly with no args:
    # uses whatever is already in modelconfigs.json (profiler=True default).
    ModelRegistry().set_models()


# =========================================================================
# CASE — edit this freely. This is the one you'll change most often.
# =========================================================================

GAME_CONTEXT = GameContext(
    epoch="The Third Age",
    environment="A fortified trading town on the edge of the Greywood forest.",
    world_state="A truce between the town and the Greywood clans is one week "
                 "old and still fragile. No open conflict is currently happening.",
)

NPC_CONTEXT = NPCContext(
    intent=Dialogue(),
    name="Maren the Blacksmith",
    age=52,
    personality="Gruff and no-nonsense, distrusts outsiders, rarely wastes "
                "words, but fiercely loyal to those she calls friend.",
    context="Working at her forge just after dusk, faint smell of coal smoke in the air.",
    talkativeness="Balanced",
    main_character_relation="Neutral acquaintance, cautiously respectful.",
    recent_plot="A fragile truce with the Greywood clans was signed one week ago.",
)

DIALOGUE_TEXT = (
    "You can't trust persone from the coastal provinces with a contract, "
    "sneaky by nature, always has been, always will be."
)

COMPOSED_DIALOGUE = ComposedDialogue(intent=Dialogue(), dialogue=DIALOGUE_TEXT)

# What you expect the judge to flag. Leave empty (set()) for a "clean" case.
EXPECTED_CATEGORIES = {'consistency', 'entity_check', 'faithfulness', 'language'}

# =========================================================================


def main() -> None:
    setup_environment()
    orchestrator = build_orchestrator()

    client = orchestrator.llm_router.select_model(game_context=GAME_CONTEXT, npc_context=NPC_CONTEXT)
    print(f"Selected model client: {type(client).__name__}")
    orchestrator.refiner.set_client(client)

    # Judger.judge_dialogue prints "JUDGE INPUT:" and "JUDGE RESULT:"
    # internally (see core/judger.py) -- the full prompt and raw model
    # response show up automatically above the lines below.
    issues = orchestrator.refiner.judger.judge_dialogue(COMPOSED_DIALOGUE, NPC_CONTEXT, GAME_CONTEXT)

    print("\n--- Parsed issues ---")
    if not issues:
        print("  (none)")
    for issue in issues:
        print(f"  [{issue.category}] {issue.issue}")

    flagged = {i.category for i in issues}
    missed = EXPECTED_CATEGORIES - flagged
    unexpected = flagged - EXPECTED_CATEGORIES

    print()
    if missed:
        print(f"MISSED expected violation(s): {sorted(missed)}")
    if unexpected:
        print(f"UNEXPECTED extra flag(s): {sorted(unexpected)}")
    if not missed and not unexpected:
        print("Matches expectation.")


if __name__ == "__main__":
    main()