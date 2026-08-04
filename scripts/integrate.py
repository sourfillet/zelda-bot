"""Smoke-test that a game under games/ is visible to stable-retro.

Usage:
    python scripts/integrate.py [game]

Defaults to Zelda. Exits non-zero if the integration is not found, which
usually means the ROM is missing from games/<game>/ or its SHA1 does not
match the one in games/<game>/rom.sha.
"""

import os
import sys

import retro

# The integration path is the games/ directory, not this script's directory.
GAMES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "games")


def main():
    game = sys.argv[1] if len(sys.argv) > 1 else "Zelda"
    print("Games path:", GAMES_DIR)
    retro.data.Integrations.add_custom_path(GAMES_DIR)

    available = retro.data.list_games(inttype=retro.data.Integrations.ALL)
    if game not in available:
        print(f"{game} not found. Is games/{game}/rom.nes present?")
        return 1

    env = retro.make(game, inttype=retro.data.Integrations.ALL, render_mode="rgb_array")
    print(f"{game} integrated:", env)
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
