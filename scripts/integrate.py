import retro

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
        print(SCRIPT_DIR)
        retro.data.Integrations.add_custom_path(SCRIPT_DIR)
        print("Zelda" in retro.data.list_games(inttype=retro.data.Integrations.ALL))
        env = retro.make("Zelda", inttype=retro.data.Integrations.ALL, render_mode="rgb_array")
        print(env)

if __name__ == "__main__":
        main()