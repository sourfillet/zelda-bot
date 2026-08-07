# Zelda (NES) RAM Map — verified for this project

Addresses used in `data.json`, cross-referenced against the community map
(https://datacrystal.tcrf.net/wiki/The_Legend_of_Zelda/RAM_map) and verified
empirically with `games/Zelda/ram_search.py` (random agent + offline analysis).

Status legend: ✅ verified in a recording · 📖 community-documented, not yet
exercised here · 🔬 discovered/confirmed empirically, not on DataCrystal.

## Core state

| Address | Hex | Variable | Notes | Status |
|---|---|---|---|---|
| 16 | $10 | Level | 0 = overworld, 1–9 = dungeon | ✅ |
| 18 | $12 | Game Mode | 5=normal, 6=prepare scroll, 7=scrolling, 4=finish scroll; anything else = death/game-over/menus. Mode goes 5→17→8 when Link dies | ✅ |
| 21 | $15 | Backend Frame Count | wraps at 256 | ✅ |
| 112 | $70 | Link X | object slot 0 of the X table $70–$7B | ✅ |
| 132 | $84 | Link Y | object slot 0 of the Y table $84–$8F | ✅ |
| 224 | $E0 | Game Paused | 1 = paused | 📖 |
| 232 | $E8 | Screen Scroll Direction | 1=E 2=W 4=S 8=N while scrolling; **can idle at 0xFF in some save states**, so test membership in {1,2,4,8}, not `!= 0` | ✅ |
| 235 | $EB | Room | map location, x + 0x10*y | ✅ |

## Combat (the interesting part)

| Address | Hex | Variable | Notes | Status |
|---|---|---|---|---|
| 80 | $50 | — (removed) | kills *without taking damage*; resets at 10 and on every hit. **Was wrongly used as "Enemies Killed"** | ✅ |
| 846 | $34E | Enemies Spawned In Room | spawn count — does **not** decrease on kills | ✅ |
| 847 | $34F | Enemies Killed Current Room | increments per kill, resets on room change | ✅ |
| 848–859 | $350–$35B | enemy type, slot 0–11 | zeroed when the slot's enemy dies; slots are reused for drops/effects | ✅ |
| 1158–1163 | $486–$48B | Enemy 1–6 HP | high nibble = hit points; a wooden-sword hit subtracts 0x10. Slot i aligns with type $350+i, X $71+i, Y $85+i | 🔬 |
| 1322 | $52A | Enemies Killed | lifetime kill counter — precision 1.00 / recall 1.00 against observed kills. **The reward signal** | ✅ |
| 1575 | $627 | killed count (current screen) | per DataCrystal; did *not* react to kills in the `monsters` room — do not rely on it | ✅ |

## Link's resources

| Address | Hex | Variable | Notes | Status |
|---|---|---|---|---|
| 1584 | $630 | Deaths | save-slot-1 death counter. **Old value 630 was a decimal/hex bug** ($276 is scratch memory that churns randomly) | ✅ |
| 1647 | $66F | Heart Containers | low nibble = filled hearts, high nibble = containers − 1 | ✅ |
| 1648 | $670 | Hearts | partial heart only: 0=empty, 0x01–0x7F=half, 0x80–0xFF=full. Combine with $66F via `adapter.get_actual_hearts` | ✅ |
| 1645 | $66D | Rupees | | ✅ |
| 1646 | $66E | Keys | | ✅ |
| 1649 | $671 | Triforce Pieces | bitfield, one bit per piece | 📖 |
| 1639 | $667 | Compass | bitfield, one bit per dungeon | 📖 |
| 1640 | $668 | Map | bitfield, one bit per dungeon | 📖 |

Compass and Map are used by the item reward but have **never been observed
changing** here — every save state is a fresh dungeon entry and all ten hold 0,
and 16k frames of random play on level1 left them constant. That constancy at
least bounds the risk: a wrong address fails silently rather than paying out
noise. To verify, collect level 1's map (room 118) and check $668 becomes
non-zero.

Inventory bytes ($657–$676: Sword, Bombs, Arrow, Bow, Candle, Flute, Food,
Potion, Magical Rod, Raft, Magic Book, Ring, Ladder, Magical Key, Power
Bracelet, Letter, Clock, Boomerang, Magical Boomerang, Shield) all match
DataCrystal and read correctly.

## Removed from data.json

- **Map Scroll LR ($FD) / Map Scroll UD ($FC)** — these are the *pause-menu
  subscreen* scroll registers, not room scrolling. Pressing START moved them,
  which the old loop misread as a room change. Room transitions are now
  detected via Game Mode ∈ {4, 6, 7}.

## Mapping more addresses

```bash
# 1. record RAM while a random agent plays (no START/SELECT, sword-heavy policy)
python games/Zelda/ram_search.py record --state monsters --frames 8000

# 2. survey everything that changes, annotated with the community map
python games/Zelda/ram_search.py analyze ram_dumps/monsters.npz
python games/Zelda/ram_search.py analyze ram_dumps/monsters.npz --kind counter

# 3. cheat-search: which addresses change exactly when an event fires?
#    (event = inc/dec/chg of a known address, e.g. a kill = $34F incrementing)
python games/Zelda/ram_search.py correlate ram_dumps/monsters.npz --event inc:0x34F

# 4. sanity-check data.json against a recording + the community map
python games/Zelda/ram_search.py verify ram_dumps/monsters.npz
```

`verify` also flags decimal/hex confusion (the class of bug that broke
`Deaths`): if an address isn't in the community map but its digits read as hex
*are*, it's reported as a suspect.
