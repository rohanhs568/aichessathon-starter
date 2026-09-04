"""Run a local arena and save every game as PGN."""

import argparse
import io
from pathlib import Path

import chess
import chess.pgn

from harness.referee import FAILED_TERMINATIONS, play_match
from harness.rules import PLY_CAP
from harness.sandbox import local

FAST_BASE_MS = 10_000
FAST_INCREMENT_MS = 100


def labelled_pgn(
    raw_pgn: str,
    round_number: int,
    white_name: str,
    black_name: str,
) -> str:
    game = chess.pgn.read_game(io.StringIO(raw_pgn))
    if game is None:
        return raw_pgn

    game.headers["Event"] = "AI Chessathon local arena"
    game.headers["Round"] = str(round_number)
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    return str(game)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score an agent over several games and save the PGNs."
    )
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--opponent", type=Path, default=Path("baselines/greedy"))
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--base-ms", type=int, default=FAST_BASE_MS)
    parser.add_argument("--increment-ms", type=int, default=FAST_INCREMENT_MS)
    parser.add_argument("--ply-cap", type=int, default=PLY_CAP)
    parser.add_argument("--start-fen", default=chess.STARTING_FEN)
    parser.add_argument("--pgn-out", type=Path, default=Path("arena_games.pgn"))
    arguments = parser.parse_args()

    agent = arguments.agent.resolve()
    opponent = arguments.opponent.resolve()

    wins = draws = losses = 0
    terminations: dict[str, int] = {}
    saved_games: list[str] = []

    agent_name = arguments.agent.as_posix()
    opponent_name = arguments.opponent.as_posix()

    for game_number in range(1, arguments.games + 1):
        plays_white = game_number % 2 == 1
        white_path, black_path = (
            (agent, opponent) if plays_white else (opponent, agent)
        )
        white_name, black_name = (
            (agent_name, opponent_name)
            if plays_white
            else (opponent_name, agent_name)
        )

        outcome = play_match(
            local(white_path),
            local(black_path),
            arguments.base_ms,
            arguments.increment_ms,
            ply_cap=arguments.ply_cap,
            start_fen=arguments.start_fen,
        )

        saved_games.append(
            labelled_pgn(
                outcome.pgn,
                game_number,
                white_name,
                black_name,
            )
        )

        terminations[outcome.termination] = (
            terminations.get(outcome.termination, 0) + 1
        )

        if outcome.result in {"draw", "void"}:
            draws += 1
        elif (outcome.result == "white") == plays_white:
            wins += 1
        else:
            losses += 1

        print(
            f"game {game_number}/{arguments.games}: "
            f"{outcome.result} by {outcome.termination}"
        )

    arguments.pgn_out.parent.mkdir(parents=True, exist_ok=True)
    arguments.pgn_out.write_text(
        "\n\n".join(saved_games) + "\n",
        encoding="utf-8",
    )

    score = (wins + draws / 2) / arguments.games

    print(
        f"\n{arguments.agent} vs {arguments.opponent} "
        f"over {arguments.games} games"
    )
    print(f"+{wins} ={draws} -{losses}, score {score:.1%}")
    print(
        "terminations: "
        + ", ".join(
            f"{name} {count}"
            for name, count in terminations.items()
        )
    )
    print(f"PGNs saved to: {arguments.pgn_out}")

    broken = {
        name: count
        for name, count in terminations.items()
        if name in FAILED_TERMINATIONS
    }
    if broken:
        raise SystemExit(
            "your agent failed to finish a game: "
            + ", ".join(
                f"{name} {count}"
                for name, count in broken.items()
            )
        )


if __name__ == "__main__":
    main()
