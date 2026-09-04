"""Play a terminal chess game against the repository's agent.py."""

import argparse
import time
from pathlib import Path

import chess
import chess.pgn

import agent


def parse_human_move(board: chess.Board, text: str) -> chess.Move | None:
    text = text.strip()

    try:
        return board.parse_san(text)
    except ValueError:
        pass

    try:
        move = chess.Move.from_uci(text)
    except chess.InvalidMoveError:
        return None

    return move if move in board.legal_moves else None


def save_pgn(
    board: chess.Board,
    human_color: chess.Color,
    destination: Path,
) -> None:
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "Human vs Chessathon bot"
    game.headers["White"] = "Human" if human_color == chess.WHITE else "Chessathon bot"
    game.headers["Black"] = "Chessathon bot" if human_color == chess.WHITE else "Human"

    destination.write_text(str(game) + "\n", encoding="utf-8")
    print(f"PGN saved to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play against agent.py")
    parser.add_argument(
        "--color",
        choices=("white", "black"),
        default="white",
        help="Your colour.",
    )
    parser.add_argument(
        "--bot-time-ms",
        type=int,
        default=120_000,
        help="Bot starting clock in milliseconds.",
    )
    parser.add_argument(
        "--increment-ms",
        type=int,
        default=500,
        help="Bot increment per move in milliseconds.",
    )
    parser.add_argument(
        "--pgn-out",
        type=Path,
        default=Path("human_vs_bot.pgn"),
    )
    arguments = parser.parse_args()

    human_color = (
        chess.WHITE if arguments.color == "white" else chess.BLACK
    )
    board = chess.Board()
    bot_clock = float(arguments.bot_time_ms)

    print()
    print("Enter moves as SAN (e4, Nf3, O-O, Qxd5+) or UCI (e2e4).")
    print("Type 'resign' or 'quit' to stop.")
    print()

    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            print(board)
            print()
            print(f"Game over: {outcome.result()} ({outcome.termination.name})")
            break

        print(board)
        print()

        if board.turn == human_color:
            while True:
                text = input("Your move: ").strip()

                if text.lower() in {"quit", "exit"}:
                    save_pgn(board, human_color, arguments.pgn_out)
                    return

                if text.lower() == "resign":
                    print("You resigned.")
                    save_pgn(board, human_color, arguments.pgn_out)
                    return

                move = parse_human_move(board, text)
                if move is None:
                    print("Illegal/unrecognised move. Try SAN or UCI.")
                    continue

                board.push(move)
                print()
                break

        else:
            print(
                f"Bot thinking... "
                f"(clock {bot_clock / 1000.0:.1f}s)"
            )

            started = time.monotonic()
            uci = agent.get_move(board.fen(), max(1, int(bot_clock)))
            elapsed_ms = (time.monotonic() - started) * 1000.0

            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                raise RuntimeError(f"agent returned illegal move: {uci}")

            san = board.san(move)
            board.push(move)

            bot_clock -= elapsed_ms
            bot_clock += arguments.increment_ms

            print(
                f"Bot plays {san} ({uci}) "
                f"in {elapsed_ms / 1000.0:.2f}s"
            )
            print()

            if bot_clock <= 0:
                print("Bot flagged.")
                break

    save_pgn(board, human_color, arguments.pgn_out)


if __name__ == "__main__":
    main()
