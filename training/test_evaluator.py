from pathlib import Path

import chess
import torch

from train_evaluator import Evaluator, encode_fen


MODEL_PATH = Path("training/models/evaluator_debug_64.pt")


def load_model(path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    model = Evaluator(
        hidden_size=checkpoint["hidden_size"]
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    model.eval()

    return model, checkpoint["scale"]


def predict(model, scale, board):
    """
    Return the model evaluation in centipawns.

    The model itself predicts from the side-to-move perspective.
    We also convert it to White's perspective for easier comparison.
    """

    fen = board.fen()

    features = encode_fen(fen)

    x = torch.tensor(
        features,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():
        prediction = model(x).item()

    stm_cp = prediction * scale

    white_cp = (
        stm_cp
        if board.turn == chess.WHITE
        else -stm_cp
    )

    return stm_cp, white_cp


def show_position(name, board, model, scale):
    stm_cp, white_cp = predict(
        model,
        scale,
        board,
    )

    print()
    print(name)
    print("-" * len(name))
    print(board)
    print()
    print(f"Side to move: {'White' if board.turn else 'Black'}")
    print(f"STM evaluation:   {stm_cp:+.1f} cp")
    print(f"White evaluation: {white_cp:+.1f} cp")

    return white_cp


def main():
    model, scale = load_model(MODEL_PATH)

    print("Loaded model:")
    print(MODEL_PATH)

    # --------------------------------------------------
    # 1. Starting position
    # --------------------------------------------------

    start = chess.Board()

    start_eval = show_position(
        "Starting position",
        start,
        model,
        scale,
    )

    # --------------------------------------------------
    # 2. Compare Nf3 and Nh3
    # --------------------------------------------------

    nf3 = chess.Board()
    nf3.push_uci("g1f3")

    nh3 = chess.Board()
    nh3.push_uci("g1h3")

    nf3_eval = show_position(
        "After 1. Nf3",
        nf3,
        model,
        scale,
    )

    nh3_eval = show_position(
        "After 1. Nh3",
        nh3,
        model,
        scale,
    )

    # --------------------------------------------------
    # 3. Black queen missing
    # --------------------------------------------------

    black_queen_missing = chess.Board(
        "rnb1kbnr/pppppppp/8/8/8/8/"
        "PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    )

    black_queen_missing_eval = show_position(
        "Black queen missing",
        black_queen_missing,
        model,
        scale,
    )

    # --------------------------------------------------
    # 4. White queen missing
    # --------------------------------------------------

    white_queen_missing = chess.Board(
        "rnbqkbnr/pppppppp/8/8/8/8/"
        "PPPPPPPP/RNB1KBNR w KQkq - 0 1"
    )

    white_queen_missing_eval = show_position(
        "White queen missing",
        white_queen_missing,
        model,
        scale,
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print()
    print("=" * 50)
    print("SANITY CHECK SUMMARY")
    print("=" * 50)

    print(
        f"Starting position:   "
        f"{start_eval:+.1f} cp"
    )

    print(
        f"1. Nf3:              "
        f"{nf3_eval:+.1f} cp"
    )

    print(
        f"1. Nh3:              "
        f"{nh3_eval:+.1f} cp"
    )

    print(
        f"Black queen missing: "
        f"{black_queen_missing_eval:+.1f} cp"
    )

    print(
        f"White queen missing: "
        f"{white_queen_missing_eval:+.1f} cp"
    )

    print()

    # Human-readable checks rather than hard assertions.
    #
    # This is a tiny debug model, so if it fails one of these
    # we want to see the failure rather than crash the test.

    print("Expected relationships:")

    print(
        "Black queen missing > 0:",
        "PASS"
        if black_queen_missing_eval > 0
        else "FAIL"
    )

    print(
        "White queen missing < 0:",
        "PASS"
        if white_queen_missing_eval < 0
        else "FAIL"
    )

    print(
        "Nf3 preferred to Nh3:",
        "PASS"
        if nf3_eval > nh3_eval
        else "FAIL",
    )


if __name__ == "__main__":
    main()