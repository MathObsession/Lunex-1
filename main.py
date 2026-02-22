import chess
import math
import threading
import tkinter as tk
from tkinter import messagebox
import sys
import os
import requests
import time
from PIL import Image, ImageTk
from io import BytesIO

try:
    import torch
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
except ImportError:
    torch = None
    DEVICE = None

UNICODE_PIECES = {
    chess.PAWN:   {True: "♙", False: "♟"},
    chess.KNIGHT: {True: "♘", False: "♞"},
    chess.BISHOP: {True: "♗", False: "♝"},
    chess.ROOK:   {True: "♖", False: "♜"},
    chess.QUEEN:  {True: "♕", False: "♛"},
    chess.KING:   {True: "♔", False: "♚"},
}


def piece_to_unicode(piece):
    if piece is None:
        return ""
    return UNICODE_PIECES[piece.piece_type][piece.color]

OPENINGS_BY_LEVEL = {
    1: [],
    2: [("Italian Game", ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"])],
    3: [("Caro-Kann Defense", ["e2e4", "c7c6", "d2d4", "d7d5"]),
        ("Queen's Gambit", ["d2d4", "d7d5", "c2c4"])],
    4: [("King's Indian Defense", ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6"]),
        ("Sicilian Defense", ["e2e4", "c7c5", "g1f3", "d7d6"])]
}

class Evaluator:

    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000,
    }

    _tensor_values = None

    @staticmethod
    def evaluate(board: chess.Board) -> float:
        if board.is_checkmate():
            if board.turn:
                return -math.inf
            else:
                return math.inf
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0.0

        score = 0.0

        if DEVICE is not None and DEVICE.type == "mps":
            keys = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
            if Evaluator._tensor_values is None:
                vals = [Evaluator.PIECE_VALUES[k] for k in keys]
                Evaluator._tensor_values = torch.tensor(vals, device=DEVICE, dtype=torch.float32)

            counts_w = [len(board.pieces(p, chess.WHITE)) for p in keys]
            counts_b = [len(board.pieces(p, chess.BLACK)) for p in keys]
                
            w_t = torch.tensor(counts_w, device=DEVICE, dtype=torch.float32)
            b_t = torch.tensor(counts_b, device=DEVICE, dtype=torch.float32)
            
            w_score = torch.dot(w_t, Evaluator._tensor_values).item()
            b_score = torch.dot(b_t, Evaluator._tensor_values).item()
            score += (w_score - b_score)
        else:
            for piece_type, value in Evaluator.PIECE_VALUES.items():
                score += len(board.pieces(piece_type, chess.WHITE)) * value
                score -= len(board.pieces(piece_type, chess.BLACK)) * value

        try:
            mobility = len(list(board.legal_moves))
        except Exception:
            mobility = 0
        if board.turn == chess.WHITE:
            score += 0.1 * mobility
        else:
            score -= 0.1 * mobility

        return score

class ChessAI:

    def __init__(self, depth: int = 3):
        self.depth = depth

    def minimax(self, board: chess.Board, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        if depth == 0 or board.is_game_over():
            return Evaluator.evaluate(board)

        if maximizing:
            max_eval = -math.inf
            for move in board.legal_moves:
                board.push(move)
                val = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                if val > max_eval:
                    max_eval = val
                if val > alpha:
                    alpha = val
                if beta <= alpha:
                    break  # prune
            return max_eval
        else:
            min_eval = math.inf
            for move in board.legal_moves:
                board.push(move)
                val = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                if val < min_eval:
                    min_eval = val
                if val < beta:
                    beta = val
                if beta <= alpha:
                    break  
            return min_eval

    def get_opening_move(self, board: chess.Board):
        history = [move.uci() for move in board.move_stack]
        valid_openings = []
        for level, ops in OPENINGS_BY_LEVEL.items():
            if level <= self.depth:
                valid_openings.extend(ops)
        
        for name, moves in valid_openings:
            if len(moves) > len(history):
                if moves[:len(history)] == history:
                    try:
                        return chess.Move.from_uci(moves[len(history)])
                    except ValueError:
                        pass
        return None

    def find_best_move(self, board: chess.Board):
        op_move = self.get_opening_move(board)
        if op_move and op_move in board.legal_moves:
            return op_move

        best_move = None
        if board.turn == chess.WHITE:
            best_value = -math.inf
            for move in board.legal_moves:
                board.push(move)
                val = self.minimax(board, self.depth - 1, -math.inf, math.inf, False)
                board.pop()
                if val > best_value:
                    best_value = val
                    best_move = move
        else:
            best_value = math.inf
            for move in board.legal_moves:
                board.push(move)
                val = self.minimax(board, self.depth - 1, -math.inf, math.inf, True)
                board.pop()
                if val < best_value:
                    best_value = val
                    best_move = move
        return best_move

def rounded_rect(canvas, x1, y1, x2, y2, r, **kwargs):
    points = [
        x1+r, y1, x1+r, y1, x2-r, y1, x2-r, y1, x2, y1,
        x2, y1+r, x2, y1+r, x2, y2-r, x2, y2-r, x2, y2,
        x2-r, y2, x2-r, y2, x1+r, y2, x1+r, y2, x1, y2,
        x1, y2-r, x1, y2-r, x1, y1+r, x1, y1+r, x1, y1
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)

class RoundedButton(tk.Canvas):
    def __init__(self, parent, text, command=None, radius=20, bg="#2D2D2D", fg="white", hover_bg="#3D3D3D", font=("Helvetica", 14, "bold"), width=160, height=45, **kwargs):
        super().__init__(parent, bg="#121212", highlightthickness=0, width=width, height=height, **kwargs)
        self.command = command
        self.radius = radius
        self.bg_color = bg
        self.hover_bg = hover_bg
        self.fg_color = fg
        self.text = text
        self.font = font
        self.bind("<Configure>", self._draw)
        self.bind("<Button-1>", self._on_click)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def _draw(self, event=None):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w > 1 and h > 1:
            rounded_rect(self, 0, 0, w, h, self.radius, fill=self.bg_color, tags="bg")
            self.create_text(w/2, h/2, text=self.text, fill=self.fg_color, font=self.font, tags="text")

    def _on_click(self, event):
        if self.command:
            self.command()

    def _on_enter(self, event):
        self.itemconfig("bg", fill=self.hover_bg)

    def _on_leave(self, event):
        self.itemconfig("bg", fill=self.bg_color)

class ModernChessGUI:

    THEMES = {
        "dark": {
            "BG_COLOR": "#121212",
            "PANEL_BG": "#202020",
            "TEXT_COLOR": "white",
            "LIGHT_SQ": "#EBECD0",
            "DARK_SQ": "#739552",
            "SELECT": "#F4F680",
            "MOVE_DOT": "#312E2B",
            "EVAL_BG": "#2C2C2C",
            "EVAL_TEXT": "#E8E8E8",
            "WIDGET_BG": "#2D2D2D"
        },
        "light": {
            "BG_COLOR": "#F5F5F5",
            "PANEL_BG": "#FFFFFF",
            "TEXT_COLOR": "#121212",
            "LIGHT_SQ": "#ECEED4",
            "DARK_SQ": "#769656",
            "SELECT": "#F4F680",
            "MOVE_DOT": "#000000",
            "EVAL_BG": "#DDDDDD",
            "EVAL_TEXT": "#222222",
            "WIDGET_BG": "#E0E0E0"
        }
    }
    
    CAPTURE_DOT_COLOR = "#E94A42" 
    ATTACK_DOT_COLOR = "#E94A42"
    AI_MOVE_HIGHLIGHT = "#F8F991" 
    DEFAULT_FONT_FAMILY = "Helvetica"
    MIN_SQUARE = 64

    def __init__(self, ai_depth: int = 3, square_size: int = 96):
        self.board = chess.Board()
        self.ai = ChessAI(depth=ai_depth)
        
        self.theme_name = "dark"
        self.theme = self.THEMES[self.theme_name]

        # Player score tracking (start 400, min 100, +8 per win)
        self.player_score = 400

        # GUI state
        self.selected_square = None  
        self.legal_targets = [] 
        self.attack_targets = [] 
        self.ai_last_move = None  
        
        self.piece_images = {} # raw images
        self.piece_tk_images = {} # tk images

        self._load_piece_images()

        self.root = tk.Tk()
        self.root.title("Modern ChessBot")
        self.root.configure(bg=self.theme["BG_COLOR"])
        self.container = tk.Frame(self.root, padx=20, pady=20, bg=self.theme["BG_COLOR"])
        self.container.pack(fill="both", expand=True)

        self.eval_frame = tk.Frame(self.container, width=30, bg=self.theme["BG_COLOR"])
        self.eval_frame.pack(side="left", fill="y", padx=(0, 20))
        self.eval_canvas = tk.Canvas(self.eval_frame, width=30, bg=self.theme["BG_COLOR"], highlightthickness=0)
        self.eval_canvas.pack(fill="both", expand=True)
        self.eval_canvas.bind("<Configure>", lambda e: self._update_eval_bar())

        # Container for rounded board effect
        self.board_frame = tk.Frame(self.container, bg=self.theme["BG_COLOR"])
        self.board_frame.pack(side="left", expand=True, fill="both")
        self.canvas = tk.Canvas(self.board_frame, bg=self.theme["BG_COLOR"], highlightthickness=0)
        self.canvas.pack(expand=True, fill="both")

        self.right_panel = tk.Frame(self.container, width=260, bg=self.theme["BG_COLOR"])
        self.right_panel.pack(side="right", fill="y", padx=(30, 0))
        self._build_controls()

        self.square_size = max(self.MIN_SQUARE, square_size)
        self.board_px = self.square_size * 8

        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Button-1>", self._on_click)

        self.square_items = {} 
        self.piece_items = {}  
        self.move_markers = []  
        self.select_border = None
        self.ai_highlight_items = [] 

        self._redraw_board()
        self.root.minsize(self.board_px + 260, self.board_px + 16)

    def _load_piece_images(self):
        os.makedirs("pieces", exist_ok=True)
        pieces = ["wp", "wn", "wb", "wr", "wq", "wk", "bp", "bn", "bb", "br", "bq", "bk"]
        for p in pieces:
            path = f"pieces/{p}.png"
            if not os.path.exists(path):
                print(f"Downloading {p}...")
                url = f"https://images.chesscomfiles.com/chess-themes/pieces/neo/150/{p}.png"
                try:
                    r = requests.get(url)
                    with open(path, "wb") as f:
                        f.write(r.content)
                except Exception as e:
                    print("Failed to download", p, e)
            try:
                self.piece_images[p] = Image.open(path).convert("RGBA")
            except Exception:
                pass
                
    def _resize_tk_images(self):
        sz = int(self.square_size * 0.9)
        for p, img in self.piece_images.items():
            resized = img.resize((sz, sz), Image.Resampling.LANCZOS)
            self.piece_tk_images[p] = ImageTk.PhotoImage(resized)

    def _toggle_theme(self):
        self.theme_name = "light" if self.theme_name == "dark" else "dark"
        self.theme = self.THEMES[self.theme_name]
        self.root.configure(bg=self.theme["BG_COLOR"])
        self.container.configure(bg=self.theme["BG_COLOR"])
        self.eval_frame.configure(bg=self.theme["BG_COLOR"])
        self.eval_canvas.configure(bg=self.theme["BG_COLOR"])
        self.board_frame.configure(bg=self.theme["BG_COLOR"])
        self.canvas.configure(bg=self.theme["BG_COLOR"])
        self.right_panel.configure(bg=self.theme["BG_COLOR"])
        self.title_lbl.configure(bg=self.theme["BG_COLOR"], fg=self.theme["TEXT_COLOR"])
        self.depth_f.configure(bg=self.theme["BG_COLOR"])
        self.depth_lbl.configure(bg=self.theme["BG_COLOR"], fg=self.theme["TEXT_COLOR"])
        self.depth_s.configure(bg=self.theme["WIDGET_BG"], fg=self.theme["TEXT_COLOR"])
        self.depth_s["menu"].configure(bg=self.theme["WIDGET_BG"], fg=self.theme["TEXT_COLOR"])
        self.status_frame.configure(bg=self.theme["PANEL_BG"])
        self.status_label.configure(bg=self.theme["PANEL_BG"], fg=self.theme["TEXT_COLOR"])
        self.gap_lbl.configure(bg=self.theme["BG_COLOR"])
        self.score_title_lbl.configure(bg=self.theme["BG_COLOR"], fg=self.theme["TEXT_COLOR"])
        self.score_display_lbl.configure(bg=self.theme["BG_COLOR"])
        self.score_info_lbl.configure(bg=self.theme["BG_COLOR"])
        
        self.btn_undo.bg_color = self.theme["WIDGET_BG"]
        self.btn_undo.fg_color = self.theme["TEXT_COLOR"]
        self.btn_undo._draw()
        self.btn_reset.bg_color = self.theme["WIDGET_BG"]
        self.btn_reset.fg_color = self.theme["TEXT_COLOR"]
        self.btn_reset._draw()
        self.btn_theme.bg_color = self.theme["WIDGET_BG"]
        self.btn_theme.fg_color = self.theme["TEXT_COLOR"]
        self.btn_theme._draw()
        self._redraw_board()

    def _build_controls(self):
        self.title_lbl = tk.Label(self.right_panel, text="Modern Chess", fg=self.theme["TEXT_COLOR"], bg=self.theme["BG_COLOR"], font=(self.DEFAULT_FONT_FAMILY, 24, "bold"))
        self.title_lbl.pack(pady=(20, 30))

        self.btn_undo = RoundedButton(self.right_panel, text="Undo Move", command=self._undo, bg=self.theme["WIDGET_BG"], fg=self.theme["TEXT_COLOR"])
        self.btn_undo.pack(pady=8)

        self.btn_reset = RoundedButton(self.right_panel, text="Reset Game", command=self._reset_board, bg=self.theme["WIDGET_BG"], fg=self.theme["TEXT_COLOR"])
        self.btn_reset.pack(pady=8)
        
        self.btn_theme = RoundedButton(self.right_panel, text="Toggle Theme", command=self._toggle_theme, bg=self.theme["WIDGET_BG"], fg=self.theme["TEXT_COLOR"])
        self.btn_theme.pack(pady=8)

        self.depth_map = {
            "Beginner": 1,
            "Intermediate": 2,
            "Master": 3,
            "Grandmaster": 4
        }
        initial_diff = "Master"
        for k, v in self.depth_map.items():
            if v == self.ai.depth:
                initial_diff = k

        self.depth_var = tk.StringVar(value=initial_diff)
        self.depth_f = tk.Frame(self.right_panel, bg=self.theme["BG_COLOR"])
        self.depth_f.pack(pady=15)
        self.depth_lbl = tk.Label(self.depth_f, text="Difficulty:", fg=self.theme["TEXT_COLOR"], bg=self.theme["BG_COLOR"], font=(self.DEFAULT_FONT_FAMILY, 14))
        self.depth_lbl.pack(side="left", padx=(0, 10))
        self.depth_s = tk.OptionMenu(self.depth_f, self.depth_var, *self.depth_map.keys(), command=self._change_depth)
        self.depth_s.config(font=(self.DEFAULT_FONT_FAMILY, 12), bg=self.theme["WIDGET_BG"], fg=self.theme["TEXT_COLOR"], highlightthickness=0, relief="flat", indicatoron=0)
        self.depth_s["menu"].config(font=(self.DEFAULT_FONT_FAMILY, 12), bg=self.theme["WIDGET_BG"], fg=self.theme["TEXT_COLOR"])
        self.depth_s.pack(side="left")

        self.status_frame = tk.Frame(self.right_panel, bg=self.theme["PANEL_BG"], padx=15, pady=15)
        self.status_frame.pack(pady=(20, 0), fill="x", padx=10)
        
        self.status_label = tk.Label(self.status_frame, text="White to move", fg=self.theme["TEXT_COLOR"], bg=self.theme["PANEL_BG"], font=(self.DEFAULT_FONT_FAMILY, 14, "bold"), wraplength=200, justify="center")
        self.status_label.pack()

        # ─── Player Score Display ───
        score_sep = tk.Frame(self.right_panel, height=1, bg="#444444")
        score_sep.pack(fill="x", padx=15, pady=(20, 10))

        self.score_title_lbl = tk.Label(self.right_panel, text="Rating",
                                         font=(self.DEFAULT_FONT_FAMILY, 12, "bold"),
                                         bg=self.theme["BG_COLOR"],
                                         fg=self.theme["TEXT_COLOR"])
        self.score_title_lbl.pack()

        self.score_display_lbl = tk.Label(self.right_panel,
                                           text=str(self.player_score),
                                           font=(self.DEFAULT_FONT_FAMILY, 28, "bold"),
                                           bg=self.theme["BG_COLOR"],
                                           fg="#26C2A3")
        self.score_display_lbl.pack(pady=(2, 0))

        self.score_info_lbl = tk.Label(self.right_panel,
                                        text="Win +8 pts  •  Min 100",
                                        font=(self.DEFAULT_FONT_FAMILY, 9),
                                        bg=self.theme["BG_COLOR"],
                                        fg="#888888")
        self.score_info_lbl.pack(pady=(0, 5))

        self.gap_lbl = tk.Label(self.right_panel, text="", bg=self.theme["BG_COLOR"])
        self.gap_lbl.pack(expand=True, fill="both")

    def _change_depth(self, *args):
        try:
            val = self.depth_var.get()
            self.ai.depth = self.depth_map.get(val, 3)
        except Exception:
            pass

    def _on_resize(self, event):
        size = min(event.width, event.height)
        size = max(size, self.MIN_SQUARE * 8)
        self.square_size = size // 8
        self.board_px = self.square_size * 8
        self.canvas.config(width=self.board_px, height=self.board_px)
        self._resize_tk_images()
        self._redraw_board()

    def _square_coords(self, square_index):
        file = chess.square_file(square_index)
        rank = chess.square_rank(square_index)
        row = 7 - rank
        col = file
        x0 = col * self.square_size
        y0 = row * self.square_size
        x1 = x0 + self.square_size
        y1 = y0 + self.square_size
        return x0, y0, x1, y1

    def _redraw_board(self):
        self.canvas.delete("all")
        self.square_items.clear()
        self.piece_items.clear()
        self.move_markers.clear()
        self.select_border = None
        self.ai_highlight_items.clear()

        # Check if resize is needed (in case started right away)
        if not self.piece_tk_images:
            self._resize_tk_images()

        # Board background framing behind squares
        rounded_rect(self.canvas, 0, 0, self.board_px, self.board_px, 15, fill=self.theme["PANEL_BG"])
        
        # We will draw squares with a small gap to make them elegantly rounded
        gap = int(self.square_size * 0.05)
        for sq in range(64):
            x0, y0, x1, y1 = self._square_coords(sq)
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            color = self.theme["LIGHT_SQ"] if (file + rank) % 2 != 0 else self.theme["DARK_SQ"]
            
            # Using rounded_rect to draw a square with rounding corners
            rect = rounded_rect(self.canvas, x0+gap, y0+gap, x1-gap, y1-gap, 8, fill=color, outline="")
            self.square_items[sq] = rect

        if self.ai_last_move:
            self._draw_ai_move_highlight(self.ai_last_move)

        for sq in range(64):
            piece = self.board.piece_at(sq)
            if piece:
                self._draw_piece_on_square(sq, piece)

        if self.selected_square is not None:
            self._show_selection_and_moves(self.selected_square, self.legal_targets, self.attack_targets)

        if hasattr(self, 'eval_canvas'):
            self._update_eval_bar()

    def _update_eval_bar(self):
        height = self.eval_canvas.winfo_height()
        if height <= 1:
            height = self.board_px

        score = Evaluator.evaluate(self.board)
        if score == math.inf:
            capped = 1000
            display_str = "M"
        elif score == -math.inf:
            capped = -1000
            display_str = "M"
        else:
            capped = max(-1000, min(1000, score))
            display_str = f"{abs(score)/100:.1f}"

        ratio = (capped + 1000) / 2000.0
        black_h = height * (1 - ratio)

        self.eval_canvas.delete("all")
        bg_c = self.theme["BG_COLOR"]
        
        # Draw full rounded rect for the dark background
        rounded_rect(self.eval_canvas, 0, 0, 30, height, 15, fill=self.theme["EVAL_BG"], outline="")
        
        if ratio > 0:
            self.eval_canvas.create_rectangle(0, black_h, 30, height, fill="#E8E8E8", outline="")
            
        r = 15
        
        # Redraw properly
        self.eval_canvas.delete("all")
        rounded_rect(self.eval_canvas, 0, 0, 30, height, 15, fill="#E8E8E8", outline="") # White base
        
        # Draw the black part as a rectangle, then we'll mask corners
        self.eval_canvas.create_rectangle(0, 0, 30, black_h, fill=self.theme["EVAL_BG"], outline="")
        
        # Masks
        # Top-Left mask
        self.eval_canvas.create_polygon(0,0, r,0, 0,r, fill=bg_c, outline=bg_c)
        self.eval_canvas.create_arc(0,0, 2*r,2*r, start=90, extent=90, fill=self.theme["EVAL_BG"], outline=self.theme["EVAL_BG"], style="pieslice")
        
        # Top-Right mask
        self.eval_canvas.create_polygon(30,0, 30-r,0, 30,r, fill=bg_c, outline=bg_c)
        self.eval_canvas.create_arc(30-2*r,0, 30,2*r, start=0, extent=90, fill=self.theme["EVAL_BG"], outline=self.theme["EVAL_BG"], style="pieslice")

        # Bottom-Left mask
        self.eval_canvas.create_polygon(0,height, r,height, 0,height-r, fill=bg_c, outline=bg_c)
        self.eval_canvas.create_arc(0,height-2*r, 2*r,height, start=180, extent=90, fill="#E8E8E8", outline="#E8E8E8", style="pieslice")

        # Bottom-Right mask
        self.eval_canvas.create_polygon(30,height, 30-r,height, 30,height-r, fill=bg_c, outline=bg_c)
        self.eval_canvas.create_arc(30-2*r,height-2*r, 30,height, start=270, extent=90, fill="#E8E8E8", outline="#E8E8E8", style="pieslice")
        
        if ratio >= 0.5:
            text_y = height - 20
            fill_color = self.theme["EVAL_BG"]
        else:
            text_y = 20
            fill_color = "#E8E8E8"

        self.eval_canvas.create_text(15, text_y, text=display_str, fill=fill_color, font=(self.DEFAULT_FONT_FAMILY, 11, "bold"))

    def _draw_piece_on_square(self, square_index, piece):
        x0, y0, x1, y1 = self._square_coords(square_index)
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        
        color_char = "w" if piece.color == chess.WHITE else "b"
        piece_char = piece.symbol().lower()
        key = color_char + piece_char
        
        if key in self.piece_tk_images:
            img_id = self.canvas.create_image(cx, cy, image=self.piece_tk_images[key])
            self.piece_items[square_index] = [img_id]
        else:
             glyph = piece_to_unicode(piece)
             shadow = self.canvas.create_text(cx + 1.5, cy + 2.0, text=glyph, font=(self.DEFAULT_FONT_FAMILY, int(self.square_size*0.72)), fill="#111111")
             fill = "white" if piece.color == chess.BLACK else "black"
             text_id = self.canvas.create_text(cx, cy, text=glyph, font=(self.DEFAULT_FONT_FAMILY, int(self.square_size*0.72)), fill=fill)
             self.piece_items[square_index] = [shadow, text_id]

    def _on_click(self, event):
        col = int(event.x // self.square_size)
        row = int(event.y // self.square_size)
        if not (0 <= col < 8 and 0 <= row < 8):
            return
        rank = 7 - row
        file = col
        sq = chess.square(file, rank)

        if self.selected_square is None:
            piece = self.board.piece_at(sq)
            if piece is None:
                return
            legal_targets = self._compute_legal_targets(sq)
            if legal_targets:
                self.selected_square = sq
                self.legal_targets = legal_targets
                self.attack_targets = []
                self._show_selection_and_moves(self.selected_square, self.legal_targets, self.attack_targets)
            else:
                attacks = sorted(self.board.attacks(sq))
                if attacks:
                    self.selected_square = sq
                    self.legal_targets = []
                    self.attack_targets = attacks
                    self._show_selection_and_moves(self.selected_square, self.legal_targets, self.attack_targets)
                else:
                    return
        else:
            if sq in self.legal_targets:
                move = chess.Move(self.selected_square, sq)
                if move.promotion is None and self.board.piece_type_at(self.selected_square) == chess.PAWN:
                    to_rank = chess.square_rank(sq)
                    if (self.board.turn == chess.WHITE and to_rank == 7) or (self.board.turn == chess.BLACK and to_rank == 0):
                        move = chess.Move(self.selected_square, sq, promotion=chess.QUEEN)
                if move in self.board.legal_moves:
                    self.board.push(move)
                    # clear selection and redraw
                    self.selected_square = None
                    self.legal_targets = []
                    self.attack_targets = []
                    self._redraw_board()
                    if not self.board.is_game_over():
                        self.status_label.config(text="AI thinking...")
                        threading.Thread(target=self._ai_move_thread, daemon=True).start()
                    else:
                        self._on_game_over()
                else:
                    self.selected_square = None
                    self.legal_targets = []
                    self.attack_targets = []
                    self._redraw_board()
            else:
                piece = self.board.piece_at(sq)
                if piece:
                    legal_targets = self._compute_legal_targets(sq)
                    if legal_targets:
                        self.selected_square = sq
                        self.legal_targets = legal_targets
                        self.attack_targets = []
                        self._show_selection_and_moves(self.selected_square, self.legal_targets, self.attack_targets)
                    else:
                        attacks = sorted(self.board.attacks(sq))
                        if attacks:
                            self.selected_square = sq
                            self.legal_targets = []
                            self.attack_targets = attacks
                            self._show_selection_and_moves(self.selected_square, self.legal_targets, self.attack_targets)
                        else:
                            self.selected_square = None
                            self.legal_targets = []
                            self.attack_targets = []
                            self._redraw_board()
                else:
                    self.selected_square = None
                    self.legal_targets = []
                    self.attack_targets = []
                    self._redraw_board()

        self._update_status_text()

    def _compute_legal_targets(self, source_sq):
        targets = []
        for move in self.board.legal_moves:
            if move.from_square == source_sq:
                targets.append(move.to_square)
        return targets

    def _show_selection_and_moves(self, source_sq, legal_targets, attack_targets):
        for item in self.move_markers:
            self.canvas.delete(item)
        self.move_markers.clear()
        if self.select_border:
            self.canvas.delete(self.select_border)
            self.select_border = None

        x0, y0, x1, y1 = self._square_coords(source_sq)
        gap = int(self.square_size * 0.05)
        self.select_border = rounded_rect(self.canvas,
            x0 + gap, y0 + gap, x1 - gap, y1 - gap, 8,
            outline=self.theme["SELECT"], width=max(3, int(self.square_size * 0.06))
        )

        dot_radius = max(6, int(self.square_size * 0.14))
        for tgt in legal_targets:
            tx0, ty0, tx1, ty1 = self._square_coords(tgt)
            cx = (tx0 + tx1) / 2
            cy = (ty0 + ty1) / 2
            is_capture = self.board.piece_at(tgt) is not None and self.board.piece_at(tgt).color != self.board.turn
            if is_capture:
                # Hollow ring for captures
                ring_width = max(3, int(self.square_size * 0.06))
                ring_radius = int(self.square_size * 0.45) - ring_width
                ring = self.canvas.create_oval(cx - ring_radius, cy - ring_radius, cx + ring_radius, cy + ring_radius,
                                               outline=self.theme["MOVE_DOT"], width=ring_width)
                self.move_markers.append(ring)
            else:
                dot = self.canvas.create_oval(cx - dot_radius, cy - dot_radius, cx + dot_radius, cy + dot_radius,
                                              fill=self.theme["MOVE_DOT"], outline="") # smooth dots
                self.move_markers.append(dot)

        attack_radius = max(5, int(self.square_size * 0.10))
        for tgt in attack_targets:
            tx0, ty0, tx1, ty1 = self._square_coords(tgt)
            cx = (tx0 + tx1) / 2
            cy = (ty0 + ty1) / 2
            ring = self.canvas.create_oval(cx - attack_radius, cy - attack_radius, cx + attack_radius, cy + attack_radius,
                                           outline=self.ATTACK_DOT_COLOR, width=max(2, int(self.square_size * 0.02)))
            self.move_markers.append(ring)

        for _, ids in self.piece_items.items():
            for id_ in ids:
                self.canvas.tag_raise(id_)

    def _ai_move_thread(self):
        try:
            val = self.depth_var.get()
            self.ai.depth = self.depth_map.get(val, 3)
        except Exception:
            pass
            
        if self.ai.depth == 1:
            time.sleep(1.2)
        elif self.ai.depth == 2:
            time.sleep(0.6)

        move = self.ai.find_best_move(self.board)
        if move is not None:
            self.board.push(move)
            self.ai_last_move = move
        self.root.after(0, self._redraw_board)

        if self.board.is_game_over():
            self.root.after(0, self._on_game_over)
        else:
            self.root.after(0, lambda: self.status_label.config(text="Your move"))

    def _draw_ai_move_highlight(self, move):
        if not move:
            return
        for item in self.ai_highlight_items:
            try:
                self.canvas.delete(item)
            except Exception:
                pass
        self.ai_highlight_items.clear()

        for sq in (move.from_square, move.to_square):
            x0, y0, x1, y1 = self._square_coords(sq)
            gap = int(self.square_size * 0.05)
            rect = rounded_rect(self.canvas, x0 + gap, y0 + gap, x1 - gap, y1 - gap, 8,
                                                fill=self.AI_MOVE_HIGHLIGHT, outline="")
            self.canvas.tag_lower(rect)
            self.ai_highlight_items.append(rect)

        for _, ids in self.piece_items.items():
            for id_ in ids:
                self.canvas.tag_raise(id_)

    def _on_game_over(self):
        self._redraw_board()
        result = self.board.result()
        self.status_label.config(text=f"Game over: {result}")
        # Update player score
        if result == "1-0":  # player wins
            self.player_score = max(100, self.player_score + 8)
        elif result == "0-1":  # player loses
            self.player_score = max(100, self.player_score - 8)
        # Draw: no change
        self.score_display_lbl.config(text=str(self.player_score))
        self._show_splash_screen(result)

    def _show_splash_screen(self, result):
        splash = tk.Toplevel(self.root)
        splash.title("Game Over")
        splash.geometry("350x250")
        splash.configure(bg=self.theme["PANEL_BG"])
        
        lbl = tk.Label(splash, text=f"Game Over!\n\nResult: {result}", font=(self.DEFAULT_FONT_FAMILY, 18, "bold"), fg=self.theme["TEXT_COLOR"], bg=self.theme["PANEL_BG"])
        lbl.pack(pady=(40, 20))
        
        btn = RoundedButton(splash, text="Game Analysis", bg="#4CAF50", fg="white", hover_bg="#45a049", width=200, height=50, command=lambda: self._open_analysis(splash))
        btn.pack()

    def _open_analysis(self, splash):
        splash.destroy()
        AnalysisWindow(self.root, self.theme, self.board, self.player_score)

    def _undo(self):
        if self.board.move_stack:
            self.board.pop()
        if self.board.move_stack:
            self.board.pop()
        self.ai_last_move = None
        self.selected_square = None
        self.legal_targets = []
        self.attack_targets = []
        self._redraw_board()
        self._update_status_text()

    def _reset_board(self):
        self.board.reset()
        self.ai_last_move = None
        self.selected_square = None
        self.legal_targets = []
        self.attack_targets = []
        self._redraw_board()
        self._update_status_text()

    def _update_status_text(self):
        side = "White" if self.board.turn == chess.WHITE else "Black"
        self.status_label.config(text=f"{side} to move")

    def run(self):
        self._update_status_text()
        self.root.mainloop()

class AnalysisWindow:
    """Full game analysis window with chess.com-style move classifications,
    strengths/weaknesses graph, and per-move alternative analysis."""

    # Chess.com move classification colors
    MOVE_COLORS = {
        "Brilliant":   "#26C2A3",  # teal
        "Great":       "#5B8BF5",  # blue
        "Best":        "#96BC4B",  # green
        "Excellent":   "#96BC4B",  # green (slightly lighter)
        "Good":        "#A3C65C",  # light green
        "Book":        "#A88B62",  # brown
        "Inaccuracy":  "#F7C631",  # yellow
        "Mistake":     "#E68A2E",  # orange
        "Blunder":     "#CA3431",  # red
        "Miss":        "#CA3431",  # red
    }

    MOVE_SYMBOLS = {
        "Brilliant":   "!!",
        "Great":       "!",
        "Best":        "★",
        "Excellent":   "👍",
        "Good":        "✓",
        "Book":        "📖",
        "Inaccuracy":  "?!",
        "Mistake":     "?",
        "Blunder":     "??",
        "Miss":        "—",
    }

    # Categories used for the strengths/weaknesses graph
    ANALYSIS_CATEGORIES = [
        "Opening", "Middlegame", "Endgame",
        "Tactics", "Accuracy", "Resilience"
    ]

    def __init__(self, parent, theme, board, player_score=400):
        self.top = tk.Toplevel(parent)
        self.top.title("Game Analysis")
        self.top.geometry("820x750")
        self.top.configure(bg=theme["BG_COLOR"])
        self.top.minsize(780, 700)
        self.theme = theme
        self.board = board
        self.player_score = player_score
        self.analysis_results = []  # list of dicts per move
        self.category_scores = {c: 0.5 for c in self.ANALYSIS_CATEGORIES}

        # ─── Layout: side pane (right) + main area (left) ───
        self.main_frame = tk.Frame(self.top, bg=theme["BG_COLOR"])
        self.main_frame.pack(fill="both", expand=True)

        # Side pane — score display
        self.side_pane = tk.Frame(self.main_frame, width=160, bg=theme["PANEL_BG"])
        self.side_pane.pack(side="right", fill="y", padx=(0, 0))
        self.side_pane.pack_propagate(False)

        self._build_side_pane()

        # Main content area
        self.content = tk.Frame(self.main_frame, bg=theme["BG_COLOR"])
        self.content.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        # Title
        tk.Label(self.content, text="Game Analysis",
                 font=("Helvetica", 22, "bold"),
                 bg=theme["BG_COLOR"], fg=theme["TEXT_COLOR"]).pack(pady=(5, 10))

        # ─── Top Graph: Strengths (white) / Weaknesses (black) ───
        self.graph_frame = tk.Frame(self.content, bg=theme["PANEL_BG"],
                                     highlightthickness=0)
        self.graph_frame.pack(fill="x", padx=10, pady=(0, 10))

        tk.Label(self.graph_frame, text="Strengths & Weaknesses",
                 font=("Helvetica", 13, "bold"),
                 bg=theme["PANEL_BG"], fg=theme["TEXT_COLOR"]).pack(pady=(8, 2))

        self.graph_canvas = tk.Canvas(self.graph_frame, height=140,
                                       bg=theme["PANEL_BG"], highlightthickness=0)
        self.graph_canvas.pack(fill="x", padx=10, pady=(0, 10))

        # ─── Move classification summary bar ───
        self.summary_frame = tk.Frame(self.content, bg=theme["PANEL_BG"])
        self.summary_frame.pack(fill="x", padx=10, pady=(0, 10))
        tk.Label(self.summary_frame, text="Move Classification Summary",
                 font=("Helvetica", 13, "bold"),
                 bg=theme["PANEL_BG"], fg=theme["TEXT_COLOR"]).pack(pady=(8, 4))
        self.summary_canvas = tk.Canvas(self.summary_frame, height=70,
                                         bg=theme["PANEL_BG"], highlightthickness=0)
        self.summary_canvas.pack(fill="x", padx=10, pady=(0, 10))

        # ─── Detailed move list (scrollable) ───
        tk.Label(self.content, text="Move-by-Move Breakdown",
                 font=("Helvetica", 13, "bold"),
                 bg=theme["BG_COLOR"], fg=theme["TEXT_COLOR"]).pack(pady=(0, 4), anchor="w", padx=12)

        self.text_area = tk.Text(self.content, bg=theme["PANEL_BG"],
                                  fg=theme["TEXT_COLOR"],
                                  font=("Helvetica", 11), wrap="word",
                                  highlightthickness=0, borderwidth=0,
                                  padx=10, pady=10)
        self.text_area.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.text_area.insert("1.0", "Analyzing game…\nEvaluating 4-6 candidate moves per position.\nThis may take a moment.")
        self.text_area.config(state="disabled")

        # Configure text tags for colouring
        for cls_name, color in self.MOVE_COLORS.items():
            self.text_area.tag_configure(cls_name, foreground=color,
                                          font=("Helvetica", 11, "bold"))
        self.text_area.tag_configure("header",
                                      font=("Helvetica", 12, "bold"),
                                      foreground=theme["TEXT_COLOR"])
        self.text_area.tag_configure("dim",
                                      foreground="#888888",
                                      font=("Helvetica", 10))

        # Start analysis in background
        threading.Thread(target=self.run_analysis, daemon=True).start()

    # ───────────────────── Side Pane ─────────────────────
    def _build_side_pane(self):
        th = self.theme
        tk.Label(self.side_pane, text="Your Score",
                 font=("Helvetica", 14, "bold"),
                 bg=th["PANEL_BG"], fg=th["TEXT_COLOR"]).pack(pady=(25, 5))

        self.score_lbl = tk.Label(self.side_pane,
                                   text=str(self.player_score),
                                   font=("Helvetica", 36, "bold"),
                                   bg=th["PANEL_BG"], fg="#26C2A3")
        self.score_lbl.pack(pady=(0, 5))

        tk.Label(self.side_pane, text="Rating Points",
                 font=("Helvetica", 10),
                 bg=th["PANEL_BG"], fg="#888888").pack()

        # Separator
        sep = tk.Frame(self.side_pane, height=1, bg="#444444")
        sep.pack(fill="x", padx=15, pady=15)

        # Point change this game (will update after analysis)
        self.delta_lbl = tk.Label(self.side_pane, text="…",
                                   font=("Helvetica", 16, "bold"),
                                   bg=th["PANEL_BG"], fg=th["TEXT_COLOR"])
        self.delta_lbl.pack(pady=(0, 2))
        tk.Label(self.side_pane, text="This Game",
                 font=("Helvetica", 10),
                 bg=th["PANEL_BG"], fg="#888888").pack()

        sep2 = tk.Frame(self.side_pane, height=1, bg="#444444")
        sep2.pack(fill="x", padx=15, pady=15)

        # Accuracy (will update)
        tk.Label(self.side_pane, text="Accuracy",
                 font=("Helvetica", 12, "bold"),
                 bg=th["PANEL_BG"], fg=th["TEXT_COLOR"]).pack(pady=(0, 3))
        self.acc_lbl = tk.Label(self.side_pane, text="…",
                                 font=("Helvetica", 24, "bold"),
                                 bg=th["PANEL_BG"], fg="#96BC4B")
        self.acc_lbl.pack()

        sep3 = tk.Frame(self.side_pane, height=1, bg="#444444")
        sep3.pack(fill="x", padx=15, pady=15)

        # Classification legend
        tk.Label(self.side_pane, text="Legend",
                 font=("Helvetica", 11, "bold"),
                 bg=th["PANEL_BG"], fg=th["TEXT_COLOR"]).pack(pady=(0, 5))
        for cls_name in ["Brilliant", "Great", "Best", "Good",
                         "Inaccuracy", "Mistake", "Blunder"]:
            row = tk.Frame(self.side_pane, bg=th["PANEL_BG"])
            row.pack(anchor="w", padx=15, pady=1)
            dot = tk.Canvas(row, width=10, height=10,
                            bg=th["PANEL_BG"], highlightthickness=0)
            dot.pack(side="left", padx=(0, 6))
            dot.create_oval(1, 1, 9, 9, fill=self.MOVE_COLORS[cls_name],
                            outline="")
            tk.Label(row, text=f"{self.MOVE_SYMBOLS[cls_name]} {cls_name}",
                     font=("Helvetica", 9),
                     bg=th["PANEL_BG"],
                     fg=th["TEXT_COLOR"]).pack(side="left")

    # ───────────────────── Analysis Engine ─────────────────────
    def _classify_move(self, board, played_move, ranked_moves, best_eval,
                       played_eval, is_book_phase):
        """Classify a move using chess.com-style categories.
        ranked_moves: list of (move, eval) sorted best-first.
        """
        if is_book_phase:
            return "Book"

        # Eval loss (from the perspective of the side that moved)
        eval_diff = abs(best_eval - played_eval)

        is_capture = board.is_capture(played_move)
        is_check = board.gives_check(played_move)

        # piece sacrifice detection
        is_sacrifice = False
        if is_capture:
            # Check if we're giving up a higher-value piece
            moving_piece = board.piece_at(played_move.from_square)
            captured_piece = board.piece_at(played_move.to_square)
            if moving_piece and captured_piece:
                if Evaluator.PIECE_VALUES.get(moving_piece.piece_type, 0) > \
                   Evaluator.PIECE_VALUES.get(captured_piece.piece_type, 0):
                    is_sacrifice = True
        # Non-capture sacrifice: moving a piece to a square attacked by opponent
        if not is_capture and board.piece_at(played_move.from_square):
            moving_piece = board.piece_at(played_move.from_square)
            if moving_piece and moving_piece.piece_type != chess.PAWN:
                # Check if destination is attacked by opponent
                opp_color = not board.turn
                if board.is_attacked_by(opp_color, played_move.to_square):
                    is_sacrifice = True

        # Best move match
        is_best = played_move == ranked_moves[0][0] if ranked_moves else False

        # ── Classification logic ──
        if is_best and is_sacrifice and eval_diff < 50:
            return "Brilliant"

        if is_best and (is_check or eval_diff < 5):
            # Check if this move was critical (large swing)
            if len(ranked_moves) >= 2:
                second_eval = ranked_moves[1][1]
                swing = abs(best_eval - second_eval)
                if swing > 80:
                    return "Great"
            return "Best"

        if eval_diff < 10:
            return "Best"
        elif eval_diff < 30:
            return "Excellent"
        elif eval_diff < 60:
            return "Good"
        elif eval_diff < 120:
            return "Inaccuracy"
        elif eval_diff < 250:
            return "Mistake"
        else:
            # Check for missed win / miss
            if best_eval > 500 and played_eval < 200:
                return "Miss"
            return "Blunder"

    def _evaluate_top_moves(self, board, n=5):
        """Evaluate the top n legal moves and return sorted (move, eval)."""
        engine = ChessAI(depth=3)
        move_evals = []
        maximizing = board.turn == chess.WHITE

        for move in list(board.legal_moves)[:30]:  # limit for speed
            board.push(move)
            val = engine.minimax(board, engine.depth - 1, -math.inf, math.inf,
                                 not maximizing)
            board.pop()
            move_evals.append((move, val))

        # Sort: best first for the side to move
        if maximizing:
            move_evals.sort(key=lambda x: x[1], reverse=True)
        else:
            move_evals.sort(key=lambda x: x[1])

        return move_evals[:n]

    def run_analysis(self):
        moves = list(self.board.move_stack)
        temp_board = chess.Board()
        results = []  # per-move analysis results

        # Counters for classification
        cls_counts = {c: 0 for c in self.MOVE_COLORS}
        player_move_count = 0  # only player (white) moves
        player_good_moves = 0

        # Category trackers
        opening_moves = 0
        opening_good = 0
        middle_moves = 0
        middle_good = 0
        end_moves = 0
        end_good = 0
        tactical_hits = 0
        tactical_chances = 0
        resilience_score = 0
        was_losing = False

        total_moves = len(moves)

        for i, move in enumerate(moves):
            turn_color = "White" if temp_board.turn == chess.WHITE else "Black"
            is_player = temp_board.turn == chess.WHITE  # player is white

            # Book phase: first 6 moves (3 per side)
            is_book = i < 6

            # Evaluate top 4-6 candidate moves
            num_candidates = min(6, max(4, len(list(temp_board.legal_moves))))
            ranked = self._evaluate_top_moves(temp_board, n=num_candidates)

            if ranked:
                best_eval = ranked[0][1]
            else:
                best_eval = 0

            # Find eval of the played move
            played_eval = best_eval  # default if it's the best
            for m, ev in ranked:
                if m == move:
                    played_eval = ev
                    break
            else:
                # If played move not in top N, evaluate it directly
                temp_board.push(move)
                played_eval = Evaluator.evaluate(temp_board)
                temp_board.pop()

            classification = self._classify_move(temp_board, move, ranked,
                                                  best_eval, played_eval,
                                                  is_book)

            # Alternatives text
            alternatives = []
            for m, ev in ranked[:5]:
                marker = "→" if m == move else " "
                alt_san = temp_board.san(m)
                alternatives.append(f"  {marker} {alt_san}  (eval: {ev/100:+.1f})")

            result = {
                "index": i,
                "move_num": (i // 2) + 1,
                "turn": turn_color,
                "move_uci": move.uci(),
                "move_san": temp_board.san(move),
                "classification": classification,
                "best_eval": best_eval,
                "played_eval": played_eval,
                "eval_diff": abs(best_eval - played_eval),
                "alternatives": alternatives,
                "is_player": is_player,
            }
            results.append(result)

            # Update counters for player moves
            if is_player:
                player_move_count += 1
                cls_counts[classification] = cls_counts.get(classification, 0) + 1

                good = classification in ("Brilliant", "Great", "Best",
                                           "Excellent", "Good", "Book")
                if good:
                    player_good_moves += 1

                # Phase detection
                phase_ratio = i / max(total_moves, 1)
                if phase_ratio < 0.25:
                    opening_moves += 1
                    if good:
                        opening_good += 1
                elif phase_ratio < 0.7:
                    middle_moves += 1
                    if good:
                        middle_good += 1
                else:
                    end_moves += 1
                    if good:
                        end_good += 1

                # Tactics
                if classification in ("Brilliant", "Great"):
                    tactical_hits += 1
                if temp_board.is_capture(move) or temp_board.gives_check(move):
                    tactical_chances += 1

                # Resilience
                current_eval = played_eval
                if (temp_board.turn == chess.WHITE and current_eval < -100) or \
                   (temp_board.turn == chess.BLACK and current_eval > 100):
                    was_losing = True
                if was_losing and good:
                    resilience_score += 1

            temp_board.push(move)

        self.analysis_results = results

        # Compute category scores (0.0 to 1.0)
        def safe_ratio(a, b):
            return a / b if b > 0 else 0.5

        accuracy = safe_ratio(player_good_moves, player_move_count) if player_move_count > 0 else 0.5

        self.category_scores = {
            "Opening":     safe_ratio(opening_good, max(opening_moves, 1)),
            "Middlegame":  safe_ratio(middle_good, max(middle_moves, 1)),
            "Endgame":     safe_ratio(end_good, max(end_moves, 1)),
            "Tactics":     min(1.0, safe_ratio(tactical_hits, max(tactical_chances, 1)) + 0.1),
            "Accuracy":    accuracy,
            "Resilience":  min(1.0, 0.4 + resilience_score * 0.15),
        }

        # Calculate score delta
        # Win = +8, base adjustments for accuracy
        result_str = self.board.result()
        if result_str == "1-0":
            base_delta = 8
        elif result_str == "0-1":
            base_delta = -8
        else:
            base_delta = 0

        # Adjust with accuracy bonus/penalty
        acc_bonus = int((accuracy - 0.5) * 6)
        delta = base_delta + acc_bonus
        new_score = max(100, self.player_score + delta)

        # Schedule UI updates
        self.top.after(0, lambda: self._render_results(
            results, cls_counts, accuracy, delta, new_score))

    # ───────────────────── Rendering ─────────────────────
    def _render_results(self, results, cls_counts, accuracy, delta, new_score):
        # Update score
        self.score_lbl.config(text=str(new_score))
        sign = "+" if delta >= 0 else ""
        delta_color = "#26C2A3" if delta >= 0 else "#CA3431"
        self.delta_lbl.config(text=f"{sign}{delta}", fg=delta_color)
        self.acc_lbl.config(text=f"{accuracy * 100:.0f}%")

        # ── Draw strengths/weaknesses graph ──
        self.graph_canvas.update_idletasks()
        self._draw_strength_graph()

        # ── Draw classification summary bar ──
        self._draw_summary_bar(cls_counts)

        # ── Populate move list ──
        self.text_area.config(state="normal")
        self.text_area.delete("1.0", "end")

        for r in results:
            cls = r["classification"]
            sym = self.MOVE_SYMBOLS.get(cls, "")
            tag = cls

            # Header line
            header = f"Move {r['move_num']} ({r['turn']}): {r['move_san']}  "
            self.text_area.insert("end", header, "header")
            self.text_area.insert("end", f"{sym} {cls}\n", tag)

            # Eval info
            diff_str = f"Eval: {r['played_eval']/100:+.1f}  |  Best: {r['best_eval']/100:+.1f}  |  Loss: {r['eval_diff']/100:.1f}\n"
            self.text_area.insert("end", diff_str, "dim")

            # Alternatives
            self.text_area.insert("end", "Candidates:\n", "dim")
            for alt in r["alternatives"]:
                self.text_area.insert("end", alt + "\n", "dim")

            self.text_area.insert("end", "\n")

        self.text_area.yview_moveto(0)
        self.text_area.config(state="disabled")

    def _draw_strength_graph(self):
        """Draw horizontal bar graph: white = good, black = bad."""
        c = self.graph_canvas
        c.delete("all")
        c.update_idletasks()
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 10:
            w = 500
        if h < 10:
            h = 130

        n = len(self.ANALYSIS_CATEGORIES)
        bar_height = max(14, (h - 20) // n - 6)
        pad_left = 90
        pad_right = 20
        bar_width = w - pad_left - pad_right
        y_start = 10

        for i, cat in enumerate(self.ANALYSIS_CATEGORIES):
            y = y_start + i * (bar_height + 6)
            score = self.category_scores.get(cat, 0.5)

            # Category label
            c.create_text(pad_left - 8, y + bar_height // 2,
                          text=cat, anchor="e",
                          fill=self.theme["TEXT_COLOR"],
                          font=("Helvetica", 10, "bold"))

            # Background bar (dark = weakness)
            c.create_rectangle(pad_left, y, pad_left + bar_width,
                               y + bar_height,
                               fill="#1A1A1A", outline="#333333")

            # Filled portion (white = strength)
            fill_w = int(bar_width * score)
            if fill_w > 0:
                c.create_rectangle(pad_left, y, pad_left + fill_w,
                                   y + bar_height,
                                   fill="#E8E8E8", outline="")

            # Percentage text
            c.create_text(pad_left + bar_width + 5,
                          y + bar_height // 2,
                          text=f"{score * 100:.0f}%", anchor="w",
                          fill=self.theme["TEXT_COLOR"],
                          font=("Helvetica", 9))

    def _draw_summary_bar(self, cls_counts):
        """Draw coloured summary blocks for each classification."""
        c = self.summary_canvas
        c.delete("all")
        c.update_idletasks()
        w = c.winfo_width()
        if w < 10:
            w = 500

        order = ["Brilliant", "Great", "Best", "Excellent", "Good", "Book",
                 "Inaccuracy", "Mistake", "Blunder", "Miss"]
        total = sum(cls_counts.values())
        if total == 0:
            total = 1

        x = 10
        block_h = 30
        y = 5

        for cls_name in order:
            count = cls_counts.get(cls_name, 0)
            if count == 0:
                continue
            bw = max(30, int((w - 20) * count / total))
            color = self.MOVE_COLORS[cls_name]
            c.create_rectangle(x, y, x + bw, y + block_h,
                               fill=color, outline="")
            # Label
            label = f"{self.MOVE_SYMBOLS[cls_name]} {count}"
            c.create_text(x + bw // 2, y + block_h // 2,
                          text=label, fill="white",
                          font=("Helvetica", 9, "bold"))
            # Name below
            c.create_text(x + bw // 2, y + block_h + 12,
                          text=cls_name, fill="#888888",
                          font=("Helvetica", 8))
            x += bw + 4

def board_to_unicode_ascii(board: chess.Board) -> str:
    rows = []
    for rank in range(7, -1, -1):
        row = f"{rank + 1} "
        for file in range(0, 8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            glyph = piece_to_unicode(piece)
            row += (glyph if glyph else ".") + " "
        rows.append(row)
    rows.append("  a b c d e f g h")
    return "\n".join(rows)


def run_cli():
    board = chess.Board()
    ai = ChessAI(depth=3)
    print("AI Chess Bot — CLI (unicode pieces)")
    print("Enter moves in UCI format (e2e4). Promotion will default to queen if omitted.")
    while not board.is_game_over():
        print(board_to_unicode_ascii(board))
        if board.turn == chess.WHITE:
            uci = input("Your move: ").strip()
            try:
                move = chess.Move.from_uci(uci)
                if move.promotion is None and board.piece_type_at(move.from_square) == chess.PAWN:
                    to_rank = chess.square_rank(move.to_square)
                    if (board.turn == chess.WHITE and to_rank == 7) or (board.turn == chess.BLACK and to_rank == 0):
                        move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("Illegal move.")
                    continue
            except Exception as e:
                print("Invalid UCI format. Example: e2e4 (error:", e, ")")
                continue
        else:
            print("AI thinking...")
            move = ai.find_best_move(board)
            if move is None:
                break
            board.push(move)
            print("AI played:", move)
    print("Game over:", board.result())
    print(board_to_unicode_ascii(board))

if __name__ == "__main__":
    if "--cli" in sys.argv:
        run_cli()
    else:
        gui = ModernChessGUI(ai_depth=3, square_size=96)
        gui.run()