import tkinter as tk
import game2048
from collections import defaultdict


class Tile(tk.Canvas):

    colors = {
        0: '#dddddd',
        2: '#f4f0ed',
        4: '#f7e2d2',
        8: '#f9c59d',
        16: '#efa56b',
        32: '#e86f47',
        64: '#d14314',
        128: '#f7f091',
        256: '#f9f06d',
        512: '#f9ee4a',
        1024: '#ede136',
        2048: '#e5d714'
    }
    color_map = defaultdict(lambda: "#23231f",colors)

    def __init__(self, master, number, size=50):
        tk.Canvas.__init__(self, master, height=size, width=size,
            background=Tile.color_map[number])
        if not number == 0:
            self.text = tk.Canvas.create_text(self, 25, 25, anchor=tk.CENTER, text=str(int(number)))
        else:
            self.text = tk.Canvas.create_text(self, 25, 25, anchor=tk.CENTER, text='')

    def set_state(self, number):
        self.configure(background=Tile.color_map[number])
        if not number == 0:
            self.itemconfigure(self.text, text=str(int(number)))
        else:
            self.itemconfigure(self.text, text='')


class Board(tk.Frame):
    """2048 game board container for all tiles"""

    def __init__(self, master, game):
        tk.Frame.__init__(self, master)
        self.game = game
        self.n = game.n

        self.tiles = []
        for row in range(self.n):
            row_tiles = []
            for col in range(self.n):
                tile = Tile(self, game.get_tile(row, col))
                tile.grid(row=row, column=col, padx=1, pady=1)
                row_tiles.append(tile)
            self.tiles.append(row_tiles)

    def update_tiles(self):
        for row in range(self.n):
            for col in range(self.n):
                self.tiles[row][col].set_state(self.game.get_tile(row, col))


class game2048GUI(tk.Frame):

    def __init__(self, master,game):
        tk.Frame.__init__(self, master)

        self.game = game
        self.board = Board(self, self.game)

        self.board.pack(side=tk.LEFT, padx=1, pady=1)
        self.focus_set()

    def update_gui(self):
        self.board.update_tiles()



class HumanGUI(tk.Frame):
    def __init__(self,master):

        tk.Frame.__init__(self, master)

        self.game = game2048.game2048(4)
        self.board = HumanBoard(self, self.game)

        self.board.pack(side=tk.LEFT, padx=1, pady=1)
        self.focus_set()

        self.bind('<Left>', lambda event: self.board.perform_move("left"))
        self.bind('<Up>', lambda event: self.board.perform_move("up"))
        self.bind('<Down>', lambda event: self.board.perform_move("down"))
        self.bind('<Right>', lambda event: self.board.perform_move("right"))

class HumanBoard(Board):

    def __init__(self,master,game):
        Board.__init__(self,master,game)

    def perform_move(self, move_dir):
        self.game.swipe(move_dir)
        self.update_tiles()
        game_over_state = self.game.check_for_game_over()
        if game_over_state:
            print(game_over_state)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("2048 Game")
    HumanGUI(root).pack()
    root.mainloop()

