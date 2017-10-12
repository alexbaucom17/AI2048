import numpy as np
import math


#parameters
TWO_FREQ = 0.75 #twos appear this fraction of the time and 4s appear 1-TWO_FREQ

class game2048:

    def __init__(self, n):
        #create a new game and initialize two tiles
        self.n = n
        self.game_state = np.zeros([n, n])
        self.generate_tile()
        self.generate_tile()

    def get_tile(self, row, col):
        return self.game_state[row][col]

    def get_empty_idx(self):
        # determine which tiles are empty and return list of linear indeces
        idx = np.arange(self.n**2)
        return idx[np.reshape(self.game_state, -1) == 0]

    def generate_tile(self):
        # randomly choose empty slot and value for new tile
        tile_idx = np.random.choice(self.get_empty_idx(), 1)
        if np.random.random() < TWO_FREQ:
            tile_val = 2
        else:
            tile_val = 4
        self.place_tile(tile_idx, tile_val)

    def place_tile(self, idx, val):
        # place given value tile at specified index
        if idx.shape[0] == 1:
            self.game_state[self.idx_lin2vec(idx)] = val
        elif idx.shape[0] == 2:
            self.game_state[idx] = val
        else:
            raise ValueError('Invalid array size')

    def idx_lin2vec(self, idx_lin):
        # convert linear index into xy coordinates
        x = math.floor(idx_lin / self.n)
        y = idx_lin - x*self.n
        return [x, y]

    def idx_vec2lin(self, idx_vec):
        # convert xy cooreindate into linear index
        return self.n*idx_vec[0] + idx_vec[1]

    def show_state(self):
        print(self.game_state)

    def swipe(self, swipe_direction):
        old_state = np.copy(self.game_state)

        if swipe_direction == 'left':
            for i in range(self.n):
                self.move_col(i, -1)
        elif swipe_direction == 'right':
            for i in range(self.n-1, -1, -1):
                self.move_col(i, 1)
        elif swipe_direction == 'up':
            for i in range(self.n):
                self.move_row(i, -1)
        elif swipe_direction == 'down':
            for i in range(self.n-1, -1, -1):
                self.move_row(i, 1)
        else:
            raise ValueError('invalid direction')

        # only generate a new tile if the game state changed
        if not np.array_equal(old_state,self.game_state):
            self.generate_tile()

    def check_for_game_over(self):
        if np.any(self.game_state == 2048):
            return "WIN"

        if self.is_board_full() and self.check_for_valid_moves() == False:
            return "LOSE"
        else:
            return None

    def check_for_valid_moves(self):

        valid_moves = False
        old_state = np.copy(self.game_state)
        for move in ['up','down','left','right']:
            self.swipe(move)
            if not np.array_equal(old_state, self.game_state):
                valid_moves = True
                self.game_state = np.copy(old_state)
                break
            else:
                self.game_state = np.copy(old_state)

        return valid_moves

    def is_board_full(self):
        return np.all(self.game_state != 0)

    def move_row(self, row, direction):
        for i in range(self.n):
            self.move_val(np.array([row, i]), np.array([direction, 0]))

    def move_col(self, col, direction):
        for i in range(self.n):
            self.move_val(np.array([i, col]), np.array([0, direction]))

    def move_val(self, idx, vec):
        # move value at idx=[x,y] in direction of vec (i.e. [1,0])

        val = self.game_state[idx[0], idx[1]]
        blocked = False

        while not blocked:
            move_type = self.check_move_type(idx, vec)

            if move_type == 0:
                blocked = True
            elif move_type == 1:
                idx_new = idx + vec
                self.game_state[idx[0], idx[1]] = 0
                self.game_state[idx_new[0], idx_new[1]] = val
                idx = idx_new
            else:
                idx_new = idx+vec
                self.game_state[idx[0], idx[1]] = 0
                self.game_state[idx_new[0], idx_new[1]] += val
                blocked = True

    def check_move_type(self,idx,vec):
        #check to figure out what move is allowed for this value in this direction
        # 0 = blocked, 1 = free, 2 = combine

        moved_idx = idx+vec
        if np.any(moved_idx < 0) or np.any(moved_idx >= self.n):
            return 0  # blocked at edge

        adjacent_val = self.game_state[moved_idx[0], moved_idx[1]]
        current_val = self.game_state[idx[0], idx[1]]
        if adjacent_val == 0:
            return 1  # free to move
        elif adjacent_val == current_val:
            return 2  # combine values
        else:
            return 0  # blocked with other tile


if __name__ == '__main__':
    mygame = game2048(4)
    mygame.generate_tile()
    mygame.generate_tile()
    mygame.show_state()
    print('====================')
    mygame.swipe('right')
    mygame.show_state()
