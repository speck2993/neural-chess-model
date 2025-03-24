import numpy as np
import re
import zlib
from typing import List, Tuple, Dict

def move_type(source_rank, source_file, dest_rank, dest_file) -> int:
    # Returns the type of move based on the source and destination squares
    # Type is encoded as the index of the plane in the policy tensor
    # Does not work for underpromotions or castling

    rank_diff = dest_rank - source_rank
    file_diff = dest_file - source_file

    knight_moves = {(-2,-1):56, (-2,1):57, (-1,-2):58, (-1,2):59, (1,-2):60, (1,2):61, (2,-1):62, (2,1):63}
    
    #first, check if it's a knight move
    if (rank_diff, file_diff) in knight_moves:
        return knight_moves[(rank_diff, file_diff)]
    
    #otherwise, it's a ray move

    #first, check if it's a lateral move
    if file_diff == 0:
        if rank_diff > 0:
            return 27 + rank_diff
        return -1 - rank_diff
    if rank_diff == 0:
        if file_diff > 0:
            return 13 + file_diff
        return 41 - file_diff
    
    #otherwise, it's a diagonal move
    if rank_diff == file_diff:
        if rank_diff > 0:
            return 20 + rank_diff
        return 48 - rank_diff
    if rank_diff == -file_diff:
        if rank_diff > 0:
            return 34 + rank_diff
        return 6 - rank_diff

class FastPGNBoard:
    def __init__(self, precompute=True):
        board = np.zeros((17, 8, 8), dtype=np.int8)
        # Set up initial positions (0-5 for white, 6-11 for black)
        board[0, 1] = 1  # White pawns
        board[1, 0, [1, 6]] = 1  # White knights
        board[2, 0, [2, 5]] = 1  # White bishops
        board[3, 0, [0, 7]] = 1  # White rooks
        board[4, 0, 4] = 1  # White queen
        board[5, 0, 3] = 1  # White king
        board[6, 6] = 1  # Black pawns
        board[7, 7, [1, 6]] = 1  # Black knights
        board[8, 7, [2, 5]] = 1  # Black bishops
        board[9, 7, [0, 7]] = 1  # Black rooks
        board[10, 7, 4] = 1  # Black queen
        board[11, 7, 3] = 1  # Black king
        board[12] = -1  # Active player (white)
        board[15, 0, [1, 5]] = 1  # Castling rights
        board[15, 7, [1, 5]] = 1
        self.board = board
        self.board_appeared_once = set()
        self.board_appeared_twice = set()
        self.board_appeared_once.add(self.hash_position())
        self.en_passant = (-1, -1)
        self.piece_locations=[] #stores locations of each piece type besides pawns, since this makes finding the source of a move easier
        self.piece_locations.append([(0,1),(0,6)]) #white knights
        self.piece_locations.append([(0,2),(0,5)]) #white bishops
        self.piece_locations.append([(0,0),(0,7)]) #white rooks
        self.piece_locations.append([(0,4)]) #white queen
        self.piece_locations.append([(0,3)]) #white king
        self.piece_locations.append([(7,1),(7,6)]) #black knights
        self.piece_locations.append([(7,2),(7,5)]) #black bishops
        self.piece_locations.append([(7,0),(7,7)]) #black rooks
        self.piece_locations.append([(7,4)]) #black queen
        self.piece_locations.append([(7,3)]) #black king

        if precompute:
            # Dictionary to store precomputed intervening squares for all possible ray moves
            self.RAY_MOVE_CACHE = {}
            self.precompute_ray_moves()
            self.PINNED_SQUARE_CACHE = {}
            self.precompute_pinned_squares()

    def reset(self):
        # Reset the board to the initial position
        self.__init__(precompute=False)

    def hash_position(self) -> int:
        return zlib.adler32(self.board.tobytes())
    
    def precompute_pinned_squares(self):
        # Precompute all possible pinned squares for each piece type
        # For every possible piece location and king location, return all squares that could pin that piece to the king
        
        # Piece type is irrelevant, just piece square and king square
        for piece_square in range(64):
            for king_square in range(64):
                piece_row = piece_square // 8
                piece_col = piece_square % 8
                king_row = king_square // 8
                king_col = king_square % 8

                #first, check if the squares are related by a ray move
                rank_diff = piece_row - king_row #note: order matters here!!
                col_diff = piece_col - king_col
                if rank_diff == 0 and col_diff == 0:
                    self.PINNED_SQUARE_CACHE[((piece_row, piece_col), (king_row, king_col))] = ([], (0, 0))
                    continue
                if rank_diff == 0 or col_diff == 0 or abs(rank_diff) == abs(col_diff):
                    dir_rank = 0
                    dir_col = 0
                    if rank_diff > 0:
                        dir_rank = 1
                    elif rank_diff < 0:
                        dir_rank = -1
                    if col_diff > 0:
                        dir_col = 1
                    elif col_diff < 0:
                        dir_col = -1

                    #the pinning squares are (piece_square + n*dir) for as many non-negative n as possible
                    #store everything as tuples for ease of use
                    #also store the direction tuple so we can tell whether we're looking for rook or bishop moves later
                    pinning_squares = []
                    n = 1
                    while 0 <= piece_row + n*dir_rank < 8 and 0 <= piece_col + n*dir_col < 8:
                        pinning_squares.append((piece_row + n*dir_rank,piece_col + n*dir_col))
                        n += 1

                    key = ((piece_row, piece_col), (king_row, king_col))
                    self.PINNED_SQUARE_CACHE[key] = (pinning_squares, (dir_rank, dir_col))
                else:
                    self.PINNED_SQUARE_CACHE[((piece_row, piece_col), (king_row, king_col))] = ([], (0, 0))

    def precompute_ray_moves(self):
        # Precompute all possible intervening squares for ray moves
        for source_rank in range(8):
            for source_file in range(8):
                for dest_rank in range(8):
                    for dest_file in range(8):
                        # Skip if source and destination are the same
                        if source_rank == dest_rank and source_file == dest_file:
                            continue
                        
                        # Only compute for rays (horizontal, vertical, diagonal)
                        rank_diff = dest_rank - source_rank
                        file_diff = dest_file - source_file
                        if rank_diff != 0 and file_diff != 0 and abs(rank_diff) != abs(file_diff):
                            continue
                        
                        # Compute increment for ray direction
                        increment = [0, 0]
                        if rank_diff > 0: increment[0] = 1
                        elif rank_diff < 0: increment[0] = -1
                        if file_diff > 0: increment[1] = 1
                        elif file_diff < 0: increment[1] = -1
                        
                        # Compute intervening squares
                        intervening_squares = []
                        current_rank = source_rank + increment[0]
                        current_file = source_file + increment[1]
                        while current_rank != dest_rank or current_file != dest_file:
                            intervening_squares.append((current_rank, current_file))
                            current_rank += increment[0]
                            current_file += increment[1]
                        
                        # Store in cache
                        self.RAY_MOVE_CACHE[(source_rank, source_file, dest_rank, dest_file)] = intervening_squares

    def is_legal_ray_move(self, source_rank, source_file, dest_rank, dest_file) -> bool:
        # Checks if a ray move from the given source to the given destination is legal
        # Uses precomputed intervening squares from the cache
        
        # Get intervening squares from cache
        key = (source_rank, source_file, dest_rank, dest_file)
        intervening_squares = self.RAY_MOVE_CACHE.get(key)
        
        # If not in cache (not a ray move), return False
        if intervening_squares is None:
            return False
        
        # Check if any piece is in the way
        for square in intervening_squares:
            if self.board[:12, square[0], square[1]].max() == 1:
                return False
            
        return True
    
    def is_pinned(self, source_file, source_rank, player):
        #check if the piece at the given square is pinned for the player
        
        #first, check if the square is occupied by a piece of the player
        player_start_plane = 0 if player == -1 else 6
        piece = self.board[player_start_plane:player_start_plane+5, source_rank, source_file].max() #king is not pinned to itself
        if piece == 0:
            return False
        
        #next, find the king
        king_rank, king_file = self.piece_locations[4][0] if player == -1 else self.piece_locations[9][0]

        #now, make sure there are no intervening pieces using the ray move cache
        
        if (source_rank, source_file, king_rank, king_file) not in self.RAY_MOVE_CACHE:
            return False #not a ray move, so can't be pinned
        
        intervening_squares = self.RAY_MOVE_CACHE[(source_rank, source_file, king_rank, king_file)]
        for square in intervening_squares:
            if self.board[:12, square[0], square[1]].max() == 1:
                return False

        pinning_squares, direction = self.PINNED_SQUARE_CACHE[((source_rank, source_file), (king_rank, king_file))]
        #check if direction is lateral or diagonal
        planes_to_check = []
        if direction[0] == 0 or direction[1] == 0:
            #look for queens and rooks
            planes_to_check = [9,10] if player == -1 else [3,4] #looking for opposite-color pieces
        else:
            #look for queens and bishops
            planes_to_check = [8,10] if player == -1 else [2,4]

        for square in pinning_squares:
            #check if the square is occupied
            #if it's occupied by a piece of the right type, the piece is pinned
            #if it's occupied by a piece of the wrong type, the piece is not pinned and we return early
            if any(self.board[plane, square[0], square[1]] == 1 for plane in planes_to_check):
                return True
            if self.board[:12, square[0], square[1]].max() == 1:
                return False
                
        return False

    def find_piece(self, piece, dest_file, dest_rank, hint, capture, player) -> Tuple[Tuple[int, int], int]:
        """
        Find the source of a move given the destination.
        Returns the rank and file of the source and the index of the piece location for faster removal.
        
        Parameters:
        - piece: The piece type ('P', 'N', 'B', 'R', 'Q', 'K')
        - dest_file, dest_rank: Destination coordinates
        - hint: Disambiguation hint (file or rank) if provided
        - capture: Whether the move is a capture
        - player: -1 for White (bottom), 1 for Black (top)
        
        Returns:
        - Tuple containing:
            - Source coordinates (rank, file)
            - Index in piece_locations (or None if not needed)
        """
        ranks = '12345678'
        files = 'hgfedcba'  # reverse order
        
        # Special handling for pawns
        if piece == 'P':
            return self._find_pawn(dest_file, dest_rank, hint, capture, player)
        
        # Handle Kings - they can't be pinned to themselves
        if piece == 'K':
            if player == -1:
                return self.piece_locations[4][0], 0
            else:
                return self.piece_locations[9][0], 0
        
        # Get piece index based on piece type and player
        piece_idx = self._get_piece_index(piece, player)
        
        # Use hint if provided (disambiguation)
        if hint is not None:
            return self._find_piece_with_hint(piece_idx, hint, ranks, files, dest_rank, dest_file, player)
        
        # Find legal moves without hint for each piece type
        if piece == 'N':
            return self._find_knight(piece_idx, dest_file, dest_rank, player)
        elif piece == 'B': 
            return self._find_bishop(piece_idx, dest_file, dest_rank, player)
        elif piece == 'R':
            return self._find_rook(piece_idx, dest_file, dest_rank, player)
        elif piece == 'Q':
            return self._find_queen(piece_idx, dest_file, dest_rank, player)

    def _get_piece_index(self, piece, player):
        """Map piece type and player to the appropriate index in piece_locations."""
        piece_map = {
            'N': 0, 'B': 1, 'R': 2, 'Q': 3, 'K': 4  # White piece indices
        }
        offset = 0 if player == -1 else 5  # Black pieces start at index 5
        return piece_map.get(piece, 0) + offset

    def _find_pawn(self, dest_file, dest_rank, hint, capture, player):
        """Find the source pawn for a move."""
        files = 'hgfedcba'  # reverse order
        
        if capture:
            # For captures, hint is the file of the pawn
            source_file = files.index(hint)
            source_rank = dest_rank + player
            return (source_rank, source_file), None
        else:
            # For non-captures, check one or two squares back based on player
            pawn_plane = 0 if player == -1 else 6
            one_step = dest_rank + player
            two_step = dest_rank + (2 * player)
            
            # Check one step first
            if self.board[pawn_plane, one_step, dest_file] == 1:
                return (one_step, dest_file), None
            
            # Then check two steps (for initial pawn moves)
            return (two_step, dest_file), None

    def _find_piece_with_hint(self, piece_idx, hint, ranks, files, dest_rank=None, dest_file=None, player=None):
        """
        Find a piece when a disambiguation hint is provided.
        Also verifies that the piece can legally make the move to the destination.
        
        Args:
            piece_idx: Index in self.piece_locations for the piece type
            hint: Disambiguation hint (file or rank)
            ranks: String of rank characters
            files: String of file characters
            dest_rank: Destination rank (required for move verification)
            dest_file: Destination file (required for move verification)
            player: Current player (-1 for white, 1 for black)
            
        Returns:
            Tuple of (source_position, index_in_piece_locations)
        """
        piece_locations = self.piece_locations[piece_idx]
        candidates = []
        
        if hint in ranks:
            # Hint is a rank
            hint_rank = int(hint) - 1
            for i, location in enumerate(piece_locations):
                if location[0] == hint_rank:
                    candidates.append((location, i))
        else:
            # Hint is a file
            hint_file = files.index(hint)
            for i, location in enumerate(piece_locations):
                if location[1] == hint_file:
                    candidates.append((location, i))
        
        # If no candidates match the hint, return None. This should not happen with a valid pgn
        if not candidates:
            return None, None
        
        # If there's only one candidate, return it
        if len(candidates) == 1:
            return candidates[0]
            
        # If we have destination coordinates, filter candidates by move legality
        if dest_rank is not None and dest_file is not None and player is not None:
            legal_candidates = []
            
            # The piece type determines how to check move legality
            piece_type = piece_idx % 5  # 0=N, 1=B, 2=R, 3=Q, 4=K
            
            for pos, idx in candidates:
                source_rank, source_file = pos
                
                # Check if the move is geometrically and legally possible
                if piece_type == 0:  # Knight
                    is_knight_move = (
                        (abs(source_rank - dest_rank) == 2 and abs(source_file - dest_file) == 1) or
                        (abs(source_rank - dest_rank) == 1 and abs(source_file - dest_file) == 2)
                    )
                    if is_knight_move:
                        legal_candidates.append((pos, idx))
                        
                elif piece_type == 1:  # Bishop
                    rank_diff = dest_rank - source_rank
                    file_diff = dest_file - source_file
                    is_diagonal = (abs(rank_diff) == abs(file_diff))
                    
                    if is_diagonal and self.is_legal_ray_move(source_rank, source_file, dest_rank, dest_file):
                        legal_candidates.append((pos, idx))
                        
                elif piece_type == 2:  # Rook
                    is_straight = (source_rank == dest_rank or source_file == dest_file)
                    
                    if is_straight and self.is_legal_ray_move(source_rank, source_file, dest_rank, dest_file):
                        legal_candidates.append((pos, idx))
                        
                elif piece_type == 3:  # Queen
                    rank_diff = dest_rank - source_rank
                    file_diff = dest_file - source_file
                    is_diagonal = (abs(rank_diff) == abs(file_diff))
                    is_straight = (rank_diff == 0 or file_diff == 0)
                    
                    if (is_diagonal or is_straight) and self.is_legal_ray_move(source_rank, source_file, dest_rank, dest_file):
                        legal_candidates.append((pos, idx))
                        
                elif piece_type == 4:  # King
                    rank_diff = abs(dest_rank - source_rank)
                    file_diff = abs(dest_file - source_file)
                    is_adjacent = (rank_diff <= 1 and file_diff <= 1)
                    
                    if is_adjacent:
                        legal_candidates.append((pos, idx))
            
            # Now check for pins among legal candidates
            if legal_candidates:
                for pos, idx in legal_candidates:
                    if not self.is_pinned(pos[1], pos[0], player):
                        return pos, idx
                
                # If all candidates are pinned, return the first one
                return legal_candidates[0]
            
            # No legal candidates found
            return None, None
        
        # Without destination coordinates, just return the first candidate
        # (This should not happen in practice as we always have destination in real usage)
        return candidates[0] if candidates else (None, None)

    def _find_knight(self, piece_idx, dest_file, dest_rank, player):
        """Find a knight that can move to the destination square."""
        candidates = []
        
        for i, pos in enumerate(self.piece_locations[piece_idx]):
            source_rank, source_file = pos
            
            # Check if the knight move is geometrically possible
            is_knight_move = (
                (abs(source_rank - dest_rank) == 2 and abs(source_file - dest_file) == 1) or
                (abs(source_rank - dest_rank) == 1 and abs(source_file - dest_file) == 2)
            )
            
            if is_knight_move:
                candidates.append((pos, i))
        
        # If only one candidate, return it
        if len(candidates) == 1:
            return candidates[0]
        
        # Multiple candidates - check for pins
        for pos, i in candidates:
            if not self.is_pinned(pos[1], pos[0], player):
                return pos, i
        
        # If all candidates are pinned, just return the first one
        # (should not happen in a legal position)
        return candidates[0] if candidates else (None, None)

    def _find_bishop(self, piece_idx, dest_file, dest_rank, player):
        """Find a bishop that can move to the destination square."""
        candidates = []
        
        for i, pos in enumerate(self.piece_locations[piece_idx]):
            source_rank, source_file = pos
            rank_diff = dest_rank - source_rank
            file_diff = dest_file - source_file
            
            # Check if move is diagonal
            is_diagonal = (abs(rank_diff) == abs(file_diff))
            
            if is_diagonal and self.is_legal_ray_move(source_rank, source_file, dest_rank, dest_file):
                candidates.append((pos, i))
        
        # If only one candidate, return it
        if len(candidates) == 1:
            return candidates[0]
        
        # Multiple candidates - check for pins
        for pos, i in candidates:
            if not self.is_pinned(pos[1], pos[0], player):
                return pos, i
        
        # If all candidates are pinned, just return the first one
        return candidates[0] if candidates else (None, None)

    def _find_rook(self, piece_idx, dest_file, dest_rank, player):
        """Find a rook that can move to the destination square."""
        candidates = []
        
        for i, pos in enumerate(self.piece_locations[piece_idx]):
            source_rank, source_file = pos
            
            # Check if move is horizontal or vertical
            is_straight = (source_rank == dest_rank or source_file == dest_file)
            
            if is_straight and self.is_legal_ray_move(source_rank, source_file, dest_rank, dest_file):
                candidates.append((pos, i))
        
        # If only one candidate, return it
        if len(candidates) == 1:
            return candidates[0]
        
        # Multiple candidates - check for pins
        for pos, i in candidates:
            if not self.is_pinned(pos[1], pos[0], player):
                return pos, i
        
        # If all candidates are pinned, just return the first one
        return candidates[0] if candidates else (None, None)

    def _find_queen(self, piece_idx, dest_file, dest_rank, player):
        """Find a queen that can move to the destination square."""
        # Optimization: If there's only one queen, return it immediately
        if len(self.piece_locations[piece_idx]) == 1:
            return self.piece_locations[piece_idx][0], 0
        
        candidates = []
        
        for i, pos in enumerate(self.piece_locations[piece_idx]):
            source_rank, source_file = pos
            rank_diff = dest_rank - source_rank
            file_diff = dest_file - source_file
            
            # Check if move is diagonal, horizontal, or vertical
            is_diagonal = (abs(rank_diff) == abs(file_diff))
            is_straight = (rank_diff == 0 or file_diff == 0)
            
            if (is_diagonal or is_straight) and self.is_legal_ray_move(source_rank, source_file, dest_rank, dest_file):
                candidates.append((pos, i))
        
        # If only one candidate, return it
        if len(candidates) == 1:
            return candidates[0]
        
        # Multiple candidates - check for pins
        for pos, i in candidates:
            if not self.is_pinned(pos[1], pos[0], player):
                return pos, i
        
        # If all candidates are pinned, just return the first one
        return candidates[0] if candidates else (None, None)
      
    def make_move(self, piece, hint, dest_rank, dest_file, promoted_to, capture, player)->np.ndarray:
        #just update the bitboard and piece locations
        #returns the policy tensor, 73x8x8 float32 because it will be used for training
        #called for all moves except castling
        pieces='PNBRQK'

        x = self.find_piece(piece,dest_file,dest_rank,hint,capture,player)
        source = None
        removal_index = None
        try:
            source, removal_index = x
        except:
            raise ValueError(f"Piece {piece} at {hint} cannot move to {dest_file}{dest_rank}")
        source_rank, source_file = source
        
        #first, remove the piece from its old location
        piece_index = pieces.index(piece)
        piece_plane = piece_index + (6 if player == 1 else 0)
        self.board[piece_plane, source_rank, source_file] = 0

        #next, remove the captured piece if necessary
        if capture:
            if player==-1:
                for i in range(6,12): #only need to check black pieces
                    if self.board[i,dest_rank,dest_file] == 1:
                        if i!=6:
                            self.piece_locations[i-2].remove((dest_rank,dest_file))
                        self.board[i,dest_rank,dest_file] = 0
                        break
            else:
                for i in range(6):
                    if self.board[i,dest_rank,dest_file] == 1:
                        if i!=0:
                            self.piece_locations[i-1].remove((dest_rank,dest_file))
                        self.board[i,dest_rank,dest_file] = 0
                        break
        
        #now branch based on pawn/other piece
        policy = np.zeros((73,8,8),dtype=np.float32)
        move_plane = move_type(source_rank,source_file,dest_rank,dest_file)
        policy[move_type(source_rank,source_file,dest_rank,dest_file),source_rank,source_file] = 1
        if piece=='P':
            #handle promotion
            if promoted_to:
                promoted_index = pieces.index(promoted_to)
                if promoted_index != 4:
                    #policy tensor is weird for underpromotions, uses last 9 planes
                    policy[move_plane,source_rank,source_file] = 0
                    direction_offsets = {0: 1, 1: 0, 3: 0, 4: 1, 5: 1, 7: 1}
                    move_plane = 63 + 3*(promoted_index-1) + direction_offsets[move_plane//7]
                    policy[move_plane,source_rank,source_file] = 1
                self.board[promoted_index + (6 if player == 1 else 0), dest_rank, dest_file] = 1
                self.piece_locations[promoted_index-1 + (5 if player == 1 else 0)].append((dest_rank,dest_file))
            else:
                self.board[piece_plane, dest_rank, dest_file] = 1
            
            #handle en passant
            if player==-1:
                if dest_rank == self.en_passant[0] and dest_file == self.en_passant[1]:
                    self.board[6, dest_rank-1, dest_file] = 0
            else:
                if dest_rank == self.en_passant[0] and dest_file == self.en_passant[1]:
                    self.board[0, dest_rank+1, dest_file] = 0
        
            #update en passant
            if self.en_passant != (-1,-1):
                self.board[16, self.en_passant[0], self.en_passant[1]] = 0
            self.en_passant = (-1,-1)
            if abs(source_rank - dest_rank) == 2:
                self.en_passant = (dest_rank + player, dest_file)
            
        else:
            self.board[piece_plane, dest_rank, dest_file] = 1
            self.piece_locations[piece_index-1 + (5 if player == 1 else 0)][removal_index] = (dest_rank,dest_file)

            if self.en_passant != (-1,-1):
                self.board[16, self.en_passant[0], self.en_passant[1]] = 0
            self.en_passant = (-1,-1)

        #update castling rights
        if piece == 'K':
            self.board[15, 0 if player == -1 else 7] = 0
        elif piece == 'R':
            if source_file == 0:
                self.board[15, 0 if player == -1 else 7, 2] = 0
            elif source_file == 7:
                self.board[15, 0 if player == -1 else 7, 6] = 0

        #update en passant plane
        if self.en_passant != (-1,-1):
            self.board[16, self.en_passant[0], self.en_passant[1]] = 1

        #update repetition planes
        hash = self.hash_position()
        if hash in self.board_appeared_once:
            if hash in self.board_appeared_twice:
                self.board[13] = np.full((8,8),1)
                self.board[14] = np.full((8,8),1)
            else:
                self.board[13] = np.full((8,8),1)
                self.board[14] = np.zeros((8,8))
                self.board_appeared_twice.add(hash)
        else:
            self.board[13] = np.zeros((8,8))
            self.board[14] = np.zeros((8,8))
            self.board_appeared_once.add(hash)

        #update active player
        self.board[12] = -self.board[12]

        #print the pgn notation for the move and the resulting board
        return policy
    
    def castle_queenside(self,player)->np.ndarray:
        #castling queenside
        #needs to be handled differently since it involves moving two pieces
        if player == -1:
            self.board[5,0,3] = 0
            self.board[5,0,5] = 1
            self.board[3,0,7] = 0
            self.board[3,0,4] = 1
            self.board[15,0,2] = 0
            self.board[15,0,6] = 0
            if self.en_passant != (-1,-1):
                self.board[16, self.en_passant[0], self.en_passant[1]] = 0
            self.en_passant = (-1,-1)
            self.board[13] = np.zeros((8,8)) #repetition is impossible after castling
            self.board[14] = np.zeros((8,8))
            self.board[12] = 1 #switch to black
            hash = self.hash_position()
            self.board_appeared_once.add(hash)
            policy = np.zeros((73,8,8),dtype=np.float32)
            policy[43,0,3] = 1
            self.piece_locations[4]=[(0,5)]
            if len(self.piece_locations[2]) == 2:
                self.piece_locations[2][1]=(0,4) #always the second rook
            else:
                self.piece_locations[2][0]=(0,4)
            return policy
        else:
            self.board[11,7,3] = 0
            self.board[11,7,5] = 1
            self.board[9,7,7] = 0
            self.board[9,7,4] = 1
            self.board[15,7,2] = 0
            self.board[15,7,6] = 0
            if self.en_passant != (-1,-1):
                self.board[16, self.en_passant[0], self.en_passant[1]] = 0
            self.en_passant = (-1,-1)
            self.board[13] = np.zeros((8,8))
            self.board[14] = np.zeros((8,8))
            self.board[12] = -1
            hash = self.hash_position()
            self.board_appeared_once.add(hash)
            policy = np.zeros((73,8,8),dtype=np.float32)
            policy[43,7,3] = 1
            self.piece_locations[9]=[(7,5)]
            if len(self.piece_locations[7]) == 2:
                self.piece_locations[7][1]=(7,4) #always the second rook
            else:
                self.piece_locations[7][0]=(7,4)
            return policy
        
    def castle_kingside(self,player)->np.ndarray:
        if player==-1:
            self.board[5,0,3] = 0
            self.board[5,0,1] = 1
            self.board[3,0,0] = 0
            self.board[3,0,2] = 1
            self.board[15,0,2] = 0
            self.board[15,0,6] = 0
            if self.en_passant != (-1,-1):
                self.board[16, self.en_passant[0], self.en_passant[1]] = 0
            self.en_passant = (-1,-1)
            self.board[13] = np.zeros((8,8))
            self.board[14] = np.zeros((8,8))
            self.board[12] = 1
            hash = self.hash_position()
            self.board_appeared_once.add(hash)
            policy=np.zeros((73,8,8),dtype=np.float32)
            policy[15,0,3] = 1
            self.piece_locations[4]=[(0,1)]
            self.piece_locations[2][0]=(0,2) #always the first rook
            return policy
        else:
            self.board[11,7,3] = 0
            self.board[11,7,1] = 1
            self.board[9,7,0] = 0
            self.board[9,7,2] = 1
            self.board[15,7,2] = 0
            self.board[15,7,6] = 0
            if self.en_passant != (-1,-1):
                self.board[16, self.en_passant[0], self.en_passant[1]] = 0
            self.en_passant = (-1,-1)
            self.board[13] = np.zeros((8,8))
            self.board[14] = np.zeros((8,8))
            self.board[12] = -1
            hash = self.hash_position()
            self.board_appeared_once.add(hash)
            policy = np.zeros((73,8,8),dtype=np.float32)
            policy[15,7,3] = 1
            self.piece_locations[9]=[(7,1)]
            self.piece_locations[7][0]=(7,2)
            return policy

    def display(self):
        #display the board
        pieces='PNBRQKpnbrqk'
        for i in range(8):
            for j in range(8):
                piece = pieces[self.board[:12,i,j].argmax()]
                if self.board[:12,i,j].max() == 0:
                    print('.', end=' ')
                else:
                    print(piece, end=' ')
            print()
        print()

    def check_consistency(self):
        #checks if piece_locations matches the bitboard
        #also checks for piece overlaps

        pieces='PNBRQK'

        #first, check if piece_locations matches the bitboard
        #don't check pawns because they're handled differently
        for i in range(1,6):
            for piece in self.piece_locations[i-1]:
                assert self.board[i,piece[0],piece[1]] == 1, f"White {pieces[i]} at {piece} not found in bitboard"
        for i in range(7,12):
            for piece in self.piece_locations[i-2]:
                assert self.board[i,piece[0],piece[1]] == 1, f"Black {pieces[i-6]} at {piece} not found in bitboard"

        #next, check for extra pieces on the bitboard
        for i in range(1,6):
            for j in range(8):
                for k in range(8):
                    if self.board[i,j,k] == 1:
                        assert (j,k) in self.piece_locations[i-1], f"Extra white {pieces[i]} at {(j,k)}"
        for i in range(7,12):
            for j in range(8):
                for k in range(8):
                    if self.board[i,j,k] == 1:
                        assert (j,k) in self.piece_locations[i-2], f"Extra black {pieces[i-6]} at {(j,k)}"

        #finally, check for overlaps
        #include pawns this time
        occupied_squares = set()
        for i in range(12):
            for j in range(8):
                for k in range(8):
                    if self.board[i,j,k] == 1:
                        assert (j,k) not in occupied_squares, f"Overlap at {(j,k)}"
                        occupied_squares.add((j,k))

def process_move(PGNboard: FastPGNBoard, move: str, player: int, verbose=False, debug=False) -> Tuple[np.ndarray, np.ndarray]:
    #processes a move in PGN notation, updating the board and returning the data needed for training
    #returns: the 17x8x8 bitboard and the 73x8x8 policy
    files = 'hgfedcba' #files are reversed because of how we represent the board
    ranks = '12345678'
    starting_board = PGNboard.board.copy()
    policy = np.zeros((73, 8, 8), dtype=np.int8)

    if move.startswith('O-O-O'):
        #castling queenside
        policy = PGNboard.castle_queenside(player)
    elif move.startswith('O-O'):
        #castling kingside
        policy = PGNboard.castle_kingside(player)
    else:
        move = move.rstrip('+#')
        
        piece = None
        dest = None
        promoted_to = None
        capture = False
        hint = None

        if '=' in move:
            #pawn promotion
            piece, dest, promoted_to = 'P', move[-4:-2], move[-1]
            if 'x' in move:
                capture = True
                hint = move[0]
        elif len(move) == 2:
            #pawn move
            piece, dest = 'P', move
        elif move[0] in files:
            #pawn capture
            piece, dest = 'P', move[-2:]
            capture = True
            hint = move[0]
        else:
            if 'x' in move:
                capture = True
                move = move.replace('x', '')
            if len(move) == 3:
                piece, dest = move[0], move[1:]
            else:
                piece, dest = move[0], move[2:]
                hint = move[1]

        dest_file, dest_rank = files.index(dest[0]), ranks.index(dest[1])
        policy = PGNboard.make_move(piece, hint, dest_rank, dest_file, promoted_to, capture, player)

    if verbose:
        print(f"Move: {move}")
        PGNboard.display()

    if debug:
        PGNboard.check_consistency()

    return starting_board, policy


def read_pgn_to_bitboard_and_policy(file_path: str, verbose=False, debug=False) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
    # returns a list of bitboards, policies, and the result of the game
    with open(file_path, 'r') as file:
        pgn_content = file.read()
        boards = []
        policies = []
        PGNboard = FastPGNBoard()
        player = -1
        moves = re.findall(r'\b(\d+\.)\s*([^\s.]+)(?:\s+([^\s.]+))?', pgn_content)
        moves_taken = 0
        for _, white_move, black_move in moves:
            if white_move:
                board, policy = process_move(PGNboard, white_move, player, verbose, debug)
                boards.append(board)
                policies.append(policy)
                player = 1
            if black_move:
                if black_move[0] == "0" or black_move[0] == "1":
                    #this is the result of the game
                    break
                board, policy = process_move(PGNboard, black_move, player, verbose, debug)
                boards.append(board)
                policies.append(policy)
                player = -1
            moves_taken += 1
        
        #find the result of the game from the end of the PGN
        result = pgn_content.split(' ')[-3] #this dataset has some ending tags to ignore
        if result == '1-0':
            result = 1
        elif result == '0-1':
            result = -1
        else:
            result = 0

        return boards, policies, result
    
def read_condensed_pgn_to_bitboard_and_policy(file_path, verbose=False, debug=True):
    #reads a condensed PGN file and returns the bitboards and policies
    #this is for the dataset of condensed games
    with open(file_path, 'r') as file:
        #read file line by line
        lines = file.readlines()
        boards = []
        policies = []
        results = []
        PGNboard = FastPGNBoard()
        player = -1
        positions_processed = 0
        games_processed = 0
        #each line is either empty or a full PGN game
        if verbose:
            print(f"Processing {file_path}")
    for line in lines:
        try:
            #check if line is empty after removing whitespace
            if not line.strip():
                continue
            #process the PGN game
            #looks very similar to the previous function
            moves = re.findall(r'\b(\d+\.)\s*([^\s.]+)(?:\s+([^\s.]+))?', line)
            boards_added = 0
            new_boards = []
            new_policies = []
            new_results = []
            for _, white_move, black_move in moves:
                if white_move:
                    board, policy = process_move(PGNboard, white_move, player, verbose, debug)
                    new_boards.append(board)
                    new_policies.append(policy)
                    player = 1
                    boards_added += 1
                    positions_processed += 1
                if black_move:
                    if black_move[0] == "0" or black_move[0] == "1":
                        #this is the result of the game
                        break
                    board, policy = process_move(PGNboard, black_move, player, verbose, debug)
                    new_boards.append(board)
                    new_policies.append(policy)
                    player = -1
                    boards_added += 1
                    positions_processed += 1

            #find the result of the game from the end of the PGN
            result = line.split(' ')[-3] #this dataset has some ending tags to ignore
            if result == '1-0':
                result = 1
            elif result == '0-1':
                result = -1
            else:
                result = 0

            for i in range(boards_added):
                new_results.append(result)

            boards += new_boards
            policies += new_policies
            results += new_results

            if positions_processed > 20000:
                yield boards, policies, results
                boards = []
                policies = []
                results = []
                positions_processed = 0

        except Exception as e:
            # Replace the generic error handling with this more detailed version
            print(f"Error processing game {games_processed+1}: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"Problematic line: {line[:100]}...")  # Print the first 100 chars
            print()

            # Log game number and error to help with debugging
            if not hasattr(PGNboard, 'processing_errors'):
                PGNboard.processing_errors = []
            PGNboard.processing_errors.append({
                'game_number': games_processed + 1,
                'error': str(e),
                'line_preview': line[:100]
            })

        #reset the board for the next game
        PGNboard.reset()
        player = -1

        games_processed += 1

    return boards, policies, results
    
if __name__ == "__main__":
    #there are some errors in the dataset - find them and see if it's a problem with the code or the dataset"
    #also, check if the code is working as intended"

    #condensed games are in condensed-pgns/condensed_standard(n).pgn

    #let's test the processing speed, error rate, and amount of data generated per game!

    import time
    total_time = 0
    total_data = 0

    for i in range(150,197):
        start = time.time()
        file_path = f'condensed-pgns/condensed_standard{i}.pgn'
        boards, policies, results = read_condensed_pgn_to_bitboard_and_policy(file_path, verbose=False, debug=False)
        print(f"Processed {file_path}")
        end = time.time()
        total_time += end-start
        #calculate the amount of data in boards, policies, and result in bits
        data = 0
        for board in boards:
            data += board.nbytes
        for policy in policies:
            data += policy.nbytes
        for result in results:
            data += result.bit_length()
        total_data += data

    print("Processed 15000 games")
    print(f"Total time: {total_time} seconds")
    print(f"Total data: {total_data/1000000000} GB")
    print(f"Average time per game: {total_time/15000} seconds")
    print(f"Average data per game: {total_data/15000} bytes")
        
    print("Extrapolating to 1000000 games...")
    print(f"Total time: {total_time*200/3} seconds")
    print(f"Total data: {total_data/15000000} GB")