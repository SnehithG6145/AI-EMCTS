import numpy as np
import random

class KTK:
    """
    Kill The King (KTK) game environment - Simplified version
    A competitive multi-agent game where each player controls units to kill the opponent's king
    """
    # Class variable to track if initialization message has been printed
    _init_message_printed = False
    
    def __init__(self, board_size=4, max_turns=20, random_setup=True):
        """
        Initialize the KTK environment
        
        Parameters:
        - board_size (int): Size of the board (default: 4)
        - max_turns (int): Maximum number of turns before counting pieces (default: 20)
        - random_setup (bool): Whether to randomize the initial setup (default: True)
        """
        self.size = board_size  # Board size (default: 4x4)
        self.unit_types = ["King", "Warrior", "Archer", "Healer"]
        self.current_unit_type_idx = 0
        self.player = 0  # Current player (0 or 1)
        self.turn_count = 0  # Track number of turns
        self.max_turns = max_turns  # Maximum number of turns before counting pieces
        
        # Initialize state as a simple 2D array with integers
        # 0: Empty, 1-4: Player 0 units, 5-8: Player 1 units
        # 1/5: King, 2/6: Warrior, 3/7: Archer, 4/8: Healer
        self.state = np.zeros((self.size, self.size), dtype=int)
        
        if random_setup:
            self._random_setup()
        else:
            self._default_setup()
        
        # Track alive units (no HP - units die immediately when attacked)
        self.alive = {i: True for i in range(1, 9)}
        
        # Track winner
        self.winner = None
        
        # Print initialization message only once
        self._is_copy = False
        if not KTK._init_message_printed and not self._is_copy:
            print(f"KTK Environment initialized with:")
            print(f"- Board size: {self.size}x{self.size}")
            print(f"- Max turns: {self.max_turns}")
            print(f"- Units per player: {len(self.unit_types)}")
            print(f"- Random setup: {random_setup}")
            KTK._init_message_printed = True

    def _default_setup(self):
        """Set up the default initial positions"""
        # Player 0 units - positioned in the top-left corner
        self.state[0, 0] = 1  # King
        self.state[0, 1] = 3  # Archer
        self.state[1, 0] = 2  # Warrior
        self.state[1, 1] = 4  # Healer
        
        # Player 1 units - positioned in the bottom-right corner
        self.state[self.size-1, self.size-1] = 5  # King
        self.state[self.size-1, self.size-2] = 7  # Archer
        self.state[self.size-2, self.size-1] = 6  # Warrior
        self.state[self.size-2, self.size-2] = 8  # Healer

    def _random_setup(self):
        """Set up random initial positions while maintaining some strategic balance"""
        # Create empty positions for each player's territory
        player0_positions = []
        player1_positions = []
        
        # Define territories (roughly 1/3 of the board for each player)
        p0_territory_size = max(2, self.size // 3)
        p1_territory_size = max(2, self.size // 3)
        
        # Generate positions for player 0 (top-left area)
        for i in range(p0_territory_size):
            for j in range(p0_territory_size):
                player0_positions.append((i, j))
                
        # Generate positions for player 1 (bottom-right area)
        for i in range(self.size - p1_territory_size, self.size):
            for j in range(self.size - p1_territory_size, self.size):
                player1_positions.append((i, j))
        
        # Shuffle positions
        random.shuffle(player0_positions)
        random.shuffle(player1_positions)
        
        # Place player 0 units
        if player0_positions:
            # King should be in a safer position (further from center)
            king_positions = [pos for pos in player0_positions if pos[0] + pos[1] <= p0_territory_size]
            if king_positions:
                king_pos = random.choice(king_positions)
                player0_positions.remove(king_pos)
                self.state[king_pos[0], king_pos[1]] = 1  # King
            else:
                king_pos = player0_positions.pop(0)
                self.state[king_pos[0], king_pos[1]] = 1  # King
            
            # Place other units
            for unit_id in range(2, 5):
                if player0_positions:
                    pos = player0_positions.pop(0)
                    self.state[pos[0], pos[1]] = unit_id
                else:
                    # Find an empty spot if we ran out of predefined positions
                    self._place_unit_in_empty_spot(unit_id, 0, p0_territory_size)
        else:
            # Fallback to default if no positions available
            self._default_setup()
            return
            
        # Place player 1 units
        if player1_positions:
            # King should be in a safer position (further from center)
            king_positions = [pos for pos in player1_positions 
                             if pos[0] + pos[1] >= 2 * self.size - p1_territory_size - 1]
            if king_positions:
                king_pos = random.choice(king_positions)
                player1_positions.remove(king_pos)
                self.state[king_pos[0], king_pos[1]] = 5  # King
            else:
                king_pos = player1_positions.pop(0)
                self.state[king_pos[0], king_pos[1]] = 5  # King
            
            # Place other units
            for unit_id in range(6, 9):
                if player1_positions:
                    pos = player1_positions.pop(0)
                    self.state[pos[0], pos[1]] = unit_id
                else:
                    # Find an empty spot if we ran out of predefined positions
                    self._place_unit_in_empty_spot(unit_id, self.size - p1_territory_size, self.size)
        else:
            # Fallback to default if no positions available
            self._default_setup()

    def _place_unit_in_empty_spot(self, unit_id, min_idx, max_idx):
        """Place a unit in an empty spot within the given range"""
        empty_spots = []
        for i in range(min_idx, max_idx):
            for j in range(min_idx, max_idx):
                if self.state[i, j] == 0:
                    empty_spots.append((i, j))
        
        if empty_spots:
            pos = random.choice(empty_spots)
            self.state[pos[0], pos[1]] = unit_id
        else:
            # Last resort: place anywhere on the board
            for i in range(self.size):
                for j in range(self.size):
                    if self.state[i, j] == 0:
                        self.state[i, j] = unit_id
                        return

    def get_state(self):
        """Return a copy of the current state"""
        return self.state.copy(), self.alive.copy(), self.player, self.current_unit_type_idx

    def copy(self):
        """Create a copy of the environment"""
        new_env = KTK(board_size=self.size, max_turns=self.max_turns, random_setup=False)
        new_env._is_copy = True  # Mark as a copy to avoid printing initialization message
        new_env.state = self.state.copy()
        new_env.alive = self.alive.copy()
        new_env.player = self.player
        new_env.current_unit_type_idx = self.current_unit_type_idx
        new_env.winner = self.winner
        new_env.turn_count = self.turn_count
        return new_env

    def get_unit_type(self, unit_id):
        """Get the type of a unit from its ID"""
        if unit_id == 0:
            return "Empty"
        elif unit_id in [1, 5]:
            return "King"
        elif unit_id in [2, 6]:
            return "Warrior"
        elif unit_id in [3, 7]:
            return "Archer"
        elif unit_id in [4, 8]:
            return "Healer"
        return "Unknown"
    
    def get_unit_owner(self, unit_id):
        """Get the owner of a unit from its ID"""
        if unit_id == 0:
            return -1
        return 0 if unit_id < 5 else 1

    def get_possible_actions(self):
        """Get all possible actions for the current player and unit type"""
        actions = []
        current_unit_type = self.unit_types[self.current_unit_type_idx]
        unit_id_base = 1 if self.player == 0 else 5
        
        # Find the unit ID for the current player and unit type
        if current_unit_type == "King":
            unit_id = unit_id_base
        elif current_unit_type == "Warrior":
            unit_id = unit_id_base + 1
        elif current_unit_type == "Archer":
            unit_id = unit_id_base + 2
        elif current_unit_type == "Healer":
            unit_id = unit_id_base + 3
        
        # Find the unit position
        unit_pos = None
        for i in range(self.size):
            for j in range(self.size):
                if self.state[i, j] == unit_id:
                    unit_pos = (i, j)
                    break
            if unit_pos:
                break
        
        # If unit not found or dead, return wait action
        if not unit_pos or not self.alive[unit_id]:
            return [(unit_id, "wait", None)]
        
        i, j = unit_pos
        
        # Movement actions
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.size and 0 <= nj < self.size and self.state[ni, nj] == 0:
                actions.append((unit_id, "move", (ni, nj)))
        
        # Attack actions for Warrior and Archer
        if current_unit_type in ["Warrior", "Archer", "King"]:  # Allow King to attack too
            attack_range = 1 if current_unit_type in ["Warrior", "King"] else 2
            for ti in range(max(0, i - attack_range), min(self.size, i + attack_range + 1)):
                for tj in range(max(0, j - attack_range), min(self.size, j + attack_range + 1)):
                    if (ti != i or tj != j) and self.state[ti, tj] > 0:
                        target_id = self.state[ti, tj]
                        if self.get_unit_owner(target_id) != self.player:
                            # Prioritize attack actions by giving them a higher weight
                            actions.append((unit_id, "attack", target_id))
                            # Add duplicate attack actions to increase probability of selection
                            if current_unit_type == "Warrior":  # Warriors are more aggressive
                                actions.append((unit_id, "attack", target_id))
                                actions.append((unit_id, "attack", target_id))
        
        # Heal actions for Healer - now revives dead units
        if current_unit_type == "Healer":
            for ti in range(max(0, i - 1), min(self.size, i + 2)):
                for tj in range(max(0, j - 1), min(self.size, j + 2)):
                    if (ti != i or tj != j) and self.state[ti, tj] > 0:
                        target_id = self.state[ti, tj]
                        if self.get_unit_owner(target_id) == self.player and not self.alive[target_id]:
                            actions.append((unit_id, "heal", target_id))
        
        # Wait action - only add if no other actions are available
        if not actions:
            actions.append((unit_id, "wait", None))
        
        return actions

    def step(self, action):
        """Execute an action and return the new state and whether the game is done"""
        unit_id, action_type, target = action
        
        # Find the unit position
        unit_pos = None
        for i in range(self.size):
            for j in range(self.size):
                if self.state[i, j] == unit_id:
                    unit_pos = (i, j)
                    break
            if unit_pos:
                break
        
        # If unit not found or dead, or wrong player's unit, skip action
        if not unit_pos or not self.alive[unit_id] or self.get_unit_owner(unit_id) != self.player:
            pass  # Skip action
        else:
            i, j = unit_pos
            
            # Execute the action
            if action_type == "move" and target:
                ti, tj = target
                if 0 <= ti < self.size and 0 <= tj < self.size and self.state[ti, tj] == 0:
                    self.state[ti, tj] = self.state[i, j]
                    self.state[i, j] = 0
            
            elif action_type == "attack" and target:
                # Find target position
                target_pos = None
                for ti in range(self.size):
                    for tj in range(self.size):
                        if self.state[ti, tj] == target:
                            target_pos = (ti, tj)
                            break
                    if target_pos:
                        break
                
                if target_pos:
                    # Instant death on attack
                    self.alive[target] = False
                    # Remove the unit from the board
                    self.state[target_pos[0], target_pos[1]] = 0
            
            elif action_type == "heal" and target:
                # Find target position
                target_pos = None
                for ti in range(self.size):
                    for tj in range(self.size):
                        if self.state[ti, tj] == target:
                            target_pos = (ti, tj)
                            break
                    if target_pos:
                        break
                
                if target_pos:
                    # Revive the unit
                    self.alive[target] = True
        
        # Check if game is done
        if not self.alive[1]:
            self.winner = 1
        elif not self.alive[5]:
            self.winner = 0
        
        # Move to next unit or player
        self.current_unit_type_idx = (self.current_unit_type_idx + 1) % len(self.unit_types)
        if self.current_unit_type_idx == 0:
            self.player = 1 - self.player
            self.turn_count += 1
            
            # Check if max turns reached
            if self.turn_count >= self.max_turns and not self.winner:
                # Count pieces to determine winner
                player0_pieces = sum(1 for i in range(1, 5) if self.alive[i])
                player1_pieces = sum(1 for i in range(5, 9) if self.alive[i])
                
                if player0_pieces > player1_pieces:
                    self.winner = 0
                elif player1_pieces > player0_pieces:
                    self.winner = 1
                # If equal, game continues
        
        return self, self.is_done()

    def display(self, show_kings_only=False):
        """Display the current state of the game"""
        unit_symbols = {
            0: ".",
            1: "K", 2: "W", 3: "A", 4: "H",  # Player 0
            5: "k", 6: "w", 7: "a", 8: "h"   # Player 1
        }
        
        print("  " + " ".join([str(i) for i in range(self.size)]))
        for i in range(self.size):
            row = [unit_symbols[self.state[i, j]] for j in range(self.size)]
            print(f"{i} " + " ".join(row))
        
        # Display unit status
        if show_kings_only:
            print("\nKings Status:")
            print(f"Player 0 King: {'Alive' if self.alive[1] else 'Dead'}")
            print(f"Player 1 King: {'Alive' if self.alive[5] else 'Dead'}")
        else:
            print("\nUnit Status:")
            for unit_id in range(1, 9):
                unit_type = self.get_unit_type(unit_id)
                player = self.get_unit_owner(unit_id)
                status = "Alive" if self.alive[unit_id] else "Dead"
                print(f"Player {player} {unit_type} (ID: {unit_id}): {status}")
        
        print(f"\nCurrent Player: {self.player}")
        print(f"Current Unit Type: {self.unit_types[self.current_unit_type_idx]}")
        print(f"Turn: {self.turn_count}/{self.max_turns}")

    def is_done(self):
        """Check if the game is done (a king is dead or max turns reached with piece count)"""
        return self.winner is not None
