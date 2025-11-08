import os
import uuid
import time
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque
from copy import deepcopy
from case_closed_game import Game, Direction, GameResult

# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

game_lock = Lock()
 
PARTICIPANT = "ParticipantX"
AGENT_NAME = "AgentX"

# Initialize the minimax agent globally
minimax_agent = None

""" 
Minimax Agent with Alpha-Beta Pruning and Voronoi Calculation for optimized performance on a CPU
"""

class GameStateSimulator:
    def __init__(self, board_grid, agent1_trail, agent2_trail, agent1_dir: str, agent2_dir: str, 
                 agent1_boosts: int, agent2_boosts: int, agent1_alive: bool, agent2_alive: bool):

        self.board = [row[:] for row in board_grid]  # Deep copy grid
        self.agent1_trail = deque(agent1_trail)  # Deep copy trails
        self.agent2_trail = deque(agent2_trail)  # Deep copy trails
        self.agent1_dir = agent1_dir
        self.agent2_dir = agent2_dir
        self.agent1_boosts = agent1_boosts  
        self.agent2_boosts = agent2_boosts 
        self.agent1_alive = agent1_alive
        self.agent2_alive = agent2_alive
        self.width = 20
        self.height = 18
    
    def _torus_check(self, position):
        """Wrap coordinates around board edges"""
        x, y = position
        return (x % self.width, y % self.height)
    
    def _str_to_direction(self, dir_str):
        """Convert string direction to (dx, dy) tuple"""
        direction_map = {
            'UP': (0, -1),
            'DOWN': (0, 1),
            'LEFT': (-1, 0),
            'RIGHT': (1, 0)
        }
        return direction_map.get(dir_str.upper(), (1, 0))
    
    def _is_opposite(self, current_dir, new_dir):
        """Check if new direction is opposite to current"""
        cur_dx, cur_dy = self._str_to_direction(current_dir)
        new_dx, new_dy = self._str_to_direction(new_dir)
        return (new_dx, new_dy) == (-cur_dx, -cur_dy)

    def simulate_move(self, p1_dir, p2_dir, p1_boost=False, p2_boost=False):
        """Simulate a move and return new state"""
        # Create a deep copy for the new state
        new_state = GameStateSimulator(
            [row[:] for row in self.board],
            list(self.agent1_trail),
            list(self.agent2_trail),
            self.agent1_dir,
            self.agent2_dir,
            self.agent1_boosts,
            self.agent2_boosts,
            self.agent1_alive,
            self.agent2_alive
        )
        
        # Simulate agent 1 move
        if new_state.agent1_alive:
            # Check for opposite direction (invalid move)
            if self._is_opposite(new_state.agent1_dir, p1_dir):
                p1_dir = new_state.agent1_dir  # Keep current direction
            
            # Handle boost
            num_moves = 2 if (p1_boost and new_state.agent1_boosts > 0) else 1
            if p1_boost and new_state.agent1_boosts > 0:
                new_state.agent1_boosts -= 1
            
            for _ in range(num_moves):
                if not new_state.agent1_alive:
                    break
                    
                head = new_state.agent1_trail[-1]
                dx, dy = self._str_to_direction(p1_dir)
                new_head = self._torus_check((head[0] + dx, head[1] + dy))
                
                # Check collision with any trail
                if new_head in new_state.agent1_trail or new_head in new_state.agent2_trail:
                    new_state.agent1_alive = False
                    break
                
                # Add to trail
                new_state.agent1_trail.append(new_head)
                new_state.board[new_head[1]][new_head[0]] = 1
            
            new_state.agent1_dir = p1_dir
        
        # Simulate agent 2 move
        if new_state.agent2_alive:
            # Check for opposite direction (invalid move)
            if self._is_opposite(new_state.agent2_dir, p2_dir):
                p2_dir = new_state.agent2_dir  # Keep current direction
            
            # Handle boost
            num_moves = 2 if (p2_boost and new_state.agent2_boosts > 0) else 1
            if p2_boost and new_state.agent2_boosts > 0:
                new_state.agent2_boosts -= 1
            
            for _ in range(num_moves):
                if not new_state.agent2_alive:
                    break
                    
                head = new_state.agent2_trail[-1]
                dx, dy = self._str_to_direction(p2_dir)
                new_head = self._torus_check((head[0] + dx, head[1] + dy))
                
                # Check collision with any trail
                if new_head in new_state.agent1_trail or new_head in new_state.agent2_trail:
                    new_state.agent2_alive = False
                    break
                
                # Add to trail
                new_state.agent2_trail.append(new_head)
                new_state.board[new_head[1]][new_head[0]] = 1
            
            new_state.agent2_dir = p2_dir
        
        return new_state


class VoronoiCalculator:
    def __init__(self, width=20, height=18):
        self.width = width
        self.height = height

    def calculate_territories(self, board, p1_head, p2_head, p1_boosts, p2_boosts):
        """
        BFS from both heads simultaneously, account for boost distance
        Returns (p1_cells, p2_cells, contested_cells)
        """
        visited = {}
        queue = deque()

        # Initialize with cells adjacent to heads (not heads themselves, which are trails)
        p1_territory = set()
        p2_territory = set()
        contested = set()
        
        # Start from adjacent positions to heads
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            p1_adj = ((p1_head[0] + dx) % self.width, (p1_head[1] + dy) % self.height)
            p2_adj = ((p2_head[0] + dx) % self.width, (p2_head[1] + dy) % self.height)
            queue.append((p1_adj, 1, 1, p1_boosts > 0))  # Start at distance 1
            queue.append((p2_adj, 2, 1, p2_boosts > 0))

        while queue:
            pos, player, dist, has_boost = queue.popleft()

            # Torus wrapping
            x, y = pos[0] % self.width, pos[1] % self.height
            pos = (x, y)

            # Weight: with boost, effective distance is halved
            effective_dist = dist * 0.5 if has_boost else dist
            
            # Skip occupied cells (trails)
            if board[y][x] == 1:
                continue
            
            # Skip if already visited with better distance
            if pos in visited:
                prev_player, prev_dist = visited[pos]
                if prev_dist < effective_dist:
                    continue  # Already claimed by someone closer
                elif prev_dist == effective_dist and prev_player != player:
                    # Same distance - contested
                    contested.add(pos)
                    if pos in p1_territory:
                        p1_territory.remove(pos)
                    if pos in p2_territory:
                        p2_territory.remove(pos)
                continue

            visited[pos] = (player, effective_dist)

            if player == 1:
                p1_territory.add(pos)
            else:
                p2_territory.add(pos)

            # Expand to neighbors (limit search depth to avoid filling entire board)
            if dist < 15:  # Only expand if not too deep
                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                    next_pos = (x + dx, y + dy)
                    queue.append((next_pos, player, dist + 1, has_boost))
        return len(p1_territory), len(p2_territory), len(contested)

class MinimaxAgent:
    def __init__(self, max_depth: int, time_limit: float):
        self.max_depth = max_depth
        self.time_limit=time_limit
        self.voronoi = VoronoiCalculator()
        self.nodes_searched = 0
        self.start_time = None

    def evaluate_state(self, state, player_num):
        """
        Heuristic evaluation function
        Returns score from perspective of player_num
        """
        # Terminal states
        if not state.agent1_alive and not state.agent2_alive:
            return 0  # Draw
        if not state.agent1_alive:
            return -10000 if player_num == 1 else 10000  # Agent1 lost
        if not state.agent2_alive:
            return 10000 if player_num == 1 else -10000  # Agent2 lost
        
        # Voronoi territory
        p1_head = state.agent1_trail[-1]
        p2_head = state.agent2_trail[-1]
        p1_terr, p2_terr, contested = self.voronoi.calculate_territories(
            state.board, p1_head, p2_head, 
            state.agent1_boosts, state.agent2_boosts
        )

        # Calculate game phase (0.0 = early, 1.0 = late)
        total_cells = 360  # 20 * 18
        occupied_cells = len(state.agent1_trail) + len(state.agent2_trail)
        game_phase = min(occupied_cells / total_cells, 1.0)
        
        # Score Components
        territory_diff = p1_terr - p2_terr
        trail_diff = len(state.agent1_trail) - len(state.agent2_trail)
        boost_diff = state.agent1_boosts - state.agent2_boosts

        # Dynamic weights based on game phase
        territory_w = 12 + (28 * game_phase)    # 12 early → 40 late (territory becomes critical)
        trail_w = 2 + (10 * game_phase)         # 2 early → 12 late (length matters for tiebreakers)
        boost_w = 45 - (30 * game_phase)        # 45 early → 15 late (save early, use late)
        contested_w = 2 + (8 * game_phase)      # 2 early → 10 late (positioning fights matter)

        # Debug first evaluation
        if self.nodes_searched == 2:
            print(f"    [EVAL DEBUG] phase={game_phase:.2f}, p1_terr={p1_terr}, p2_terr={p2_terr}, contested={contested}")
            print(f"    [EVAL DEBUG] weights: terr={territory_w:.1f}, trail={trail_w:.1f}, boost={boost_w:.1f}, cont={contested_w:.1f}")

        # Weighted score - dynamic based on game phase
        score = (
            territory_diff * territory_w     # Territory control (scales up)
            + trail_diff * trail_w           # Trail length = survival time (scales up)
            + boost_diff * boost_w           # Boosts are tools (scales down)
            + contested * contested_w        # Positioning advantage (scales up)
        )

        return score if player_num == 1 else -score

    def get_valid_moves(self, state, player_num):
        """Get valid moves (excluding opposite direction)"""
        all_moves = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        current_dir = state.agent1_dir if player_num == 1 else state.agent2_dir
        
        opposite_map = {
            'UP': 'DOWN',
            'DOWN': 'UP',
            'LEFT': 'RIGHT',
            'RIGHT': 'LEFT'
        }
        
        opposite = opposite_map.get(current_dir)
        valid_moves = [m for m in all_moves if m != opposite]
        
        return valid_moves if valid_moves else all_moves  # Fallback to all if something's wrong
    
    def minimax(self, state, depth, alpha, beta, maximizing_player, player_num):
        """
        Mimimax with alpha-beta pruning
        maximizing_player: True if we're maximizing for player_num
        """
        self.nodes_searched += 1
        
        # Timeout check
        if time.time() - self.start_time > self.time_limit:
            return self.evaluate_state(state, player_num), None
        
        # Terminal conditions
        if depth == 0 or not state.agent1_alive or not state.agent2_alive:
            return self.evaluate_state(state, player_num), None

        # Get valid moves (excluding opposite direction)
        my_moves = self.get_valid_moves(state, player_num)
        opp_player = 3 - player_num  # 1->2, 2->1
        opp_moves = self.get_valid_moves(state, opp_player)
        best_move = my_moves[0]  # default to first valid move

        if maximizing_player:
            max_eval = float('-inf')
            for move in my_moves:
                # Simulate with/without boost
                for use_boost in [False, True]:
                    if use_boost and (state.agent1_boosts if player_num == 1 else state.agent2_boosts) <= 0:
                        continue

                    # For this move, find worst-case opponent response
                    move_eval = float('inf')  # We want the MINIMUM over opponent moves
                    
                    for opp_move in opp_moves:
                        if player_num == 1:
                            new_state = state.simulate_move(move, opp_move, use_boost, False)
                        else:
                            new_state = state.simulate_move(opp_move, move, False, use_boost)
                        
                        eval_score, _ = self.minimax(new_state, depth - 1, alpha, beta, False, player_num)
                        move_eval = min(move_eval, eval_score)  # Opponent picks worst for us
                    
                    # Debug at top level
                    if depth == self.max_depth:
                        boost_str = ":BOOST" if use_boost else ""
                        print(f"  {move}{boost_str} -> eval={move_eval}")
                    
                    # Now check if this move (with worst-case opponent response) is best
                    if move_eval > max_eval:
                        max_eval = move_eval
                        best_move = f"{move}:BOOST" if use_boost else move

                    alpha = max(alpha, move_eval)
                    if beta <= alpha:
                        break  # Beta Cutoff
                    
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in opp_moves:  # Opponent's valid moves
                for use_boost in [False, True]:
                    # For this opponent move, find best response from our side
                    move_eval = float('-inf')  # We want the MAXIMUM over our responses
                    
                    for opp_move in my_moves:  # Our valid moves as response
                        if player_num == 1:
                            new_state = state.simulate_move(opp_move, move, False, use_boost)
                        else:
                            new_state = state.simulate_move(move, opp_move, use_boost, False)
                        
                        eval_score, _ = self.minimax(new_state, depth - 1, alpha, beta, True, player_num)
                        move_eval = max(move_eval, eval_score)  # We pick best for us

                    # Check if this opponent move (with best response) is worst for us
                    if move_eval < min_eval:
                        min_eval = move_eval
                        
                    beta = min(beta, move_eval)
                    if beta <= alpha:
                        break  # Alpha Cutoff
                    
                if beta <= alpha:
                    break
            return min_eval, None
    
    def get_best_move(self, game_state, player_num):
        """ Main entry point"""
        self.start_time = time.time()
        self.nodes_searched = 0
        
        # Get current directions from the trail history
        def infer_direction(trail):
            if len(trail) < 2:
                return 'RIGHT'  # Default
            prev = trail[-2]
            curr = trail[-1]
            dx = (curr[0] - prev[0]) % 20  # Handle torus wrap
            dy = (curr[1] - prev[1]) % 18
            
            # Normalize for torus (e.g., 19 -> -1)
            if dx > 10:
                dx = dx - 20
            if dy > 9:
                dy = dy - 18
            
            if dx == 1:
                return 'RIGHT'
            elif dx == -1:
                return 'LEFT'
            elif dy == 1:
                return 'DOWN'
            elif dy == -1:
                return 'UP'
            return 'RIGHT'

        # Create simulator state
        sim_state = GameStateSimulator(
            game_state.get('board'),
            game_state.get('agent1_trail'),
            game_state.get('agent2_trail'),
            infer_direction(game_state.get('agent1_trail', [])),
            infer_direction(game_state.get('agent2_trail', [])),
            game_state.get('agent1_boosts', 3),
            game_state.get('agent2_boosts', 3),
            game_state.get('agent1_alive', True),
            game_state.get('agent2_alive', True)
        )
        
        # Run minimax
        score, move = self.minimax(sim_state, self.max_depth, 
                                   float('-inf'), float('inf'), True, player_num)
        
        elapsed = time.time() - self.start_time
        print(f"[Player {player_num}] Minimax: {self.nodes_searched} nodes, {elapsed:.2f}s, move={move}, eval={score}")
        
        return move if move else 'RIGHT'  # Fallback to RIGHT if no move found






@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity.

    Returns participant and agent_name (so Judge.check_latency can create Agent objects).
    """
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


def _update_local_game_from_post(data: dict):
    """Update the local GLOBAL_GAME using the JSON posted by the judge.

    The judge posts a dictionary with keys matching the Judge.send_state payload
    (board, agent1_trail, agent2_trail, agent1_length, agent2_length, agent1_alive,
    agent2_alive, agent1_boosts, agent2_boosts, turn_count).
    """
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)

        if "board" in data:
            try:
                GLOBAL_GAME.board.grid = data["board"]
            except Exception:
                pass

        if "agent1_trail" in data:
            GLOBAL_GAME.agent1.trail = deque(tuple(p) for p in data["agent1_trail"]) 
        if "agent2_trail" in data:
            GLOBAL_GAME.agent2.trail = deque(tuple(p) for p in data["agent2_trail"]) 
        if "agent1_length" in data:
            GLOBAL_GAME.agent1.length = int(data["agent1_length"])
        if "agent2_length" in data:
            GLOBAL_GAME.agent2.length = int(data["agent2_length"])
        if "agent1_alive" in data:
            GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
        if "agent2_alive" in data:
            GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])
        if "agent1_boosts" in data:
            GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
        if "agent2_boosts" in data:
            GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])
        if "turn_count" in data:
            GLOBAL_GAME.turns = int(data["turn_count"])


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge calls this to push the current game state to the agent server.

    The agent should update its local representation and return 200.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge calls this (GET) to request the agent's move for the current tick.

    Query params the judge sends (optional): player_number, attempt_number,
    random_moves_left, turn_count. Agents can use this to decide.
    
    Return format: {"move": "DIRECTION"} or {"move": "DIRECTION:BOOST"}
    where DIRECTION is UP, DOWN, LEFT, or RIGHT
    and :BOOST is optional to use a speed boost (move twice)
    """
    player_number = request.args.get("player_number", default=1, type=int)

    with game_lock:
        state = dict(LAST_POSTED_STATE)   
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        boosts_remaining = my_agent.boosts_remaining
   
    # -----------------your code here-------------------
    # Initialize minimax agent if not already created
    global minimax_agent
    if minimax_agent is None:
        minimax_agent = MinimaxAgent(max_depth=4, time_limit=3.5)
    
    # Use minimax to get the best move
    try:
        print(f"\n=== TURN {state.get('turn_count', 0)} - Player {player_number} ===")
        print(f"State received: agent1_alive={state.get('agent1_alive')}, agent2_alive={state.get('agent2_alive')}")
        print(f"Trail lengths: P1={len(state.get('agent1_trail', []))}, P2={len(state.get('agent2_trail', []))}")
        move = minimax_agent.get_best_move(state, player_number)
        print(f"Returning move: {move}")
    except Exception as e:
        print(f"Error in minimax: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to a safe move
        move = "RIGHT"
    # -----------------end code here--------------------

    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state.

    We update local state for record-keeping and return OK.
    """
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    app.run(host="0.0.0.0", port=port, debug=True)
