"""
Sample agent for Case Closed Challenge - Works with Judge Protocol
This agent runs as a Flask server and responds to judge requests.
"""

import os
from flask import Flask, request, jsonify
from collections import deque

app = Flask(__name__)

# Basic identity
PARTICIPANT = os.getenv("PARTICIPANT", "SampleParticipant")
AGENT_NAME = os.getenv("AGENT_NAME", "SampleAgent")

# Track game state
game_state = {
    "board": None,
    "agent1_trail": [],
    "agent2_trail": [],
    "agent1_length": 0,
    "agent2_length": 0,
    "agent1_alive": True,
    "agent2_alive": True,
    "agent1_boosts": 3,
    "agent2_boosts": 3,
    "turn_count": 0,
    "player_number": 1,
}


@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity."""
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge calls this to push the current game state to the agent server."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    
    # Update our local game state
    game_state.update(data)
    
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge calls this (GET) to request the agent's move for the current tick.
    
    Return format: {"move": "DIRECTION"} or {"move": "DIRECTION:BOOST"}
    """
    player_number = request.args.get("player_number", default=1, type=int)
    turn_count = game_state.get("turn_count", 0)
    
    # Get our current state
    if player_number == 1:
        my_trail = game_state.get("agent1_trail", [])
        my_boosts = game_state.get("agent1_boosts", 3)
        other_trail = game_state.get("agent2_trail", [])
    else:
        my_trail = game_state.get("agent2_trail", [])
        my_boosts = game_state.get("agent2_boosts", 3)
        other_trail = game_state.get("agent1_trail", [])
    
    # Simple decision logic
    move = decide_move(my_trail, other_trail, turn_count, my_boosts)
    
    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state."""
    data = request.get_json()
    if data:
        result = data.get("result", "UNKNOWN")
        print(f"\nGame Over! Result: {result}")
    return jsonify({"status": "acknowledged"}), 200


def decide_move(my_trail, other_trail, turn_count, my_boosts):
    """Simple decision logic for the agent.
    
    Strategy:
    - Move LEFT 10 times
    - Move DOWN 1 time
    - Move RIGHT 10 times
    - Repeat
    """
    # Pattern repeats every 21 turns
    pattern_position = turn_count % 21
    
    if pattern_position < 10:
        # First 10 moves: LEFT
        chosen_dir = "LEFT"
    elif pattern_position == 10:
        # 11th move: DOWN
        chosen_dir = "DOWN"
    else:
        # Last 10 moves: RIGHT
        chosen_dir = "RIGHT"
    
    return chosen_dir


if __name__ == "__main__":
    # For development only. Port can be overridden with the PORT env var.
    port = int(os.environ.get("PORT", "5009"))
    print(f"Starting {AGENT_NAME} ({PARTICIPANT}) on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
