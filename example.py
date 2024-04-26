from dotenv import load_dotenv
import chesseval
import chess
import random

load_dotenv()

stockfish_player_1 = chesseval.StockfishPlayer(
    player_name="stockfish",
    engine_path="stockfish/stockfish-macos-x86-64-avx2",
    skill_level=10,
    eval_time=1,
)

gpt4turbo = chesseval.OpenAI(model_name="gpt-4-turbo", temperature=0)
haiku = chesseval.Anthropic(model_name="claude-3-haiku-20240307")
gemini_1_pro = chesseval.GoogleGenAI(model_name="gemini-1.0-pro-latest")
llama_v3_70b_instruct = chesseval.FireworksAI(
    model_name="accounts/fireworks/models/llama-v3-70b-instruct",
    price_per_million_tokens=0.9,
)

# single game
result = chesseval.new_game(
    player_one=stockfish_player_1, player_two=gpt4turbo, max_turns=100
)

print(result)

# round-robin 
result = chesseval.round_robin_tournament(
    players=[stockfish_player_1, gpt4turbo, llama_v3_70b_instruct, gemini_1_pro, haiku],
    games_per_match=1,
)

print(result)

# total cost for each agent 
print(gpt4turbo.total_cost)

# custom agent via subclass
class RandomAgent(chesseval.Player):
    def move(self, board: chess.Board) -> str:
        legal_moves = list(board.legal_moves)
        chosen_move = random.choice(legal_moves)
        return chosen_move.uci()

random_agent = RandomAgent()

# custom agent via decorator 
@chesseval.player()
def random_move(board: chess.Board) -> str:
    legal_moves = list(board.legal_moves)
    chosen_move = random.choice(legal_moves)
    return chosen_move.uci()

random_agent = random_move()