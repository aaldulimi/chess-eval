
# Chess-eval
Chess-eval is a library designed to evaluate the capabilities of Large Language Models (LLMs) by engaging them in chess games. It supports matchups between different LLMs, integration with the Stockfish engine, and the organization of round-robin tournaments.

## Usage

### Installation
```
git clone https://github.com/aaldulimi/chess-eval.git
cd chess-eval
poetry install or pip install -r requirements.txt
```
Read the docs below or take a look at the `example.py` to get started right away.

### Single Game
To play a single game between two models:

```python
import chesseval
import os

# Initialize players
gpt4turbo = chesseval.OpenAI(model_name="gpt-4-turbo", api_key=os.getenv("OPENAI_API_KEY"))
haiku = chesseval.Anthropic(model_name="claude-3-haiku-20240307", api_key=os.getenv('ANTHROPIC_API_KEY'))

# Start a new game
result = chesseval.new_game(
    player_one=gpt4turbo,
    player_two=haiku,
    max_turns=100 # Optional
)

# Display the result
print(result)
```
### Round-robin Tournament
To organize a round-robin tournament among multiple players:

```python
import chesseval
import os

# Initialize players
gpt4turbo = chesseval.OpenAI(model_name="gpt-4-turbo", api_key=os.getenv("OPENAI_API_KEY"))
haiku = chesseval.Anthropic(model_name="claude-3-haiku-20240307", api_key=os.getenv('ANTHROPIC_API_KEY'))
stockfish = chesseval.StockfishPlayer(
    player_name="stockfish-lvl-10",
    engine_path="stockfish/stockfish-macos-x86-64-avx2",
    skill_level=10,
    eval_time=50,
)
llama_v3_70b_instruct = chesseval.FireworksAI(
    model_name="accounts/fireworks/models/llama-v3-70b-instruct",
    price_per_million_tokens=0.9,
    api_key=os.getenv('FIREWORKS_API_KEY')
)

# Conduct the tournament
result = chesseval.round_robin_tournament(
    players=[gpt4turbo, haiku, stockfish, llama_v3_70b_instruct],
    games_per_match=2,
)

# Display the results
print(result)
```

### Cost Calculation
To calculate the total cost of using each AI player:

```python
print(llama_v3_70b_instruct.total_cost)
print(gpt4turbo.total_cost)
```

### Additional Configuration
You can configure additional parameters for specific AI models:

```python
llama_v3_70b_instruct = chesseval.FireworksAI(
    model_name="accounts/fireworks/models/llama-v3-70b-instruct",
    price_per_million_tokens=0.9,
    api_key=os.getenv('FIREWORKS_API_KEY'),
    temperature=0,
    max_retries_per_move=5,  # Maximum retries for illegal moves
    top_k=40,
    top_p=1,
    _system_prompt_file_path="prompts/no_example.txt",  # Avoids returning example moves
    player_name='llama-3-70b',
    use_system_prompt=True  # Use if supported by the model
)
```

### Custom Agents/Providers
You can also create custom agents via a subclass or a decorator.

#### Subclass
```python
class RandomAgent(chesseval.Player):
    def move(self, board: chess.Board) -> str:
        legal_moves = list(board.legal_moves)
        chosen_move = random.choice(legal_moves)
        return chosen_move.uci()


random_agent = random_agent()

result = chesseval.new_game(
    player_one=gpt4turbo,
    player_two=random_agent,
)
``` 

#### Decorator
```python 
@chesseval.player(player_name="random")
def random_agent(board: chess.Board) -> str:
    legal_moves = list(board.legal_moves)
    chosen_move = random.choice(legal_moves)
    return chosen_move.uci()

random_agent = random_agent()

result = chesseval.new_game(
    player_one=gpt4turbo,
    player_two=random_agent,
)
```