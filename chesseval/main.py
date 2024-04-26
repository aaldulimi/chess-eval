import chess
from itertools import zip_longest
import re
import typing
import openai
import stockfish
import dataclasses
import abc
import google.generativeai
import os
import fireworks.client
import random
import anthropic
import logging
import sys
import functools

logger = logging.getLogger("chesseval")
stdout_handler = logging.StreamHandler(stream=sys.stdout)
stdout_handler.setFormatter(
    logging.Formatter(
        "%(name)s: %(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | %(process)d >>> %(message)s"
    )
)
logger.addHandler(stdout_handler)


def log_move_call(func):
    @functools.wraps(func)
    def wrapper(self, board, *args, **kwargs):
        logger.debug(f"{self.__class__.__name__} is making a move")
        result = func(self, board, *args, **kwargs)
        logger.debug(f"{self.__class__.__name__} made the move: {result}")
        return result

    return wrapper


class Player(abc.ABC):
    def __init__(self):
        self.total_cost = 0
        self.player_name = "Default"
        logger.debug(f"Player initialized with total_cost set to {self.total_cost}")

    @abc.abstractmethod
    @log_move_call
    def move(self, board: chess.Board) -> str:
        pass

    def _format_board_history(self, board: chess.Board) -> str:
        logger.debug("Formatting board history")
        black_moves = []
        white_moves = []
        for i, move in enumerate(board.move_stack):
            if i % 2 == 0:
                white_moves.append(board.uci(move))
            else:
                black_moves.append(board.uci(move))

        paired_moves = list(zip_longest(white_moves, black_moves, fillvalue=""))
        formatted_moves = "\n".join(
            "{:<15}{}".format(white, black) for white, black in paired_moves
        )
        full_output = "white".ljust(15) + "black\n" + formatted_moves
        logger.debug(f"Formatted board history: {full_output}")

        return full_output

    def _list_legal_moves(
        self, board: chess.Board, uci_format: bool = True
    ) -> list[str]:
        logger.debug("Listing legal moves")
        moves = [
            board.uci(move) if uci_format else board.san(move)
            for move in board.legal_moves
        ]

        logger.debug(f"Legal moves: {moves}")
        return moves

    def _load_system_prompt(self, file_path: str) -> str:
        logger.debug(f"Loading system prompt from {file_path}")
        with open(file_path, "r", encoding="utf-8") as file:
            prompt = file.read()

        logger.debug(f"Loaded system prompt: {prompt}")
        return prompt


def player(player_name: str = None):
    def decorator(func):
        class PlayerAdapter(Player):
            def __init__(self):
                super().__init__()
                self.player_name = player_name or "Default"

            def move(self, board):
                return func(board)

        return PlayerAdapter

    return decorator


class StockfishPlayer(Player):
    def __init__(
        self,
        engine_path: str,
        player_name: str | None = None,
        skill_level: int = 1,
        depth_level: int = 1,
        eval_time: typing.Optional[int] = 50,
    ):
        super().__init__()
        self.player_name = player_name or "Stockfish"
        self.engine_path = engine_path
        self._stockfish = stockfish.Stockfish(path=self.engine_path)

        self.skill_level = skill_level
        self.depth_level = depth_level
        self.eval_time = eval_time
        self._stockfish.set_skill_level(self.skill_level)
        self._stockfish.set_depth(self.depth_level)
        logger.debug(
            f"Initialized StockfishPlayer with engine_path={engine_path}, skill_level={skill_level}, depth_level={depth_level}, eval_time={eval_time}"
        )

    def move(self, board: chess.Board) -> str:
        logger.debug("Stockfish computing best move")

        self._stockfish.set_fen_position(board.fen())
        best_move = self._stockfish.get_best_move_time(self.eval_time)
        logger.debug(f"Best move determined by Stockfish: {best_move}")

        return best_move


class OpenAI(Player):
    _VALID_MODEL_NAMES = {
        "gpt-4-turbo": {"prompt_token_price": 0.00001, "response_token_price": 0.00003},
        "gpt-4-turbo-2024-04-09": {
            "prompt_token_price": 0.00001,
            "response_token_price": 0.00003,
        },
        "gpt-4-0125-preview": {
            "prompt_token_price": 0.00001,
            "response_token_price": 0.00003,
        },
        "gpt-4-1106-preview	": {
            "prompt_token_price": 0.00001,
            "response_token_price": 0.00003,
        },
        "gpt-3.5-turbo-0125": {
            "prompt_token_price": 5e-7,
            "response_token_price": 0.0000015,
        },
        "gpt-3.5-turbo-instruct": {
            "prompt_token_price": 0.0000015,
            "response_token_price": 0.000002,
        },
    }

    def __init__(
        self,
        api_key: typing.Optional[str] = None,
        model_name: str = "gpt-4-turbo",
        temperature: float = 0,
        max_retries_per_move: int = 5,
        player_name: str | None = None,
        _system_prompt_file_path: str = "prompts/no_example.txt",
    ):
        super().__init__()
        if model_name not in self._VALID_MODEL_NAMES.keys():
            raise ValueError(
                f"Invalid model name. Valid options are: {', '.join(self._VALID_MODEL_NAMES.keys())}"
            )

        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "An API key must be provided either directly or through the OPENAI_API_KEY environment variable."
            )

        openai.api_key = self._api_key
        self._client = openai.OpenAI()

        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = self._load_system_prompt(_system_prompt_file_path)
        self.max_retries_per_move = max_retries_per_move
        self.player_name = player_name or model_name

        self.prompt_token_price = self._VALID_MODEL_NAMES[self.model_name][
            "prompt_token_price"
        ]
        self.response_token_price = self._VALID_MODEL_NAMES[self.model_name][
            "response_token_price"
        ]

        logger.debug(
            f"Initialized OpenAI player with model_name={model_name}, temperature={temperature}"
        )

    def move(self, board: chess.Board) -> str:
        legal_moves = self._list_legal_moves(board)

        for attempt in range(self.max_retries_per_move):
            prompt = self._create_prompt(board)
            response = self._call_model(prompt)
            self._update_usage_details(response.usage)

            try:
                uci_code = self._extract_move(response.choices[0].message.content)
                if uci_code in legal_moves:
                    return uci_code

            except ValueError:
                if attempt >= self.max_retries_per_move - 1:
                    break

        return random.choice(legal_moves)

    def _create_prompt(self, board: chess.Board) -> str:
        color = "White" if board.turn else "Black"
        legal_moves = ", ".join(self._list_legal_moves(board))
        return (
            f"Board state:\n{board}\n"
            f"Game history (UCI):\n{self._format_board_history(board)}\n"
            f"Legal moves you must choose from: {legal_moves}\n"
            f"You are {color}. It's your turn to make a move. "
            f"Choose the best possible move to increase your chance of winning this game."
        )

    def _call_model(self, prompt: str) -> openai.types.Completion:
        return self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )

    def _update_usage_details(self, usage: openai.types.CompletionUsage) -> None:
        self.total_cost += usage.prompt_tokens * self.prompt_token_price
        self.total_cost += usage.completion_tokens * self.response_token_price

    def _extract_move(self, response: str) -> str:
        match = re.search(r"Action: (\w+)\[(\w*)\]", response)
        if match:
            _ = match.group(1)
            action_input = match.group(2) if match.group(2) else None
            return action_input
        else:
            raise ValueError(
                "The response from the model could not be parsed into an action."
            )


class Anthropic(Player):
    _VALID_MODEL_NAMES = {
        "claude-3-opus-20240229": {
            "prompt_token_price": 0.000015,
            "response_token_price": 0.000075,
        },
        "claude-3-sonnet-20240229": {
            "prompt_token_price": 0.000003,
            "response_token_price": 0.000015,
        },
        "claude-3-haiku-20240307": {
            "prompt_token_price": 2.5e-7,
            "response_token_price": 0.00000125,
        },
    }

    def __init__(
        self,
        api_key: typing.Optional[str] = None,
        model_name: str = "claude-3-haiku-20240307",
        temperature: float = 0,
        max_retries_per_move: int = 5,
        player_name: str | None = None,
        _system_prompt_file_path: str = "prompts/no_example.txt",
    ):
        super().__init__()
        if model_name not in self._VALID_MODEL_NAMES.keys():
            raise ValueError(
                f"Invalid model name. Valid options are: {', '.join(self._VALID_MODEL_NAMES.keys())}"
            )

        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError(
                "An API key must be provided either directly or through the ANTHROPIC_API_KEY environment variable."
            )

        self._client = anthropic.Anthropic(api_key=self._api_key)

        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = self._load_system_prompt(_system_prompt_file_path)
        self.max_retries_per_move = max_retries_per_move
        self.player_name = player_name or model_name
        self._max_output_tokens = 1024

        self.prompt_token_price = self._VALID_MODEL_NAMES[self.model_name][
            "prompt_token_price"
        ]
        self.response_token_price = self._VALID_MODEL_NAMES[self.model_name][
            "response_token_price"
        ]

        logger.debug(
            f"Initialized Anthropic player with model_name={model_name}, temperature={temperature}"
        )

    def move(self, board: chess.Board) -> str:
        legal_moves = self._list_legal_moves(board)

        for attempt in range(self.max_retries_per_move):
            prompt = self._create_prompt(board)
            response = self._call_model(prompt)
            self._update_usage_details(response.usage)

            try:
                uci_code = self._extract_move(response.content[0].text)
                if uci_code in legal_moves:
                    return uci_code

            except ValueError:
                if attempt >= self.max_retries_per_move - 1:
                    break

        return random.choice(legal_moves)

    def _create_prompt(self, board: chess.Board) -> str:
        color = "White" if board.turn else "Black"
        legal_moves = ", ".join(self._list_legal_moves(board))
        return (
            f"Board state:\n{board}\n"
            f"Game history (UCI):\n{self._format_board_history(board)}\n"
            f"Legal moves you must choose from: {legal_moves}\n"
            f"You are {color}. It's your turn to make a move. "
            f"Choose the best possible move to increase your chance of winning this game."
        )

    def _call_model(self, prompt: str) -> anthropic.types.Message:
        return self._client.messages.create(
            model=self.model_name,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self._max_output_tokens,
        )

    def _update_usage_details(self, usage) -> None:
        self.total_cost += usage.input_tokens * self.prompt_token_price
        self.total_cost += usage.output_tokens * self.response_token_price

    def _extract_move(self, response: str) -> str:
        match = re.search(r"Action: (\w+)\[(\w*)\]", response)
        if match:
            _ = match.group(1)
            action_input = match.group(2) if match.group(2) else None
            return action_input
        else:
            raise ValueError(
                "The response from the model could not be parsed into an action."
            )


class GoogleGenAI(Player):
    _VALID_MODEL_NAMES = [
        "gemini-pro",
        "gemini-1.5-pro-latest",
        "gemini-1.0-pro-latest",
        "gemini-1.0-pro-001",
        "gemini-1.0-pro",
    ]

    _SAFETY_SETTINGS = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    def __init__(
        self,
        api_key: typing.Optional[str] = None,
        model_name: str = "gemini-pro",
        temperature: float = 0,
        max_retries_per_move: int = 5,
        top_k: int = 1,
        top_p: int = 1,
        player_name: str | None = None,
        _system_prompt_file_path: str = "prompts/with_example.txt",
    ):
        super().__init__()
        if model_name not in self._VALID_MODEL_NAMES:
            raise ValueError(
                f"Invalid model name. Valid options are: {', '.join(self._VALID_MODEL_NAMES)}"
            )

        self._api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "An API key must be provided either directly or through the GOOGLE_API_KEY environment variable."
            )

        google.generativeai.configure(api_key=self._api_key)

        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = self._load_system_prompt(_system_prompt_file_path)
        self.player_name = player_name or f"{self.model_name}"

        self.max_retries_per_move = max_retries_per_move

        self.top_k = top_k
        self.top_p = top_p
        self._max_output_tokens = 2048

        self._client = google.generativeai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": self.temperature,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "max_output_tokens": self._max_output_tokens,
            },
            safety_settings=self._SAFETY_SETTINGS,
        )

        if "gemini-1.5" in self.model_name:
            self._client._system_instruction = self.system_prompt

        logger.debug(
            f"Initialized GoogleGenAI player with model_name={model_name}, temperature={temperature}"
        )

    def move(self, board: chess.Board) -> str:
        legal_moves = self._list_legal_moves(board)

        for attempt in range(self.max_retries_per_move):
            prompt = self._create_prompt(board)
            response = self._call_model(prompt)

            try:
                uci_code = self._extract_move(response.text)
                if uci_code in legal_moves:
                    return uci_code

            except ValueError:
                if attempt >= self.max_retries_per_move - 1:
                    break

        return random.choice(legal_moves)

    def _create_prompt(self, board: chess.Board) -> str:
        color = "White" if board.turn else "Black"
        legal_moves = ", ".join(self._list_legal_moves(board))
        if "gemini-1.5" in self.model_name:
            return (
                f"Board state:\n{board}\n"
                f"Game history (UCI):\n{self._format_board_history(board)}\n"
                f"Legal moves you must choose from: {legal_moves}\n"
                f"You are {color}. It's your turn to make a move. "
                f"Choose the best possible move to increase your chance of winning this game."
            )

        return (
            self.system_prompt,
            f"Board state:\n{board}\n"
            f"Game history (UCI):\n{self._format_board_history(board)}\n"
            f"Legal moves you must choose from: {legal_moves}\n"
            f"You are {color}. It's your turn to make a move. "
            f"Choose the best possible move to increase your chance of winning this game.",
        )

    def _call_model(self, prompt: str):
        return self._client.generate_content(prompt)

    def _extract_move(self, response: str) -> str:
        match = re.search(r"Action: (\w+)\[(\w*)\]", response)
        if match:
            _ = match.group(1)
            action_input = match.group(2) if match.group(2) else None
            return action_input
        else:
            raise ValueError(
                "The response from the model could not be parsed into an action."
            )


class FireworksAI(Player):
    def __init__(
        self,
        model_name: str,
        api_key: typing.Optional[str] = None,
        price_per_million_tokens: float = 0,
        temperature: float = 0,
        max_retries_per_move: int = 5,
        top_k: int = 40,
        top_p: int = 1,
        _system_prompt_file_path: str = "prompts/no_example.txt",
        player_name: str | None = None,
        use_system_prompt: bool = True,
    ):
        super().__init__()

        self._api_key = api_key or os.getenv("FIREWORKS_API_KEY")
        if not self._api_key:
            raise ValueError(
                "An API key must be provided either directly or through the FIREWORKS_API_KEY environment variable."
            )

        self._client = fireworks.client.Fireworks(api_key=self._api_key)

        self.use_system_prompt = use_system_prompt
        self.system_prompt = self._load_system_prompt(_system_prompt_file_path)

        self.model_name = model_name
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self._max_output_tokens = 1024

        self.player_name = player_name or f"FireworksAI-{self.model_name}"

        self.price_per_million_tokens = price_per_million_tokens
        self.token_price = self.price_per_million_tokens / 1000000

        self.max_retries_per_move = max_retries_per_move

        logger.debug(
            f"Initialized FireworksAI player with model_name={model_name}, temperature={temperature}"
        )

    def move(self, board: chess.Board) -> str:
        legal_moves = self._list_legal_moves(board)

        for attempt in range(self.max_retries_per_move):
            prompt = self._create_prompt(board)
            response = self._call_model(prompt)
            self._update_usage_details(response.usage)

            try:
                uci_code = self._extract_move(response.choices[0].message.content)
                if uci_code in legal_moves:
                    return uci_code

            except ValueError:
                if attempt >= self.max_retries_per_move - 1:
                    break

        return random.choice(legal_moves)

    def _create_prompt(self, board: chess.Board) -> str:
        color = "White" if board.turn else "Black"
        legal_moves = ", ".join(self._list_legal_moves(board))
        return (
            f"Board state:\n{board}\n"
            f"Game history (UCI):\n{self._format_board_history(board)}\n"
            f"Legal moves you must choose from: {legal_moves}\n"
            f"You are {color}. It's your turn to make a move. "
            f"Choose the best possible move to increase your chance of winning this game."
        )

    def _call_model(self, prompt: str) -> fireworks.client.ChatCompletion:
        if self.use_system_prompt:
            return self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )

        return self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": self.system_prompt + "\n" + prompt}],
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )

    def _update_usage_details(self, usage) -> None:
        self.total_cost += usage.prompt_tokens * self.token_price
        self.total_cost += usage.completion_tokens * self.token_price

    def _extract_move(self, response: str) -> str:
        match = re.search(r"Action: (\w+)\[(\w*)\]", response)
        if match:
            _ = match.group(1)
            action_input = match.group(2) if match.group(2) else None
            return action_input
        else:
            raise ValueError(
                "The response from the model could not be parsed into an action."
            )


@dataclasses.dataclass
class MatchResult:
    player_one: str
    player_two: str
    score: typing.Tuple[float, float]
    reason: str

    def __str__(self) -> str:
        max_name_length = max(
            len(self.player_one), len(self.player_two), len("Agent 1"), len("Agent 2")
        )
        max_score_length = max(len(f"{self.score[0]}-{self.score[1]}"), len("Result"))

        header = f"+{'-' * (max_name_length + 4)}+{'-' * (max_name_length + 4)}+{'-' * (max_score_length + 4)}+"
        result_str = f"{header}\n"
        result_str += f"| {'Agent 1'.center(max_name_length + 3)}| {'Agent 2'.center(max_name_length + 3)}| {'Result'.center(max_score_length + 3)}|\n"
        result_str += f"{header}\n"
        score = f"{self.score[0]}-{self.score[1]}"
        result_str += f"| {self.player_one.center(max_name_length + 3)}| {self.player_two.center(max_name_length + 3)}| {score.center(max_score_length + 3)}|\n"
        result_str += f"{header}"
        return result_str


def new_game(
    player_one: Player,
    player_two: Player,
    max_turns: int | None = None,
    max_illegal_moves: int | None = None,
) -> MatchResult:
    board = chess.Board()
    illegal_moves_count = {player_one: 0, player_two: 0}
    turn_count = 0

    while not board.is_game_over(claim_draw=True) and (
        max_turns is None or turn_count < max_turns
    ):
        current_player = player_one if turn_count % 2 == 0 else player_two
        move = current_player.move(board)

        if move not in [m.uci() for m in board.legal_moves]:
            illegal_moves_count[current_player] += 1
            if (
                max_illegal_moves is not None
                and illegal_moves_count[current_player] >= max_illegal_moves
            ):
                winner = player_two if current_player is player_one else player_one
                return MatchResult(
                    player_one.player_name,
                    player_two.player_name,
                    (1.0, 0.0) if winner is player_one else (0.0, 1.0),
                    "Disqualification due to illegal moves",
                )
        else:
            board.push(chess.Move.from_uci(move))
            illegal_moves_count[current_player] = 0
            turn_count += 1

    if max_turns is not None and turn_count >= max_turns:
        return MatchResult(
            player_one.player_name,
            player_two.player_name,
            (0.5, 0.5),
            "Game drawn by turn limit",
        )

    result = board.result()
    if result == "1-0":
        return MatchResult(
            player_one.player_name,
            player_two.player_name,
            (1.0, 0.0),
            f"Checkmate by {player_one.player_name}",
        )
    elif result == "0-1":
        return MatchResult(
            player_one.player_name,
            player_two.player_name,
            (0.0, 1.0),
            f"Checkmate by {player_one.player_name}",
        )
    else:
        reason = "Draw"
        if board.is_stalemate():
            reason = "Stalemate"
        elif board.is_insufficient_material():
            reason = "Insufficient material"
        elif board.can_claim_threefold_repetition():
            reason = "Threefold repetition"
        elif board.can_claim_fifty_moves():
            reason = "Fifty-move rule"
        return MatchResult(
            player_one.player_name, player_two.player_name, (0.5, 0.5), reason
        )


class TournamentResult:
    def __init__(self):
        self.results = {}

    def add_match(
        self, player_one_name: str, player_two_name: str, score: tuple[int, int]
    ):
        if player_one_name not in self.results:
            self.results[player_one_name] = {
                "Matches Played": 0,
                "Wins": 0,
                "Losses": 0,
            }
        if player_two_name not in self.results:
            self.results[player_two_name] = {
                "Matches Played": 0,
                "Wins": 0,
                "Losses": 0,
            }

        self.results[player_one_name]["Matches Played"] += 1
        self.results[player_two_name]["Matches Played"] += 1

        if score == (1, 0):
            self.results[player_one_name]["Wins"] += 1
            self.results[player_two_name]["Losses"] += 1
        elif score == (0, 1):
            self.results[player_one_name]["Losses"] += 1
            self.results[player_two_name]["Wins"] += 1

    def __str__(self) -> str:
        headers = ["Player", "Matches", "Score"]
        header_lengths = [
            max(len(player) for player in self.results.keys()) + 2,
            10,
            10,
        ]
        header_row = "|".join(
            header.center(length + 1) for header, length in zip(headers, header_lengths)
        )
        result_str = f"+{'-' * (sum(header_lengths) + 3 * len(header_lengths) - 1)}+\n"
        result_str += f"| {header_row} |\n"
        result_str += f"+{'-' * (sum(header_lengths) + 3 * len(header_lengths) - 1)}+\n"
        for player, stats in self.results.items():
            score = f"{stats['Wins']} - {stats['Losses']}"
            row = f"| {player.center(header_lengths[0])} | {str(stats['Matches Played']).center(header_lengths[1])} | {score.center(header_lengths[2])} |"
            result_str += f"{row}\n"
            result_str += (
                f"+{'-' * (sum(header_lengths) + 3 * len(header_lengths) - 1)}+\n"
            )
        return result_str


def round_robin_tournament(
    players: list[Player],
    games_per_match: int,
    max_turns: int | None = None,
    max_illegal_moves: int | None = None,
) -> TournamentResult:
    tournament_result = TournamentResult()

    for i, player_one in enumerate(players):
        for player_two in players[i + 1 :]:
            for _ in range(games_per_match):
                match_result = new_game(
                    player_one, player_two, max_turns, max_illegal_moves
                )
                tournament_result.add_match(
                    player_one.player_name, player_two.player_name, match_result.score
                )

    return tournament_result
