# from games.digit_party.digit_party import human_game as digit_party_human_game
# from games.digit_party.digit_party import (
#     many_trained_games as digit_party_many_trained_games,
# )
# from games.digit_party.digit_party import trained_game as digit_party_trained_game

# digit_party_human_game()
# digit_party_trained_game(game_size=3)
# digit_party_many_trained_games(game_size=3)

# from games.random_walk.random_walk import q_trained_game as random_walk_trained_game

# random_walk_trained_game()

# from games.tictactoe.tictactoe import monte_carlo_many_games as ttt_mc_many_games
# from games.tictactoe.tictactoe import monte_carlo_trained_game as ttt_mc_trained_game

# ttt_mc_trained_game(training_episodes=1000000)
# ttt_mc_many_games()

from games.ultimate_ttt.run import alpha_zero_trained_game

alpha_zero_trained_game()
