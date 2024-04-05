# from games.digit_party.run import human_game as digit_party_human_game

# digit_party_human_game()

# from games.digit_party.run import q_trained_game as digit_party_trained_game

# digit_party_trained_game(game_size=3, num_games=100_000)

# from games.random_walk.random_walk import q_trained_game as random_walk_trained_game

# random_walk_trained_game()

# from games.tictactoe.run import monte_carlo_many_games as ttt_mc_many_games
# from games.tictactoe.run import monte_carlo_trained_game as ttt_mc_trained_game

# ttt_mc_trained_game(training_episodes=1)
# ttt_mc_many_games()

from games.tictactoe.run import (
    a0_vs_mc_games,
    alpha_zero_many_games,
    alpha_zero_trained_game,
)

alpha_zero_trained_game()
alpha_zero_many_games()
a0_vs_mc_games()

# from games.ultimate_ttt.run import alpha_zero_trained_game

# alpha_zero_trained_game()

# from games.digit_party.run import deep_q_3x3_trained_game

# deep_q_3x3_trained_game()
