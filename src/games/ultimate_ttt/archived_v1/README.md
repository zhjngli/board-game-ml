# Archived UTTT AlphaZero Training Artifacts (v1)

These artifacts were produced by the original UTTT AlphaZero training configuration, which had several issues that prevented effective learning. They are kept here for reference before deletion.

## Original Architecture

- Input: (3,3,3,3) reshaped to (9,9,1) — spatial layout was incorrect (row-major reshape scrambled board topology)
- 6 conv layers, 3 filters each, 3x3 kernel
- 3 dense layers: 2048, 1024, 512
- Dropout: 0.3, Learning rate: 0.01, Batch size: 64, Epochs: 10
- No active_nonant encoding — NN could not distinguish forced vs free move constraints
- No Dirichlet noise in MCTS
- MCTS searches: 1000 (both training and evaluation)

## Original Training Parameters

- training_episodes: 200 (only reached episode 12)
- training_games_per_episode: 50
- training_queue_length: 50000
- training_hist_max_len: 50
- thread_max_workers: 8
- temp_threshold: 11
- pit_games: 20, pit_threshold: 0.55

## Results

- 13 episodes completed over ~2 months (June-August 2024)
- Only 4 model improvements saved (episodes 6, 7, 8, 9)
- Agent barely better than random play
