# Digit Party 5x5 Deep Q Plan

This is the first-pass plan for training a `5x5` Digit Party deep q model.

The point is not to invent a huge new architecture. The point is to take what finally worked on `3x3`, scale it up carefully, and give it enough training to matter.

## Baseline To Copy From

The `3x3` deep q setup that actually worked uses:

- board input plus current digit plus next digit
- a simple conv stack
- a single linear q-value head
- `epochs=1` per replay update
- masked legal actions
- small, frequent replay updates
- a target network
- enough evaluation and checkpointing to trust the run

That is still the right basic shape for `5x5`.

## Final Recommendation

### Network Architecture

Use:

- board input: `5x5`
- current digit input: scalar
- next digit input: scalar
- normalize board and visible digits by the max digit for the board size
- `8` conv layers
- `24` conv filters
- `3x3` kernels
- `1` dense layer
- `512` dense units
- `dropout_rate=0.0`
- `epochs=1`
- `batch_size=128`
- `learning_rate=0.00075`
- linear q-value output with `25` actions

Keep the current digit and next digit concatenated after flattening for the first pass. That keeps the implementation close to the `3x3` path.

There is still a good follow-up idea to try later:

- broadcast current digit and next digit as extra `5x5` planes before the conv stack

That would let the spatial filters condition on digit context earlier. It is a good next architectural experiment, but not the first one.

### Training Schedule

Use:

- `alpha=0.1`
- `gamma=0.95`
- `min_epsilon=0.05`
- `max_epsilon=1.0`
- `epsilon_decay=0.00002`
- `valid_action_reward=0.01`
- `memory_size=500_000`
- `min_replay_size=25_000`
- `minibatch_size=128`
- `steps_to_train_longterm=4`
- `steps_to_train_shortterm=0`
- `steps_per_target_update=500`
- `training_episodes=500_000`
- `episodes_per_model_save=10_000`
- `episodes_per_memory_save=10_000`
- `episodes_per_stats_print=1_000`
- `episodes_per_evaluation=5_000`
- training evaluation over `250` games
- final evaluation over `1000` games
- `state_tracker_size_bits=67_108_864`
- `state_tracker_num_hashes=7`

## Why This Is Not A Huge Network

The raw state space blows up far faster than the network size should.

That is because the network is not trying to memorize every possible `5x5` state. If it had to do that, the problem would be hopeless. The whole point of the conv net is to learn reusable local scoring rules:

- matching neighbors
- cluster growth
- edge and corner effects
- when the current digit should be placed greedily
- when a low immediate score is worth it for future structure

So the architecture should scale with:

- board geometry
- number of candidate actions
- richness of local patterns

not with:

- the full combinatorial state count

## Why This Is Still A Large Step Up From 3x3

Even though this looks like a small architectural change on paper, it is not small in practice.

Compared to the `3x3` run:

- actions go from `9` to `25`
- episode length goes from `9` moves to `25`
- max digit goes from `4` to `9`
- replay memory gets `5x` bigger
- minibatch goes from `64` to `128`
- target sync interval gets looser in episode terms but still regular in step terms
- epsilon decays much more slowly

The training budget increase matters even more than the network increase.

Roughly:

- `3x3`: `200,000` episodes is about `1.8M` environment steps
- `5x5`: `500,000` episodes is about `12.5M` environment steps

That is about `7x` more interaction, which is more in line with how much harder `5x5` is.

## Why Normalize By Max Digit

For `3x3`, digits go up to `4`.

For `5x5`, digits go up to `9`.

If the network sees raw values, then moving from `3x3` to `5x5` changes the numeric input scale a lot. Normalizing by the max digit keeps inputs in a more stable range like `[0, 1]`.

That helps because:

- optimization is usually easier with bounded inputs
- the `3x3` and `5x5` setups become more comparable
- the model spends less effort adapting to raw value scale

This is a very cheap change and it is worth doing.

## What Not To Add Yet

Do not start with:

- residual blocks
- dueling DQN
- prioritized replay
- one-hot digit channels
- a second value head
- a much deeper or much wider model than this

Those may be worth trying later, but they make failure harder to interpret.

## Simple Read Of The Plan

The first `5x5` attempt should be:

- the same deep q setup that finally worked on `3x3`
- a moderately larger conv net
- normalized inputs
- much slower exploration decay
- much longer training

If that does not move clearly above random, then the next architectural thing to try is not "make it huge". The next thing to try is:

- give the conv stack earlier access to the current digit and next digit by turning them into input planes
