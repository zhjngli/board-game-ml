# Digit Party Deep Q Notes

This document is a deeper read on:

- what Q-learning is
- how deep Q-learning differs from tabular Q-learning
- how Digit Party fits the reinforcement learning setup
- why the current `3x3` deep Q setup is probably underperforming
- what to verify before making architectural changes

This is analysis only. It does not propose code changes yet.

## Goal

The practical goal is:

- get `3x3` deep Q to consistently approach the strong behavior already seen with tabular Q-learning
- use that as a correctness target before moving to `5x5`

That order matters because if `3x3` is still unstable, then `5x5` results are hard to interpret. A bad `5x5` result could mean:

- the algorithm is wrong
- the state encoding is weak
- the hyperparameters are poor
- the training schedule is too small
- or `5x5` is simply harder

`3x3` is the easier debugging target because the simple Q learner already shows that strong play is reachable in this game family.

## What The Agent Should Observe

The intended constraint is:

- the agent should only observe what a human player can see

For the current game implementation, that means the learning state should be limited to:

- the board
- the current digit
- the next digit

This matches the current representation in [game.py](./game.py), where `DigitPartyIR` stores the board and the `next` tuple, and where `next_digits()` exposes the current and next digit.

That is a reasonable constraint if the goal is to learn the same game humans are actually playing. It does mean the environment is not fully observable with respect to the entire future digit sequence, but that is acceptable if the game itself only reveals two digits.

## What Q-Learning Is

Q-learning tries to learn an action-value function:

`Q(s, a)`

This means:

- `s` is a state
- `a` is an action available in that state
- `Q(s, a)` estimates the long-term value of taking action `a` in state `s`

The core idea is not "what reward do I get right now", but:

- "if I take this action now, how good is the future that follows?"

The classical update is:

```text
Q(s, a) <- (1 - alpha) * Q(s, a) + alpha * (reward + gamma * max_a' Q(s', a'))
```

Where:

- `alpha` is the learning rate
- `gamma` is the discount factor
- `reward` is the immediate reward from the transition
- `s'` is the next state
- `max_a' Q(s', a')` is the best estimated future value after arriving in `s'`

This is an off-policy algorithm. That means:

- the learner is allowed to explore with one behavior
- while still updating toward the value of the greedy future action

In plainer language:

- explore now
- learn as if you will behave better later

## How The Simple Q Version In This Repo Works

The tabular learner in [q.py](../../learners/q.py) stores a dictionary of Q-values for every visited state-action pair. In Digit Party, that means:

- for each board plus visible-next-digit situation
- store a score estimate for each placement

In [train_q_simple.py](./train_q_simple.py), the Digit Party trainer uses:

- the current score difference as the immediate reward
- the next state as the bootstrap target

That works very well for `3x3` because the state space is still small enough that the learner can effectively memorize a huge fraction of what it sees.

Why it works:

- the game is deterministic once the visible state is known and a move is chosen
- local reward is meaningful because points are earned immediately when matches are created
- enough repeated experience eventually fills in a strong table

Why it becomes impractical:

- the state space grows very quickly
- the table becomes huge
- memory and training time explode

That is exactly why the `3x3` policy file can become enormous while still performing very well.

## What Deep Q-Learning Changes

Deep Q-learning keeps the same basic target as tabular Q-learning, but replaces the explicit lookup table with a neural network:

`Q_theta(s, a)`

Now the model tries to approximate the Q-values rather than storing every state separately.

That has the big advantage you care about:

- memory is mostly in the network weights rather than in a giant table

The intended benefit is not just compression. It is also generalization:

- if two states are similar, the network can produce similar Q estimates

That is something the tabular version does not do naturally. The tabular learner mostly memorizes exact situations.

## Why DQN Needs Extra Machinery

A neural network makes Q-learning less stable. The usual DQN fixes are:

- epsilon-greedy exploration
- replay memory
- a target network

### Epsilon-greedy

Most of the time:

- choose the action with the highest predicted Q-value

Sometimes:

- choose a random action instead

This prevents the policy from getting stuck too early and helps it discover better trajectories.

### Replay memory

Instead of training only on the latest move:

- save many transitions `(state, action, next_state, reward, done)`
- sample random minibatches from that memory

This matters because sequential game data is highly correlated. Random replay makes updates less brittle and reuses old experience efficiently.

### Target network

If the same network both predicts current Q-values and defines the bootstrap target, the target shifts every time the model changes. That can destabilize learning.

So DQN usually keeps:

- an online network that is actively trained
- a separate target network that is updated less often

This makes the bootstrap target more stable.

## How Digit Party Maps To Q-Learning

Digit Party is a single-player sequential decision problem.

A natural RL mapping is:

- state: board plus currently visible digits
- action: choose an empty tile
- reward: score gained by placing the current digit there
- episode end: board is full

This is actually a nice fit for Q-learning because the reward is not purely terminal. Many placements immediately change the score.

That said, the game still has delayed consequences:

- a move that scores nothing now may create better future adjacency opportunities
- a move that scores now may destroy future structure

That is exactly the kind of tradeoff Q-values are meant to learn.

## Why The Supervised `Simple Q -> Neural Net` Path Helps But Does Not Solve DQN

The supervised path in [train_deep.py](./train_deep.py) uses the tabular Q results as training labels for a network.

That is useful because it can answer:

- can this network architecture represent a decent approximation to the strong tabular policy?

If the answer is yes, then the architecture is at least expressive enough to imitate a good policy on `3x3`.

But that does not guarantee the same setup will work for DQN. The objectives are different:

- supervised fitting learns from a large static labeled dataset
- DQN learns from non-stationary self-generated data and bootstrapped targets

A network and hyperparameter set that works for offline imitation may be poor for online temporal-difference learning.

That distinction matters a lot in this repo.

## Why The Current DQN Setup Is Probably Stalling

The main issue is that the current deep Q experiment appears to have several implementation-level problems that are more fundamental than architecture search.

## 1. The Training Schedule Barely Updates The Model

The current `3x3` DQN setup is defined in [train_q_deep.py](./train_q_deep.py).

Important settings:

- `training_episodes = 5000`
- `steps_to_train_longterm = 15041`
- `steps_to_train_shortterm = 0`
- `minibatch_size = 15041`

In a `3x3` game there are only 9 moves per episode, so `5000` episodes is only about `45,000` environment steps.

That implies:

- replay only runs about 2 times over the entire training job

This is likely the single clearest reason the model may look like it is not improving.

If the network is only updated around twice, then:

- it has almost no chance to converge
- epsilon decay can outrun learning
- target network synchronization becomes mostly irrelevant

This is visible in [deep_q.py](../../learners/deep_q.py), where long-term replay only runs when `self.steps % self.steps_to_train_longterm == 0`, and short-term replay is disabled.

## 2. Training Chooses Invalid Actions

During training in [deep_q.py](../../learners/deep_q.py):

- random exploration samples from all action indices
- greedy selection uses `argmax` over the full predicted policy

That means the learner is allowed to choose already-filled tiles during both exploration and exploitation.

Invalid actions are punished, but they still damage learning because:

- they consume training time
- they distort the behavior policy
- they make the replay distribution less about real game decisions and more about avoidable illegal moves

In contrast, evaluation in [train_deep.py](./train_deep.py) masks to valid actions before taking `argmax`.

So the agent is trained under one action rule and evaluated under a different one.

That mismatch is a strong sign that the measured training dynamics are not aligned with the intended game policy.

## 3. The Bellman Target Uses Invalid Next Actions

The current replay logic computes:

`reward + gamma * max(next_qs)`

But the `max` is taken over every action index, not just valid next actions.

That means the target can bootstrap from illegal next moves.

Why that matters:

- the model may assign artificially high Q-values to impossible actions
- those illegal values then infect the Bellman target
- the network learns toward unreachable futures

This is a common DQN failure mode in games with masked action spaces.

## 4. The DQN Uses Hyperparameters Borrowed From A Different Problem

The network parameters in [train_q_deep.py](./train_q_deep.py) come from `opt_nn_params` in [train_deep.py](./train_deep.py).

Those parameters were found for supervised fitting against chunked simple-Q data.

That is not the same optimization problem as DQN.

Some examples of why the transfer is risky:

- `epochs = 42` per training call is normal in supervised fitting, but usually too much for online DQN updates
- dropout can be useful for supervised generalization, but can add noise to already-noisy TD targets
- batch normalization can complicate online RL because minibatch statistics shift during training
- a huge minibatch can make learning slow and infrequent rather than responsive

So even if the architecture is strong for imitation learning, it may still be a poor DQN training configuration.

## 5. There Is A Likely Bug In The Convolutional Layer Search

In [train_deep.py](./train_deep.py), the model parameters include:

- `conv_layers`
- `conv_filters`

But the network construction loops over `self.params.conv_filters` instead of `self.params.conv_layers`.

That means:

- the searched number of convolutional layers is effectively ignored
- the model depth is tied to the number of filters instead

So the Bayesian search space is not really being evaluated as intended.

That makes the architecture results harder to trust.

## 6. Replay Memory Stores Mutable Game State Objects

In [game.py](./game.py), `state()` returns a `DigitPartyState` that reuses:

- `self.board`
- `self.digits`

Those are mutable objects.

In [deep_q.py](../../learners/deep_q.py), replay memory stores `state` directly.

That is risky because replay memory should ideally capture a frozen snapshot of what the agent saw at the time of the transition.

Even if this is not currently causing a hard bug, it is a fragile design for experience replay.

The immutable representation already exists in `DigitParty.to_immutable()`, which is a sign that the codebase already recognizes this need in other places.

## 7. The Value Head Does Not Help The DQN Objective Much

The model outputs:

- a `policy` vector
- a scalar `value`

But the DQN update only directly edits one chosen action in the `policy` output.

The scalar value head is not meaningfully used to form the Bellman target for the Q update.

So the model is carrying extra output structure without a clear DQN benefit.

That does not necessarily make the method fail, but it adds complexity without a strong reason.

## 8. The Model May Be Learning "Illegal Action Avoidance" More Than "Good Placement"

Because invalid actions are frequent in training, the network may spend much of its capacity learning:

- which actions are illegal

instead of learning:

- which valid placements are strategically strongest

That distinction matters. On a nearly full board, avoiding one illegal move is much easier than ranking the remaining legal moves correctly.

The current setup may therefore reward the wrong kind of progress.

## Why `3x3` Should Be Solved Before `5x5`

`3x3` is small enough that:

- the tabular learner already acts like a high-quality oracle
- the state/action space is still tractable for analysis
- failures are easier to interpret

For `5x5`, it becomes much harder to tell whether failure is due to:

- representation limits
- insufficient compute
- weak exploration
- poor reward shaping
- or a still-broken DQN loop

So the best scientific order is:

1. make the DQN loop mechanically sound on `3x3`
2. verify it against strong tabular behavior
3. then move to `5x5`

## What "Real Learning" Means Here

You asked whether richer hidden information would still count as real learning.

That is a good concern.

If the network sees more information than a human sees, then it is learning a stronger decision problem than the one a human player actually faces. That can still be interesting, but it is not the same game.

If the goal is:

- learn to play the visible game well

then the right constraint is:

- only use the observable board plus visible upcoming digits

That is the constraint assumed in this document.

## A Good Mental Model For Your Two Existing Paths

Think of the current project as having two separate questions:

### Path 1: Tabular Q-learning

Question:

- if we memorize enough exact states, can we learn strong `3x3` play?

Current answer:

- yes, apparently very well

### Path 2: Supervised network trained from tabular data

Question:

- can a neural network compress and approximate the strong tabular policy?

Current answer:

- yes, partially
- good enough to beat random
- not yet as strong as the table

### Path 3: True DQN self-learning

Question:

- can the network discover that strong policy from its own self-generated experience?

Current answer:

- unclear, because the current loop likely has enough mechanical issues that poor performance does not yet isolate the architecture question

## What To Verify Before Trusting Any DQN Result

These are the highest-value checks to run conceptually before spending more laptop time on large training jobs.

### Check 1: How often does the model actually update?

You want to know:

- total environment steps
- total replay calls
- total gradient updates
- how often target weights sync

If those counts are tiny, architecture tuning is premature.

### Check 2: Are invalid actions masked everywhere that matters?

You want consistency in:

- behavior policy during exploration
- behavior policy during greedy action choice
- Bellman target over next-state actions
- evaluation-time action choice

If the masking rules differ, training becomes hard to interpret.

### Check 3: Is replay storing frozen states?

Replay should represent the actual historical transition, not an object that may later mutate.

### Check 4: Is the network being trained for DQN or for offline imitation?

The following settings should be treated differently depending on the problem:

- number of epochs per update
- batch size
- dropout
- batch normalization
- target outputs

### Check 5: Can the network imitate the tabular action ranking on a fixed validation set?

This is a useful bridge test.

Before asking DQN to discover the policy from scratch, first ask:

- when handed a strong target, how well can the network reproduce the best move ordering?

That is a cleaner architecture test than self-play score alone.

## Recommended Learning Path

If you want to build intuition rather than only patch code, this is the learning order I would use.

### 1. Re-derive the tabular update

Take one Digit Party move and write down:

- current state
- chosen action
- immediate score gain
- next state
- best next action estimate

Then manually compute one Q update with the classical formula.

That makes the Bellman target concrete.

### 2. Compare tabular Q-learning to DQN conceptually

Ask:

- what does the table store exactly?
- what does the network approximate instead?
- what information is lost when exact lookup becomes function approximation?

The answer is:

- the table keeps exact per-state values
- the network trades exactness for compression and generalization

### 3. Study action masking

For board games with illegal actions, this is essential.

If invalid actions are allowed into either:

- action selection
- or target maximization

the Q-values can become misleading very quickly.

### 4. Study replay and target networks

These are not optional extras. They are stability tools added because naive neural-network Q-learning is often unstable.

### 5. Separate representation questions from algorithm questions

A bad result can come from:

- poor state encoding
- poor DQN mechanics
- poor optimization settings

You want tests that isolate each one.

## Summary

The current evidence suggests:

- tabular Q-learning is strong on `3x3`
- the supervised network can partially compress that policy
- the true DQN path is probably underperforming for mechanical reasons before we even reach the architecture question

So the next analytical step is not "search a bigger network".

It is:

- verify that the DQN loop is actually performing frequent, valid, meaningful Bellman updates on legal actions only

Once that is true, architecture search becomes much more informative.

## The Three Learning Paths In This Repo

There are really three different projects living under the same Digit Party umbrella. They are related, but they are not the same problem.

### 1. Tabular Q-learning

Question:

- can we learn strong play by storing explicit Q-values for exact state-action pairs?

What it does:

- stores a table entry for each observed state-action pair
- updates that table with the Bellman target
- chooses only legal actions when acting

Strengths:

- very direct and easy to reason about
- gets very strong on `3x3`
- useful as an oracle or reference policy

Weaknesses:

- state explosion
- terrible memory scaling
- not realistic for `5x5`

This is the baseline proof that the game is learnable at least in the small case.

### 2. Supervised Imitation Of The Tabular Policy

Question:

- can a neural network compress the strong tabular policy into weights?

What it does:

- treats the tabular learner as a source of labeled training data
- trains a network offline on saved states and target values
- checks whether the network can approximate the strong table

Strengths:

- useful architecture sanity check
- useful data representation sanity check
- can beat random without the full memory cost of the table

Weaknesses:

- does not prove that online DQN will work
- trains on static labels, not self-generated TD targets
- can look decent while true DQN still fails

This is best thought of as:

- an approximation/compression experiment
- and possibly a warm-start or diagnostic tool

It is not the final goal.

### 3. True DQN Self-learning

Question:

- can a neural network learn the action-value function from its own experience, without relying on the tabular table as the main teacher?

What it does:

- interacts with the environment directly
- stores transitions in replay memory
- updates the Q-network from Bellman targets
- uses a target network for stability

Strengths:

- much closer to the intended scalable solution
- memory is in the weights and bounded replay memory
- can, in principle, move from `3x3` to `5x5`

Weaknesses:

- much more fragile to implementation mistakes
- much more sensitive to action masking, replay cadence, and optimization choices
- harder to debug because both the data distribution and the targets change during training

This is the main target of future code changes.

## Practical Relationship Between The Three Paths

The right way to use these paths together is:

1. tabular Q proves that strong `3x3` play exists in the chosen observable game setting
2. supervised imitation checks whether a neural network can represent something close to that policy
3. true DQN tries to learn a strong policy from experience rather than from the table

So the table is:

- a proof of learnability on `3x3`

The supervised network is:

- a representation sanity check

And true DQN is:

- the actual algorithmic goal

That is the frame I would keep in mind going forward.

## Plan Toward True DQN

The plan below is ordered to maximize signal and minimize wasted training time.

### Stage 1: Fix DQN Mechanics First

These are the low-hanging fruit changes that should happen before further architecture tuning.

- mask invalid actions during epsilon exploration
- mask invalid actions during greedy action selection
- mask invalid actions when computing the Bellman `max` over next actions
- store immutable or deep-copied replay states instead of mutable live objects
- separate DQN training settings from the supervised network settings
- add simple counters or logs for total steps, replay calls, target syncs, and illegal-action rate

Why this stage comes first:

- if these are wrong, the network can fail even if the architecture is fine

### Stage 2: Make DQN Updates Frequent And Cheap

The current replay cadence is too sparse, and the current training call is too heavy.

The next step should be to:

- reduce the replay interval dramatically
- use much smaller minibatches
- avoid training for many epochs per replay call
- treat each replay step as a small TD update, not a large offline fitting pass

Conceptually, the goal is:

- many small online-ish updates

not:

- two giant supervised retraining events

### Stage 3: Simplify The DQN Model For The DQN Objective

Once the mechanics are sound, revisit the model itself.

Likely questions to test:

- should the DQN use only a Q head instead of both policy and value heads?
- should dropout be removed for DQN?
- should batch normalization be removed for DQN?
- should the digit inputs be encoded differently or more explicitly?
- should the network be shallower and updated more often rather than deeper and updated rarely?

The bias here should be:

- simpler and more stable first

not:

- larger and more expressive first

### Stage 4: Verify On `3x3` With Stronger Metrics

Average score alone is not enough.

The verification pass should include:

- average percent of theoretical max over a fixed evaluation set
- illegal action frequency during training
- illegal action frequency during greedy evaluation
- replay/update counts
- comparison against random baseline
- optional comparison against a fixed sample of tabular best actions

The main question at this stage is:

- is the DQN genuinely learning better placement strategy, or only avoiding obvious mistakes?

### Stage 5: Use Supervised Fitting Only As A Support Tool

Supervised fitting can still help, but only in a supporting role.

Good uses:

- check whether the network can represent the table at all
- create a fixed validation set of strong states/actions
- possibly warm-start the network before online DQN

Less useful use:

- treating supervised success as proof that DQN is solved

It is a bridge, not the destination.

### Stage 6: Move To `5x5` Only After `3x3` Is Mechanically Sound

Once `3x3` DQN is stable and clearly above random for the right reasons, then it becomes worth porting the method to `5x5`.

At that point, the main new questions become:

- whether the same observable-state formulation scales
- whether the network needs a more structured encoding
- whether laptop-scale training is enough

But those are later questions. First the `3x3` pipeline has to be trustworthy.

## Immediate Coding Order

If the next step is implementation, the coding order I would use is:

1. fix action masking in true DQN behavior and targets
2. freeze replay memory states properly
3. make replay updates much more frequent and much smaller
4. decouple DQN network/training params from the supervised imitation params
5. add basic training instrumentation so results are interpretable
6. then run short `3x3` experiments before touching deeper architecture questions

That should give the highest return per line of code.
