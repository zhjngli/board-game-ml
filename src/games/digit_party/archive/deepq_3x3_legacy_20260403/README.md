# Legacy Digit Party DQN Checkpoints

This folder contains the old local `deepq_3x3_models/` and `deepq_3x3_memory/`
directories that were moved aside on April 3, 2026 so true DQN training could
restart from a clean slate.

## Why they were archived

The current `digit_party_deep_q` code now uses a different model construction
than the one these checkpoints were created with, so loading them into the
current network fails with shape mismatches.

## Expected legacy architecture

These checkpoints were produced by the older Digit Party DQN path that:

- used the `opt_nn_params` values as the DQN network params
- effectively built the convolution stack using `conv_filters` iterations
  instead of `conv_layers`
- wrote checkpoints into the original local `deepq_3x3_models/` folder

Relevant legacy parameter values:

- `conv_layers=8`
- `conv_filters=14`
- `dense_layers=1`
- `dense_units=337`
- `learning_rate=0.0009647204266707786`
- `batch_size=64`
- `epochs=42`
- `dropout_rate=0.06842343999759844`
- `output_activation="linear"`

## Likely source commit

These are most likely associated with code at or near the branch-base commit:

- `fe7bb20f2d156f647eda3acc6868d65caf08d0b4`

That is the merge-base of this branch with `main`.

## Fresh training

The active training path now starts fresh by recreating `deepq_3x3_models/`
and `deepq_3x3_memory/` as needed for the current architecture.
