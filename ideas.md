# Future Ideas & Experiments

## Ablations

- **Grayscale** - strip color, test if shape alone is enough
- **Silhouette only** - black filled sprite, pure shape
- **Generational split** - train on early gens, test on later gens (generalization test)
- **Frozen vs unfrozen backbone** - does fine-tuning all layers vs just the head matter
- **Shiny sprite test** - train on normal sprites, test on shiny (completely different colors)
  - Combined with Grad-CAM: compare attention maps on normal vs shiny versions of the same Pokemon
  - If attention shifts, model was color-dependent. If it stays the same, it learned shape.
  - This is a clean experiment for a paper

## More Data

- **Animation frames** - multiple frames per Pokemon already exist in split_sprites, dataset only uses one per Pokemon right now. Using all frames = ~10x more training data for free
- **Back sprites** - different angle, already in the sprite sheets

## Better Modeling

- **Grad-CAM attention maps** - visualize what part of the sprite the model actually looks at
- **Type-aware loss** - penalize bigger type mismatches more (Fire vs Ice worse than Water vs Ice)
- **Dual-head model** - separate output heads for primary and secondary type
- **Ensemble CNN + RF** - different models make different mistakes, combining could help
- **Domain-appropriate pretraining** - EfficientNet was trained on real photos not pixel art

## Interesting Questions

- Does accuracy correlate with type rarity? (rare types harder to predict)
- Do visually distinctive types (Fire, Ice) score higher than ambiguous ones (Ghost, Dark)?
- Cross-game generalization - train on one game's sprites, test on another
