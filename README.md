# predict_plays

# TODO
```
Setup initial model
[ ] load and format data
[ ] define training dataset
[ ] setup initial scaffolding & model training

Measure success
[ ] Start to measure success

Ways to improve things
[ ] Expand categoriztion
```

## Expanded Categorization
### Original Idea
Run Left, Middle (between tackles), Right
Play Action Pass
Drop Back Pass, Rollout left, Rollout right
Screen Left, Middle, Right
QB Spike, QB Kneel

### Better Idea
Try to predict pff_runConceptPrimary, pff_runConceptSecondary, pff_runPassOption, 
Try to predict defense coverage: pff_passCoverage, pff_manZone

Try to predict rushLocationType (INSIDE_LEFT', 'INSIDE_RIGHT', 'OUTSIDE_LEFT', 'OUTSIDE_RIGHT)
Left and Right would depend on where you are, wrt hashes
Do we also get weather at time of game???