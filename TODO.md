# Modelling TODOs

## Evaluation
- [ ] Split video segments into training and test splits + holdout for all model training and evaluation
- [x] Add evaluation metrics for direction and circle count.

## Traditional ML
- [ ] Improve XGBoost performance
    - [ ] Run stage 1 labeling, use non overlapping results as "difficult" true negative samples
    - [ ] Tune XGBoost hyper params
    - [ ] Build more models for direction and circle count
- [ ] Improve rule-based classifier
    - [ ] Handle M-F pairing, multiple fish on screen
    - [ ] Output direction
    - [ ] Output circle count
    - [ ] Check improvement from reflection removal

## Neural Methods
- [ ] Data augmentation
    - [ ] Translation, inversion, rotation of video samples
    - [ ] Follow paper parameters, 40M param for 600h of videos (check scaling laws)
- [ ] Reimplement ASFormer, MSTCN, I3D, etc
    - [ ] Select a good model, attempt to beat rule-based classifier

## Moonshot
- [ ] Use a multimodal LLM (gemma 4b?)
    - attempt to transfer learn video understanding in LLM for our segmentation task
