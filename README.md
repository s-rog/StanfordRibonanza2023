# StanfordRibonanza2023
This repo contains the code to recreate my models in the Stanford Ribonanza RNA Folding competition 2023 5th Place Solution on a single 4090.

### Summary
1. A 5-fold ensemble is trained on the competition training data then used to create pseudo-labels by inferencing test data. The checkpoints are discarded.
2. The pseudo-labels are filtered then used to train a single model to serve as a pretrained checkpoint for the next step.
3. A 5-fold ensemble is trained on the competition training data from the pretrained checkpoint. These are the final models.

### Hyperparameters
Hyperparameters are set in `trainer.py` and it also serves as the entry point for training then inference. Included in the repo are the hparam yaml files from the 3 stages above.

### Quickstart
1. Install `requirements.txt`
2. Download the preprocessed data from this kaggle dataset (WIP) and put them in `data/`
3. Preprocess Eternafold BPP files (WIP)
4. Set hyperparamers in `exp/trainer.py` according to the desired training stage
5. (`chmod +x trainer.py`) `./trainer.py`