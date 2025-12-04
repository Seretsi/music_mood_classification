# music_mood_classification
Project to build a finetuned ensemble music-mood classifier.

## Project organization

### Ensemble
The ensemble code is executed mainly within the jupyter notebooks at the root of the project `main` and `main_torch`. `main` hosts the complete regression classifier code we started with in our first checkpoint. `main_torch` was our progress towards a neural network approach. The neural net is not complete by the time of submission.
The implementation files are:
- `FAI_Sklearner.py`: this holds the regression logic based on scikit-learn packages. It uses to separate classifiers to learn a best fit line across the arousal and valence values independantly.

    Call ``FAI_Sklearner.FAI_Sklearner.train(...)`` to fit lines of best fit over the valence and arousal predications.
    
    Call ``FAI_Sklearner.FAI_Sklearner.validate(...)`` to validate learned performance against validation data.
    
    Call ``FAI_Sklearner.FAI_Sklearner.predict(...)`` to generate predictions based on input arrays of valence and arousal predictions.

- `FAI_LinRegClassifier.py`: contains code for a regression classifier we were constructing from scratch with no external libraries bar, numpy and, panda.
- `FAI_NN_Learner`: contains the in progress neural network code.

### Audio Feature extraction and Classification

`FAI_Final_Project_Audio.ipynb` is our jupyter notebook entry point into our audio feature extraction and classification code.

### Lyrics Feature extraction and Classification

