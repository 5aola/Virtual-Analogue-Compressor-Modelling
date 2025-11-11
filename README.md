# Virtual Analogue Modelling of a Compressor

## Deliverables

1. Gather a subset of the dataset of Diff-SSL for experimenting
2. Extract gain reduction
3. Test models/do small trainings
    - build new
        - predict gain reduction and condition another network with that
    - Used models:
        - TCN
        - GCN
        - LSTM
        - ED
        - S6 ??
    - conditioning: FILM
5. Optimize models
6. Improve Loss function
    - RDC
    - FRAC
    - Spectral Flatness
7. Evaluate on bigger datasets and compare with other models
