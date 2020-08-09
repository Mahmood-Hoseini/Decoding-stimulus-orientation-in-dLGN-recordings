# Decoding stimulus orientation in dLGN recordings
This notebook is used to analyze data presented in Figure 9 of the following paper: 

https://www.biorxiv.org/content/10.1101/2020.05.06.081877v1.full.pdf


![Figure 9. Optogenetic perturbation of SOM nRT neurons diminishes spiking reliability and stimulus location encoding ability of dLGN cells.](https://github.com/Mahmood-Hoseini/Decoding-stimulus-orientation-in-dLGN-recordings/blob/master/FINAL-Fig9-LGN-CV%20and%20decoding%20acc%20from%20NN.png)


Since most dLGN units are not orientation or direction selective, predicting stimulus orientation using the LDA-LOOXV approach is not appropriate. Instead, we decided to use patterns of correlations in single-units responses to decode stimulus identity. First, we selected all the on-center single-units that were not orientation selective from all the recordings in each condition. Using these non-simultaneously recorded neurons, we computed correlations among all possible pairs (28 pairs with optogenetic suppression and 21 with optogenetic activation of SOM nRT neurons) across roughly randomly selected trials for each stimulus orientation. Now the data is divided into 3 sets of training (80%), cross-validation (10%) and testing (10%). Then, a deep neural network was trained on the training set and its hyperparameters were tuned using its performance on the cross-validation dataset. Finally, the network predictions were tested using the test set.
