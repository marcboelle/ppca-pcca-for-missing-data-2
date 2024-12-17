# ppca-pcca-for-missing-data-2

GitHub repository for our project on Probabilistic PCA and Probabilistic CCA, from the MVA course "Introduction to Probabilistic Graphical Models and Deep Generative Models". We worked on the papers ["Probabilistic Principal Component Analysis"](https://www.robots.ox.ac.uk/~cvrg/hilary2006/ppca.pdf) by Tipping and Bishop, and ["A Probabilistic Interpretation of Canonical Correlation Analysis"](https://www.di.ens.fr/~fbach/probacca.pdf) by Bach and Jordan. We implemented EM algorithms for both methods, able to handle input datasets with missing entries. 

The file [models.py](./models.py) contains three classes: PCA, PPCA and PCCA, which can be used as `scikit-learn` classes with `.fit_transform(X)`.

The notebooks from the [notebooks/](./notebooks/) folder show how to use the code on simple examples.
