# LING-165-Project
Code runs on python 2.7 using nltk 3.2.5, numpy 1.13.1, and pyscipopt

pyscipopt requires the installation of the scip optimization suite (version 5.0.1)

First run seqtagger.py to obtain the trained sequential backoff tagger that will be used to tag the haikus

Then run structure.py. This should create a pickle file containing a list of 'templates' as described in the paper

Aterwards, run parameters.py to generate the parameters necessary to run the model

Lastly, run fullhaiku.py this should print the haiku
