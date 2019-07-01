# Recsys2019
Mike O'Malley's and my joint entry to the 2019 recommender systems challenge

# Summary 

Mike and I used a light gradient boosted regression model to predict which item a user was most likely to click out on. We used an NGCD-
acquisition function as it has previously been found to correlate very well with the MRR. It got a leaderboard score of 0.692.

Below you will find steps reproducing our solution. Please don't hesitate to get in touch!

# Reproducing the Solution

1) This solution requires approximately 400GB of RAM. It takes about 36h to train using 40 cores. 
2) Put all data into the data folder.
3) Run preprocessing_scripts/update_metadata.py
4) Run model/Main.py
5) The predictions can now be found in the predictions folder. 


