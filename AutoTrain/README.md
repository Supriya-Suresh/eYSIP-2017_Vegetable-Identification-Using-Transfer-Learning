This folder contains code for running a server to accept images and scripts for adding cron tasks to run the autotraining code every 4 days.
The subfolders are listed below.

# Trainer
In this folder, `transferveg.py` is the main script which should be run every 4 days to Retrain the model.

# Images
Create a folder called Images which should contain all the test images which are labelled as described in the wiki.
Note that here the folder is called Images rather than veg.

# Logs
Everytime `transferveg.py` is run, all debugging information is output to this folder. This folder is automatically created.
