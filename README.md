# Brain States prediction with physiological data
## Installation
1. Clone this repository on your local machine.
2. Go to the newly created repository `cd ~/myprojects/brain_states_prediction_physio` (replace `myproject` by the folder name where you have all your projects. I assumed it's in your home folder thus the tilde)
3. Activate your python environment
4. You will need to install bids_explorer:
    4.1 clone the repository in your projects folder (no this one. Go to the
    parent folder).
    4.2 go to the cloned repository
    4.3 follow the instruction at https://github.com/Sam54000/bids_explorer
5. Go back to the `brain_states_prediction_physio` folder
6. Run `pip install -e .`

## Organisation of the codebase
### sample_data
Here are data to benchmark our codes. It is in BIDS format and shouldn't be
renamed or moved.
- 
### labs
What I call labs are where we write our notebooks scrap codes. It is the area
we prototype our code.
### src
Here are the working functions, classes and pipelines to be used.
### viz
Everything related to plotting.