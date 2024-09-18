
## Project Setup

### Create Conda Environment
To set up the project environment, run the script located in conda/create_env.sh:

    conda/create_env.sh

### Config project:
##### Option 1: Mark Source Root
Mark the src/ folder as the source root in your development environment.

#### Option 2: Set PYTHONPATH
For Linux users, you can export the PYTHONPATH to include the src/ folder:

    export PYTHONPATH="${PYTHONPATH}:/{your-project-location-path}/topic_classification/src/"

#### Running the Pipelines
##### Training Pipeline
Navigate to the src/ folder and run the following command to execute the training pipeline:

    entrypoint.py train 

##### Testing Pipeline
To run the testing pipeline, use this command:

    entrypoint.py test 

### Run the Streamlit API
To start the Streamlit web interface, navigate to the src/ folder and run:

    run_app.py

It will open a web interface in you browser.
