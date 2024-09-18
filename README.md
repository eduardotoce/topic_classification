
### Setting the project 
#### Create conda env:
    Run code in conda/create_env.sh

#### Config project:
Marksource root the src folder or export the topic_classificacion/src to PYTHONPATH variable. or (for linux) run in terminal

    export PYTHONPATH="${PYTHONPATH}:/{your-project-location-path}/topic_classification/src/"

 To execute the train pipeline locate in src/ folder and run in terminal

    entrypoint.py train 

To run the test pipeline locate in src/ folder and run in terminal 

    entrypoint.py test 

To execute the Streamlit API locate in src/ folder and run in terminal 

    run_app.py

It will open a web interface in you browser.
