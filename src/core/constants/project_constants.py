import pathlib

current_path = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = current_path.parent.parent

ARTIFACTS_DIRECTORY = pathlib.Path.joinpath(PROJECT_ROOT, "artifacts")
INPUT_DATA_DIRECTORY = pathlib.Path.joinpath(ARTIFACTS_DIRECTORY, "input_data")
OUTPUT_DATA_DIRECTORY = pathlib.Path.joinpath(ARTIFACTS_DIRECTORY, "output_data")

MODEL_DATA_DIRECTORY = pathlib.Path.joinpath(ARTIFACTS_DIRECTORY, "models")

TOPIC_CATEGORIES = ['employee_benefits',
                    'employee_training',
                    'other',
                    'payroll',
                    'performance_management',
                    'talent_acquisition',
                    'tax_services',
                    'time_and_attendance']

TOPIC_CATEGORIES_PRETTY = ['Employee benefits',
                           'Employee training',
                           'Topics not supported',
                           'Payroll',
                           'Performance management',
                           'Talent acquisition',
                           'Tax services',
                           'Time and attendance']
