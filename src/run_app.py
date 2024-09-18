# -*- encoding: utf-8 -*-
"""
Streamlit command to run the Main Script
"""
import os

os.system("streamlit run main_app.py --server.port 8081 --logger.level=debug")
