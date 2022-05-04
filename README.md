# FYP_submission
Code for my final year project submission, to obtain an MEng in Mechanical Engineering. This project focuses on ML methods for the automation of the English Wheel. Files worth of note are described below. Most files are fully commented.

A) For the G-code interpreter to enable actuation of the English Wheel:

1) Open 'G_code_generation' folder

2) To run app open 'WheelingProfileVisual.mlapp'. This will open MATLAB's app designer with the application ready to start. Press the 'play' button. The app is now running. For user features see the user documentation in the appendix of the final report pdf.

Note: This is application is not yet complete. Further work is required for full implementation on the English Wheel.

B) For processing of data such as data orientation: 

1) Open 'data_processing' folder

2) 'Preprocess.m' - This holds different functions for preprocessing data. Quite messy, needs tidying and refining.

3) 'GenData.m' - This is used to create synthetic data using strategies proposed in the project report.

Note: Synthetic data is not included. Data either has to be generate using 'GenData.m' or collected from the machine.

C) For machine learning architecture:

1) Open 'ml_architecture' folder
 
2) 'ml_arch.py' - Architecture for the machine learning model. Quite messy, needs tidying and refining. Includes unit and integration test functions. These should be standardised.

3) 'framework_analysis.py' - Test script for obtaining data from the architecture.

Note: These files require data to be stored in a particular format. Directory paths need to be changed before use.
