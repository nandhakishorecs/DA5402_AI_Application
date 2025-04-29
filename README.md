# DA5402 Machine Learning Operations Laboratory 

Offered as a part of Masters' in Technology program at Indian Institute of Technology Madras (IIT-M). <br>

Course Instructor: **Dr Sudarsun Santhiappan** <br>
Submitted by: **Nandhakishore C S** <br>
Roll Number: **DA24M011**

## IPL Score Predictor

- This repository contains code for a web application which predicts score of an IPL match using a Deep Learning Neural Network built using PyTorch. 

- The application uses the data from an Airflow data pipeline. It is hosted using FastAPI backend, and frontend made with HTML, CSS and Javascript.

## Setup 

The application is packaged as a docker image and saved as **.tar file**. The application can be run in two ways: 

1. Building docker image and running app in the local device: 
    1. Download this repository using GitHub ssh 
    ```console
    $ git clone git@github.com:nandhakishorecs/DA5402_AI_Application.git
    $ cd DA5402_AI_Application 
    ```
    2. Initialise Docker in your system. This can be done in two ways: 
        - Using CLI: <br>
            In Linux: <br>

            ```console 
            $ sudo systemctl start docker
            ``` 
            In MacOS: <br>
            ```console 
            $ open -a "Docker"
            ```
        - Using Docker's UI installed in the local system. <br>
    3. This application requires docker version 28.0.4 or more. Check for Docker's version and update if needed. 
    4. Navigate to the root directory and build the docker image to run the application: 
    ```console
    $ docker build -t webapp . 
    $ docker run -p 8000:8000 -it da5402_ai_application-app
    ```
    5. Using docker's compose up and compose down to start and stop the container. 
    ```console
    $ docker compose up 
    $ docker compose down
    ```
    6. The web app can be accessed in the following url: [http://localhost:8000/](http://localhost:8000/)
2. Download the docker image from this repository and load the docker image and run the container 
    1. Download the docker image file (.tar) file and save it in a directory in the local computer
    ```console
    $ docker load -i ipl_app.tar  
    ```
    2. Run the docker image to run the app: 
    ```console
    $ docker run -it ipl_app:latest
    ```
    3. Using docker's compose up and compose down to start and stop the container. 
    ```console
    $ docker compose up 
    $ docker compose down
    ```
    4. The web app can be accessed in the following url: [http://localhost:8000/](http://localhost:8000/)

### Accessing Metrics 
- To setup monitering in the local device, the metrics of the web application can be accessed using: [http://localhost:8000/metrics](http://localhost:8000/metrics)

