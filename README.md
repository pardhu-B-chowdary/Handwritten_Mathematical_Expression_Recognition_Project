This Repo is part of My M.Tech Major Project contating files of the program and model.

This project is a deep learning-based system for recognizing handwritten and printed mathematical expressions from images and converting them into structured LaTeX format. The system uses a Vision Encoder–Decoder Transformer for mathematical expression recognition, with image preprocessing and beam search decoding to improve prediction quality. The generated LaTeX can be viewed as editable code and rendered visually using MathJax through a Flask-based web interface.

# Setup
1. Create a Virtual Environemtn and activate it <br>


    ```
    python -m venv .venv
    .venv\Scripts\activate
    ```
    
    the slash change based on IDLE and for antigravity the activation should start from '.\.venv...'

2. We First install all neccessary Requirements
     
    ```
    pip install -r requirements.txt
    ```

3. Install the Model at by running the _model_inference.py_
     
    ```
    python model_inference.py
    ```

4. Start the app
   
     ```
     python app.py
     ```
   
