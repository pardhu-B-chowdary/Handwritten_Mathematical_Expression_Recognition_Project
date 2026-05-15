from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, join_room, emit
import os, requests
from model_inference import run_inference

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=100000000)

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =========================
# MODEL CONFIG
# =========================

USE_COLAB = True # False means use local model

COLAB_API = "https://snarl-legwarmer-unread.ngrok-free.dev/"


@app.route("/")
def index():
    return render_template("index.html")

def inference(filepath):

    if USE_COLAB:

        with open(filepath, "rb") as f:

            response = requests.post(
                COLAB_API,
                files={"image": f}
            )
        
        print(response.text)

        try:
            result = response.json()

            return result.get("latex")

        except Exception:

            print("Invalid Response:", response.text)

            raise Exception("Invalid response from Colab API")

    else:

        return run_inference(filepath)

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files.get("image")

    if not file:
        return jsonify({
            "error": "No file uploaded"
        }), 400

    filepath = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(filepath)

    try:

        latex = inference(filepath)

        return jsonify({
            "latex": latex
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

@socketio.on('join')
def on_join(data):
    room = data['room']
    join_room(room)

@socketio.on('sync_image')
def on_sync_image(data):
    room = data.get('room')
    image_data = data.get('image')
    if room and image_data:
        # Broadcast the image to everyone else in the room
        emit('sync_image', {'image': image_data}, room=room, include_self=False)

@socketio.on('sync_prediction')
def on_sync_prediction(data):
    room = data.get('room')
    latex = data.get('latex')
    if room and latex is not None:
        # Broadcast the prediction to everyone else in the room
        emit('sync_prediction', {'latex': latex}, room=room, include_self=False)

@socketio.on('sync_action')
def on_sync_action(data):
    room = data.get('room')
    if room:
        emit('sync_action', data, room=room, include_self=False)


if __name__ == "__main__":

    socketio.run(app, host="0.0.0.0", debug=True)