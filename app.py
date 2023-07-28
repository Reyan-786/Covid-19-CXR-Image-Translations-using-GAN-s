from flask import Flask, request, jsonify, send_file, render_template
from PIL import Image
from helper import preprocess_image, serve_pil_image, tensor_to_image, load_model
import tensorflow_addons as tfa
import instancenormalization
app = Flask(__name__)

@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/api/transform', methods=['POST'])
def transform_image():
    # Check if the request contains an image file
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'})

    image_file = request.files['image']
    
    # Check if the file is an image
    # if not is_image_file(image_file):
    #     return jsonify({'error': 'Invalid file type. Only image files are allowed.'})

    # Load the image using PIL
    img = Image.open(image_file)
    
    # Perform your image transformation operations here
    img = preprocess_image(img)

    # custom_objects = {'InstanceNormalization': tfa.layers.InstanceNormalization}
    model_AB = load_model("C:\\Users\\asus\\Desktop\\All GAN work\\CXR Translations trained models\\g_model_AtoB12500.h5", custom_objects=instancenormalization)
    pred = model_AB.predict(img)
    img = tensor_to_image(pred)
    # Return the transformed image as a response
    return serve_pil_image(img)


def is_image_file(file):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in file.filename and \
           file.filename.rsplit('.', 1)[1].lower() in allowed_extensions


if __name__ == '__main__':
    app.run()
