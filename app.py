from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from model import ImageCaptionGenerator

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SESSION_SECRET', 'dev-secret-key')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

print("Loading Image Caption Generator model...")
caption_generator = ImageCaptionGenerator()
print("Model loaded successfully!")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_caption', methods=['POST'])
def generate_caption():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if not file.filename or file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400
    
    try:
        filename = secure_filename(file.filename)
        if not filename:
            return jsonify({'error': 'Invalid filename'}), 400
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        caption = caption_generator.generate_caption(filepath)
        
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'caption': caption
        })
    
    except Exception as e:
        return jsonify({'error': f'Error generating caption: {str(e)}'}), 500

@app.route('/generate_caption_gallery', methods=['GET'])
def generate_caption_gallery():
    image_name = request.args.get('image')
    
    if not image_name:
        return jsonify({'error': 'No image specified'}), 400
    
    try:
        filepath = os.path.join('static', 'images', secure_filename(image_name))
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'Image not found'}), 404
        
        caption = caption_generator.generate_caption(filepath)
        
        return jsonify({
            'success': True,
            'caption': caption
        })
    
    except Exception as e:
        return jsonify({'error': f'Error generating caption: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': 'loaded'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
