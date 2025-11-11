
# 🧠 Image Caption Generator (CNN + LSTM + Flask)

## 📌 Overview
The Image Caption Generator is a deep learning-based web application that automatically generates descriptive captions for images. 
It combines Convolutional Neural Networks (CNN) for visual feature extraction and Recurrent Neural Networks (LSTM) for natural language generation.

## 🚀 Project Structure
Image-Caption-Generator/
│
├── app.py                         # Flask web application
├── templates/
│   ├── index.html                 # Homepage for uploading images
│   ├── result.html                # Caption display page
│
├── static/
│   ├── style.css                  # Web styling
│
├── model/
│   ├── image_caption_model.h5     # Trained CNN + LSTM model
│   ├── tokenizer.pkl              # Tokenizer for text processing
│
├── dataset/
│   ├── sample1.jpg                # Preloaded images for testing
│   ├── sample2.jpg
│
├── requirements.txt               # Dependencies list
├── README.md                      # Project documentation
└── Image_Caption_Training.ipynb   # Colab training notebook

## 🧠 Model Architecture
- Encoder: CNN (InceptionV3) extracts image features (2048-dimension vector).
- Decoder: LSTM generates word sequences based on extracted features.
- Output: Text caption like "a man riding a horse on a beach".

## ⚙️ Setup Instructions
1️⃣ Clone or extract the project  
2️⃣ Install dependencies  
```bash
pip install -r requirements.txt
```
3️⃣ Run Flask App  
```bash
python app.py
```
Visit: http://127.0.0.1:5000/

## 🧾 Training the Model 
1. Open Image_Caption_Generator
2. Upload Flickr8k dataset
3. Train CNN+LSTM model
4. Save model (`image_caption_model.h5`) and tokenizer (`tokenizer.pkl`)
5. Place both inside `/model` folder.

## 🛠️ Tools & Technologies
| Category | Tools / Frameworks |
|-----------|--------------------|
| Language | Python |
| Deep Learning | TensorFlow / Keras |
| CNN Architecture | InceptionV3 |
| Web Framework | Flask |
| Frontend | HTML, CSS |
| Dataset | Flickr8k / Flickr30k |
| Environment | Google Colab |

## 🎯 Future Enhancements
- Add MS-COCO dataset
- Use Transformer models (ViT + GPT)
- Add voice caption narration
- Multi-language caption generation

📜 Authors

Pratigya Tripathi
202210101150044

Aryan Srivastava
202210101150053
