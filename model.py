import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import re

class ImageCaptionGenerator:
    def __init__(self):
        self.model = InceptionV3(weights='imagenet')
        
        self.scene_contexts = {
            'outdoor': ['mountain', 'lake', 'beach', 'ocean', 'sea', 'valley', 'cliff', 'park', 'forest', 'tree', 'sky', 'cloud', 'sunset', 'sunrise'],
            'urban': ['building', 'street', 'city', 'skyscraper', 'bridge', 'tower', 'monument'],
            'indoor': ['room', 'furniture', 'lamp', 'window', 'door', 'wall', 'floor'],
            'nature_animal': ['animal', 'bird', 'dog', 'cat', 'horse', 'leopard', 'tiger', 'elephant', 'bear', 'wolf', 'deer', 'rabbit', 'fox'],
            'water': ['boat', 'ship', 'yacht', 'surfboard', 'swimming', 'diving', 'ocean', 'lake', 'river'],
            'transport': ['car', 'bus', 'train', 'airplane', 'motorcycle', 'bicycle', 'vehicle'],
            'food': ['pizza', 'burger', 'bread', 'fruit', 'vegetable', 'dish', 'meal', 'food'],
            'people': ['person', 'people', 'man', 'woman', 'child', 'boy', 'girl', 'face']
        }
        
        self.descriptive_templates = {
            'nature_animal': [
                "{article} {main} in {article2} natural setting",
                "{article} beautiful {main} captured in this photo",
                "{article} {main} {context}",
                "{article} {main} in the wild",
            ],
            'outdoor_scene': [
                "{article} scenic view of {main}",
                "{article} beautiful {main} landscape",
                "{main} with {secondary} in the background",
                "{article} picturesque {main} scene",
            ],
            'urban': [
                "{article} {main} in an urban environment",
                "{main} architecture captured in this photo",
                "{article} modern {main}",
            ],
            'objects': [
                "{article} {main} with {secondary}",
                "{article} photo featuring {main}",
                "{article} {main} {context}",
            ],
            'single': [
                "{article} {main}",
                "{article} {main} in the photo",
                "this image shows {article} {main}",
            ]
        }
        
    def _clean_label(self, label):
        """Clean and simplify ImageNet labels for natural captions"""
        label = label.lower()
        label = re.sub(r'[_-]', ' ', label)
        label = re.sub(r'\b(web|site)\b', '', label)
        label = label.strip()
        return label
    
    def _get_article(self, word):
        """Get appropriate article (a/an) for a word"""
        if not word:
            return "a"
        vowels = ['a', 'e', 'i', 'o', 'u']
        return "an" if word[0].lower() in vowels else "a"
    
    def _detect_scene_type(self, labels):
        """Detect the type of scene from labels"""
        for scene_type, keywords in self.scene_contexts.items():
            for label in labels:
                for keyword in keywords:
                    if keyword in label:
                        return scene_type
        return 'general'
    
    def _create_context_phrase(self, labels, scores):
        """Create contextual phrases based on secondary objects"""
        contexts = []
        
        for i, (label, score) in enumerate(zip(labels[1:], scores[1:]), 1):
            if score > 0.05:
                if any(word in label for word in ['outdoor', 'mountain', 'sky', 'cloud']):
                    contexts.append('outdoors')
                elif any(word in label for word in ['park', 'grass', 'field']):
                    contexts.append('in a park')
                elif any(word in label for word in ['beach', 'ocean', 'sea']):
                    contexts.append('at the beach')
                elif any(word in label for word in ['street', 'road']):
                    contexts.append('on the street')
                    
        return contexts[0] if contexts else ''
    
    def _create_enhanced_caption(self, predictions):
        """Create enhanced natural language captions from predictions"""
        labels = [self._clean_label(pred[1]) for pred in predictions]
        scores = [pred[2] for pred in predictions]
        
        if len(labels) == 0:
            return "an interesting scene captured in this image"
        
        main_label = labels[0]
        main_score = scores[0]
        article = self._get_article(main_label)
        
        scene_type = self._detect_scene_type(labels)
        context = self._create_context_phrase(labels, scores)
        
        if main_score > 0.7:
            if scene_type == 'nature_animal':
                templates = self.descriptive_templates['nature_animal']
                return np.random.choice(templates).format(
                    article=article,
                    main=main_label,
                    article2=self._get_article(labels[1]) if len(labels) > 1 else "a",
                    context=context if context else "in nature"
                )
            elif scene_type == 'outdoor':
                if len(labels) >= 2 and scores[1] > 0.2:
                    secondary = labels[1]
                    templates = self.descriptive_templates['outdoor_scene']
                    return np.random.choice(templates).format(
                        article=article,
                        main=main_label,
                        secondary=secondary
                    )
        
        if len(labels) >= 2 and scores[0] > 0.4 and scores[1] > 0.15:
            secondary = labels[1]
            
            if scene_type in ['nature_animal', 'outdoor']:
                template = f"{article} {main_label} with {self._get_article(secondary)} {secondary} in the scene"
            elif scene_type == 'urban':
                template = f"{article} {main_label} with {self._get_article(secondary)} {secondary} nearby"
            else:
                template = f"{article} photo of {article} {main_label} and {self._get_article(secondary)} {secondary}"
            
            return template
        
        if main_score > 0.3:
            if context:
                return f"{article} {main_label} {context}"
            else:
                templates = self.descriptive_templates['single']
                return np.random.choice(templates).format(
                    article=article,
                    main=main_label
                )
        
        if len(labels) >= 3:
            return f"an image featuring {main_label}, {labels[1]}, and {labels[2]}"
        
        return f"{article} {main_label}"
    
    def extract_features_and_predict(self, image_path):
        """Extract features and get predictions from an image using InceptionV3"""
        try:
            image = load_img(image_path, target_size=(299, 299))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            
            predictions = self.model.predict(image, verbose=0)
            decoded = decode_predictions(predictions, top=8)[0]
            
            return decoded
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    
    def generate_caption(self, image_path):
        """Generate enhanced caption for an image using InceptionV3 predictions"""
        predictions = self.extract_features_and_predict(image_path)
        
        if predictions is None or len(predictions) == 0:
            return "Error processing image"
        
        caption = self._create_enhanced_caption(predictions)
        
        return caption.capitalize() if caption else "An interesting image"

if __name__ == "__main__":
    print("Initializing Image Caption Generator...")
    generator = ImageCaptionGenerator()
    print("Model initialized successfully!")
    print("Using InceptionV3 pre-trained on ImageNet for enhanced image classification and caption generation")
