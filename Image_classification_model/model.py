import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

class RegularizedDentalClassifier(nn.Module):
    """Same model architecture used in training"""
    
    def __init__(self, num_classes=6, dropout_rate=0.5):
        super(RegularizedDentalClassifier, self).__init__()
        
        self.backbone = models.resnet50(pretrained=False)
        
        num_features = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class DentalDiseasePredictor:
    """Class for predicting dental diseases from images"""
    
    def __init__(self, model_path='.dental_classifier_balanced.pth'):
        """
        Load the trained model
        
        Args:
            model_path: Path to the saved model file
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 Using device: {self.device}")
        
        # Load model and data
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.class_names = checkpoint['class_names']
        print(f"\n📋 Available classes:")
        for i, name in enumerate(self.class_names):
            print(f"   {i}: {name}")
        
        # Create model
        self.model = RegularizedDentalClassifier(num_classes=len(self.class_names))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ Model loaded successfully!")
        if 'test_accuracy' in checkpoint:
            print(f"📊 Model test accuracy: {checkpoint['test_accuracy']:.2f}%")
    
    def predict_image(self, image_path, show_image=True):
        """
        Predict the class of a single image
        
        Args:
            image_path: Path to the image
            show_image: Display image with result
            
        Returns:
            predicted_class: Name of predicted class
            confidence: Confidence percentage
            all_probabilities: Probabilities for all classes
        """
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = self.class_names[predicted.item()]
        confidence_percent = confidence.item() * 100
        
        # All class probabilities
        all_probs = probabilities.cpu().numpy()[0]
        
        # Display result
        print(f"\n{'='*60}")
        print(f"🖼️  Image: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        print(f"🎯 Diagnosis: {predicted_class}")
        print(f"📊 Confidence: {confidence_percent:.2f}%")
        print(f"{'='*60}")
        print(f"\n📈 All class probabilities:")
        
        # Sort classes by probability
        sorted_indices = np.argsort(all_probs)[::-1]
        for idx in sorted_indices:
            print(f"   {self.class_names[idx]:<25} {all_probs[idx]*100:>6.2f}%")
        
        # Display image
        if show_image:
            self._display_prediction(image, predicted_class, confidence_percent, all_probs)
        
        return predicted_class, confidence_percent, all_probs
    
    def predict_batch(self, image_folder, max_images=10):
        """
        Predict a batch of images
        
        Args:
            image_folder: Folder containing images
            max_images: Maximum number of images to process
        """
        image_files = [f for f in os.listdir(image_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"❌ No images found in folder: {image_folder}")
            return
        
        image_files = image_files[:max_images]
        results = []
        
        print(f"\n🔍 Processing {len(image_files)} images...")
        print("="*70)
        
        for img_file in image_files:
            img_path = os.path.join(image_folder, img_file)
            try:
                pred_class, confidence, probs = self.predict_image(
                    img_path, show_image=False
                )
                results.append({
                    'image': img_file,
                    'prediction': pred_class,
                    'confidence': confidence
                })
            except Exception as e:
                print(f"❌ Error processing {img_file}: {e}")
        
        # Display summary
        self._display_batch_summary(results)
        
        return results
    
    def _display_prediction(self, image, predicted_class, confidence, probabilities):
        """Display image with prediction result"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Display image
        ax1.imshow(image)
        ax1.axis('off')
        ax1.set_title(f'Diagnosis: {predicted_class}\nConfidence: {confidence:.1f}%', 
                     fontsize=14, pad=15, weight='bold')
        
        # Display probability chart
        colors = ['#2ecc71' if i == np.argmax(probabilities) else '#3498db' 
                 for i in range(len(self.class_names))]
        
        y_pos = np.arange(len(self.class_names))
        ax2.barh(y_pos, probabilities * 100, color=colors)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(self.class_names, fontsize=10)
        ax2.set_xlabel('Probability (%)', fontsize=11)
        ax2.set_title('Probability Distribution', fontsize=12, pad=10)
        ax2.grid(axis='x', alpha=0.3)
        
        # Add values on bars
        for i, v in enumerate(probabilities * 100):
            ax2.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def _display_batch_summary(self, results):
        """Display batch results summary"""
        print(f"\n{'='*70}")
        print(f"📊 Results Summary")
        print(f"{'='*70}")
        
        # Results table
        print(f"\n{'Image':<30} {'Diagnosis':<25} {'Confidence':<10}")
        print("-"*70)
        for result in results:
            print(f"{result['image']:<30} {result['prediction']:<25} {result['confidence']:>6.2f}%")
        
        # Statistics
        print(f"\n{'='*70}")
        print(f"📈 Statistics")
        print(f"{'='*70}")
        
        from collections import Counter
        predictions_count = Counter([r['prediction'] for r in results])
        
        for disease, count in predictions_count.most_common():
            percentage = (count / len(results)) * 100
            print(f"   {disease:<25} {count:>3} ({percentage:.1f}%)")
        
        avg_confidence = np.mean([r['confidence'] for r in results])
        print(f"\n   Average Confidence: {avg_confidence:.2f}%")


def main():
    """Usage example"""
    
    print("🦷 Dental Disease Diagnosis System")
    print("="*70)
    
    # Load model
    try:
        predictor = DentalDiseasePredictor('dental_classifier_balanced.pth')
    except FileNotFoundError:
        print("❌ Error: Model file 'dental_classifier_balanced.pth' not found")
        print("💡 Make sure to run the training code first")
        return
    
    print("\n" + "="*70)
    print("Choose usage mode:")
    print("="*70)
    print("1️⃣  Diagnose single image")
    print("2️⃣  Diagnose multiple images from folder")
    print("="*70)
    
    choice = input("\nYour choice (1 or 2): ").strip()
    
    if choice == '1':
        # Single image diagnosis
        image_path = input("\n📁 Enter image path: ").strip()
        
        if os.path.exists(image_path):
            predictor.predict_image(image_path, show_image=True)
        else:
            print(f"❌ Image not found: {image_path}")
    
    elif choice == '2':
        # Batch diagnosis
        folder_path = input("\n📁 Enter folder path: ").strip()
        
        if os.path.exists(folder_path):
            max_imgs = input("📊 Max number of images (press Enter for all): ").strip()
            max_imgs = int(max_imgs) if max_imgs else 100
            
            predictor.predict_batch(folder_path, max_images=max_imgs)
        else:
            print(f"❌ Folder not found: {folder_path}")
    
    else:
        print("❌ Invalid choice")
    
    print("\n✅ Program finished")


# Quick usage from another script
def quick_predict(image_path, model_path='dental_classifier_balanced.pth'):
    """
    Quick function to predict a single image
    
    Example:
        from inference import quick_predict
        result = quick_predict('my_image.jpg')
        print(f"Diagnosis: {result['prediction']}")
    """
    predictor = DentalDiseasePredictor(model_path)
    pred_class, confidence, probs = predictor.predict_image(
        image_path, show_image=False
    )
    
    return {
        'prediction': pred_class,
        'confidence': confidence,
        'probabilities': {name: float(prob*100) 
                         for name, prob in zip(predictor.class_names, probs)}
    }


if __name__ == "__main__":
    main()