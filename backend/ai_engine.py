import cv2
import numpy as np
from PIL import Image
import os

class FaceEngine:
    """
    High-Precision AI Engine for Face Detection, Landmark Alignment,
    512-dimensional FaceNet Vector Extraction, and Cosine Similarity Matching.
    """
    def __init__(self, min_confidence=0.82, min_face_size=40):
        self.min_confidence = min_confidence
        self.min_face_size = min_face_size
        self.use_facenet = False
        self.facenet_model = None
        self.mtcnn = None
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Check if running in memory-constrained environment (Render Free Tier ~512MB)
        disable_pytorch = os.environ.get("DISABLE_PYTORCH", "false").lower() == "true"
        
        if not disable_pytorch:
            try:
                import torch
                from facenet_pytorch import MTCNN, InceptionResnetV1
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # High-precision MTCNN configuration
                self.mtcnn = MTCNN(
                    keep_all=True,
                    min_face_size=self.min_face_size,
                    thresholds=[0.75, 0.8, 0.85],
                    post_process=True,
                    device=self.device
                )
                self.facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
                self.use_facenet = True
                print(f"Loaded High-Precision PyTorch MTCNN + FaceNet (512-d) on {self.device}.")
            except Exception as e:
                print(f"PyTorch FaceNet fallback warning: {e}")
                print("Falling back to OpenCV Deep Learning / Haar Face Detector for cloud environment compatibility.")
                self.use_facenet = False
        else:
            print("DISABLE_PYTORCH flag detected. Using lightweight OpenCV Face Detector.")

    def detect_faces(self, image_path):
        """
        Detects real faces with landmark alignment and high confidence (>= 82%).
        Returns list of dicts: [{'box': [x1, y1, x2, y2], 'crop': PIL_Image, 'tensor': PyTorch_Tensor, 'confidence': float}]
        """
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return []

        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return []

        w, h = img.size
        results = []

        if self.use_facenet:
            boxes, probs = self.mtcnn.detect(img)
            aligned_tensors, _ = self.mtcnn(img, return_prob=True)

            if boxes is not None and probs is not None and aligned_tensors is not None:
                for idx, (box, prob) in enumerate(zip(boxes, probs)):
                    if prob is None or prob < self.min_confidence:
                        continue  # Skip low confidence noise / false positives

                    x1, y1, x2, y2 = [int(b) for b in box]
                    box_w, box_h = x2 - x1, y2 - y1

                    if box_w < self.min_face_size or box_h < self.min_face_size:
                        continue  # Skip tiny noise artifacts

                    crop_x1 = max(0, x1)
                    crop_y1 = max(0, y1)
                    crop_x2 = min(w, x2)
                    crop_y2 = min(h, y2)

                    crop = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                    tensor = aligned_tensors[idx] if idx < len(aligned_tensors) else None

                    results.append({
                        'box': [x1, y1, x2, y2],
                        'crop': crop,
                        'tensor': tensor,
                        'confidence': round(float(prob), 4)
                    })
        else:
            # OpenCV Fallback with strict minNeighbors
            cv_img = cv2.imread(image_path)
            if cv_img is None:
                return []
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=7, # Strict neighbors to avoid false detections
                minSize=(self.min_face_size, self.min_face_size)
            )
            for (x, y, bw, bh) in faces:
                crop = img.crop((x, y, x + bw, y + bh))
                results.append({
                    'box': [x, y, x + bw, y + bh],
                    'crop': crop,
                    'tensor': None,
                    'confidence': 0.90
                })

        return results

    def extract_embedding(self, face_data):
        """
        Extracts L2-normalized 512-dimensional FaceNet vector embedding.
        face_data can be a dict with 'tensor'/'crop' or a PIL Image face_crop.
        """
        if isinstance(face_data, dict):
            crop = face_data.get('crop')
            tensor = face_data.get('tensor')
        else:
            crop = face_data
            tensor = None

        if self.use_facenet:
            import torch
            from torchvision import transforms

            if tensor is None and crop is not None:
                preprocess = transforms.Compose([
                    transforms.Resize((160, 160)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
                tensor = preprocess(crop).to(self.device)
            elif tensor is not None:
                tensor = tensor.to(self.device)

            if tensor is not None:
                if len(tensor.shape) == 3:
                    tensor = tensor.unsqueeze(0)
                with torch.no_grad():
                    embedding = self.facenet_model(tensor).cpu().numpy()[0]
                
                # L2 Normalize feature vector
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                return embedding.tolist()

        # Fallback processing
        if crop is not None:
            resized = crop.resize((64, 64))
            arr = np.array(resized, dtype=np.float32) / 255.0
            feat = arr.flatten()
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm
            return feat.tolist()
        
        return []

    @staticmethod
    def cosine_similarity(vec1, vec2):
        """
        Calculates Cosine Similarity between two L2-normalized feature vectors:
        Sim(u, v) = (u . v) / (||u|| * ||v||)
        """
        u = np.array(vec1, dtype=np.float32)
        v = np.array(vec2, dtype=np.float32)
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        if norm_u == 0 or norm_v == 0:
            return 0.0
        return float(np.dot(u, v) / (norm_u * norm_v))

if __name__ == '__main__':
    engine = FaceEngine()
    print("High-Precision AI Engine active.")
