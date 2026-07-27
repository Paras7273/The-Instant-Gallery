"""
===============================================================================
Deep Learning Face Detection & Recognition Educational Script
===============================================================================
This script provides a clean, step-by-step demonstration of:
  1. Deep Learning Face Detection & Alignment using MTCNN (P-Net, R-Net, O-Net)
  2. 512-Dimensional Vector Feature Extraction using FaceNet (Inception-ResNet-v1)
  3. L2 Normalization & Cosine Similarity Face Matching

Run this script directly with:
    python backend/explain_face_detection.py
===============================================================================
"""

import os
import sys
import numpy as np
from PIL import Image

# Ensure standard UTF-8 stdout encoding for Windows console compatibility
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def explain_model_concepts():
    print("=" * 70)
    print("DEEP LEARNING FACE RETRIEVAL PIPELINE OVERVIEW")
    print("=" * 70)
    print("""
Step 1: Face Detection & Alignment (MTCNN)
  - P-Net (Proposal Network): Scans image pyramid to propose face candidates.
  - R-Net (Refinement Network): Filters out false positive background noise.
  - O-Net (Output Network): Refines bounding box & pinpoints 5 key facial landmarks:
      * Left Eye, Right Eye, Nose Tip, Left Mouth Corner, Right Mouth Corner.
  - Alignment: Rotates face crop so eyes are level, resized to 160x160 RGB.

Step 2: Face Feature Embedding (FaceNet / Inception-ResNet-v1)
  - Takes 160x160 face crop.
  - Passes through Inception-ResNet-v1 (pretrained on VGGFace2).
  - Produces a 512-dimensional continuous feature vector: v in R^512.
  - Applies L2 Normalization: v_norm = v / ||v||_2 (places face on unit sphere).

Step 3: Face Comparison (Cosine Similarity)
  - Compares 512-d unit vectors using Dot Product: Similarity(u, v) = u . v
  - Score >= 0.65 indicates matching face identity!
""")
    print("=" * 70)


def demo_pipeline():
    explain_model_concepts()

    try:
        import torch
        from facenet_pytorch import MTCNN, InceptionResnetV1
    except ImportError:
        print("[WARNING] PyTorch or facenet-pytorch not installed. Please run: pip install torch torchvision facenet-pytorch")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[INFO] PyTorch loaded successfully! Using device: {device}")

    # 1. Initialize Detector (MTCNN)
    print("[1] Initializing MTCNN Detector (P-Net -> R-Net -> O-Net)...")
    mtcnn = MTCNN(
        keep_all=True,
        min_face_size=40,
        thresholds=[0.75, 0.8, 0.85],
        post_process=True,
        device=device
    )

    # 2. Initialize Recognizer (FaceNet)
    print("[2] Initializing FaceNet Network (Inception-ResNet-v1)...")
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    print("\n[SUCCESS] Deep Learning Models Loaded and Ready!")
    print("For full line-by-line explanation, see: FACE_DETECTION_EXPLANATION.md")

if __name__ == '__main__':
    demo_pipeline()
