import os
import shutil
import json
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN

try:
    from ai_engine import FaceEngine
except ImportError:
    from backend.ai_engine import FaceEngine

class GalleryIndexer:
    """
    Scans an event photo directory (including user-created sub-folders/albums),
    detects faces with landmark alignment, extracts 512-d FaceNet embeddings,
    clusters them into Unique Person Profiles, and performs high-precision cosine similarity search.
    Supports folder creation, photo/folder deletion, and automatic database re-indexing.
    """
    def __init__(self, gallery_dir="data/event_gallery", db_file="data/indexed_db/embeddings.json", thumbs_dir="data/person_thumbnails"):
        self.gallery_dir = gallery_dir
        self.db_file = db_file
        self.thumbs_dir = thumbs_dir
        self.engine = FaceEngine()
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.db_file), exist_ok=True)
        os.makedirs(self.gallery_dir, exist_ok=True)
        os.makedirs(self.thumbs_dir, exist_ok=True)

    def get_folders(self):
        """Returns list of folder metadata in event_gallery."""
        folders = []
        # Include Root / General folder
        root_files = [f for f in os.listdir(self.gallery_dir) 
                      if os.path.isfile(os.path.join(self.gallery_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
        folders.append({
            'name': 'Default Gallery',
            'folder_id': 'default',
            'photo_count': len(root_files)
        })

        for item in sorted(os.listdir(self.gallery_dir)):
            item_path = os.path.join(self.gallery_dir, item)
            if os.path.isdir(item_path):
                f_files = [f for f in os.listdir(item_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
                folders.append({
                    'name': item,
                    'folder_id': item,
                    'photo_count': len(f_files)
                })

        return folders

    def create_folder(self, folder_name):
        """Creates a new user folder inside event_gallery."""
        # Sanitize folder name
        safe_name = "".join([c for c in folder_name if c.isalnum() or c in (' ', '_', '-')]).strip()
        if not safe_name:
            safe_name = "New_Album"
        
        folder_path = os.path.join(self.gallery_dir, safe_name)
        os.makedirs(folder_path, exist_ok=True)
        return safe_name

    def delete_folder(self, folder_name):
        """Deletes a folder and all contained photos, then re-indexes."""
        if folder_name == 'default' or folder_name == 'Default Gallery':
            # Clear root photos
            for f in os.listdir(self.gallery_dir):
                file_path = os.path.join(self.gallery_dir, f)
                if os.path.isfile(file_path) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    os.remove(file_path)
        else:
            folder_path = os.path.join(self.gallery_dir, folder_name)
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                shutil.rmtree(folder_path)

        # Trigger database re-index
        return self.index_gallery()

    def delete_photo(self, rel_path):
        """Deletes an individual photo file and re-indexes."""
        # Normalize slashes
        rel_path = rel_path.replace('\\', '/').lstrip('/')
        full_path = os.path.join(self.gallery_dir, rel_path)
        
        if os.path.exists(full_path) and os.path.isfile(full_path):
            os.remove(full_path)
            print(f"Deleted photo: {full_path}")

        # Trigger database re-index
        return self.index_gallery()

    def clear_all(self):
        """Deletes all photos and user folders in event_gallery and clears database."""
        if os.path.exists(self.gallery_dir):
            shutil.rmtree(self.gallery_dir)
        os.makedirs(self.gallery_dir, exist_ok=True)

        if os.path.exists(self.thumbs_dir):
            shutil.rmtree(self.thumbs_dir)
        os.makedirs(self.thumbs_dir, exist_ok=True)

        return self.index_gallery()

    def index_gallery(self):
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        if not os.path.exists(self.gallery_dir):
            print(f"Gallery directory {self.gallery_dir} does not exist.")
            return

        # Find all images recursively inside gallery_dir
        image_entries = []
        for root, dirs, files in os.walk(self.gallery_dir):
            for file in sorted(files):
                if file.lower().endswith(valid_exts):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, self.gallery_dir).replace('\\', '/')
                    folder_name = os.path.dirname(rel_path)
                    if not folder_name:
                        folder_name = "Default Gallery"

                    image_entries.append({
                        'rel_path': rel_path,
                        'full_path': full_path,
                        'image_name': file,
                        'folder_name': folder_name,
                        'url': f"/gallery-photos/{rel_path}"
                    })

        print(f"Found {len(image_entries)} total photo(s) across folders.")

        records = []
        flat_face_list = []
        total_faces = 0

        for entry in image_entries:
            img_path = entry['full_path']
            rel_path = entry['rel_path']
            img_name = entry['image_name']
            folder_name = entry['folder_name']
            url = entry['url']

            try:
                faces = self.engine.detect_faces(img_path)
                
                face_data = []
                for idx, face in enumerate(faces):
                    emb = self.engine.extract_embedding(face)
                    if not emb:
                        continue

                    face_record = {
                        'photo_index': len(records),
                        'image_name': img_name,
                        'rel_path': rel_path,
                        'image_path': img_path,
                        'folder_name': folder_name,
                        'url': url,
                        'face_index_in_photo': idx,
                        'box': face['box'],
                        'confidence': face['confidence'],
                        'crop': face['crop'],
                        'embedding': emb
                    }

                    face_data.append(face_record)
                    flat_face_list.append(face_record)
                    total_faces += 1

                records.append({
                    'image_name': img_name,
                    'rel_path': rel_path,
                    'image_path': img_path,
                    'folder_name': folder_name,
                    'url': url,
                    'faces': face_data
                })

            except Exception as e:
                print(f"Error processing {img_name}: {e}")

        # Clean thumbnails directory
        if os.path.exists(self.thumbs_dir):
            for f in os.listdir(self.thumbs_dir):
                try: os.remove(os.path.join(self.thumbs_dir, f))
                except Exception: pass

        # Perform DBSCAN Face Clustering to identify Unique Person Identities
        unique_persons = []
        if len(flat_face_list) > 0:
            embeddings_matrix = np.array([f['embedding'] for f in flat_face_list], dtype=np.float32)
            
            dbscan = DBSCAN(eps=0.38, min_samples=1, metric='cosine').fit(embeddings_matrix)
            cluster_labels = dbscan.labels_

            clusters = {}
            for label, face_info in zip(cluster_labels, flat_face_list):
                cluster_id = int(label)
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(face_info)

            sorted_cluster_ids = sorted(clusters.keys(), key=lambda cid: len(clusters[cid]), reverse=True)

            person_counter = 1
            for cid in sorted_cluster_ids:
                c_faces = clusters[cid]
                person_id = f"person_{person_counter}"
                person_name = f"Person #{person_counter}"
                person_counter += 1

                c_embs = np.array([f['embedding'] for f in c_faces], dtype=np.float32)
                centroid = np.mean(c_embs, axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm

                best_face = max(c_faces, key=lambda f: f['confidence'])
                thumb_filename = f"{person_id}.jpg"
                thumb_filepath = os.path.join(self.thumbs_dir, thumb_filename)
                
                try:
                    best_face['crop'].resize((180, 180)).save(thumb_filepath, 'JPEG', quality=95)
                except Exception as e:
                    print(f"Error saving thumbnail for {person_id}: {e}")

                photo_names = list(set([f['image_name'] for f in c_faces]))

                for f in c_faces:
                    f['person_id'] = person_id

                unique_persons.append({
                    'person_id': person_id,
                    'name': person_name,
                    'face_count': len(c_faces),
                    'photo_count': len(photo_names),
                    'photos': photo_names,
                    'thumbnail_url': f"/person-thumbnails/{thumb_filename}",
                    'centroid_embedding': centroid.tolist()
                })

        # Sanitize records for JSON output
        sanitized_records = []
        for r in records:
            clean_faces = []
            for f in r['faces']:
                clean_faces.append({
                    'box': f['box'],
                    'confidence': f['confidence'],
                    'person_id': f.get('person_id', 'unknown'),
                    'embedding': f['embedding']
                })
            sanitized_records.append({
                'image_name': r['image_name'],
                'rel_path': r['rel_path'],
                'image_path': r['image_path'].replace('\\', '/'),
                'folder_name': r['folder_name'],
                'url': r['url'],
                'faces': clean_faces
            })

        # Get folder metadata
        folders_list = self.get_folders()

        db_payload = {
            'total_photos': len(sanitized_records),
            'total_faces': total_faces,
            'total_unique_persons': len(unique_persons),
            'folders': folders_list,
            'persons': unique_persons,
            'records': sanitized_records
        }

        with open(self.db_file, 'w', encoding='utf-8') as f:
            json.dump(db_payload, f, indent=2)

        print(f"Successfully indexed {len(sanitized_records)} photos across {len(folders_list)} folders.")
        print(f"Detected {total_faces} face vectors & clustered into {len(unique_persons)} Unique Persons!")
        return db_payload

    def search_by_selfie(self, selfie_path, similarity_threshold=0.55):
        """
        Receives an attendee selfie, detects target face with landmark alignment,
        matches query embedding against Person Clusters and Indexed Photos.
        """
        if not os.path.exists(self.db_file):
            print("No index database found. Please run index_gallery first.")
            return []

        with open(self.db_file, 'r', encoding='utf-8') as f:
            db_data = json.load(f)

        query_faces = self.engine.detect_faces(selfie_path)
        if not query_faces:
            print("No face detected in selfie.")
            return []

        query_face = max(query_faces, key=lambda f: (f['box'][2]-f['box'][0]) * (f['box'][3]-f['box'][1]))
        query_emb = self.engine.extract_embedding(query_face)

        # Build mapping of person_id to person name
        persons_map = {p['person_id']: p['name'] for p in db_data.get('persons', [])}

        matches = []
        for record in db_data.get('records', []):
            max_sim = 0.0
            best_box = None
            best_person_id = None

            for face in record.get('faces', []):
                sim = FaceEngine.cosine_similarity(query_emb, face['embedding'])
                if sim > max_sim:
                    max_sim = sim
                    best_box = face['box']
                    best_person_id = face.get('person_id')

            if max_sim >= similarity_threshold:
                matched_name = persons_map.get(best_person_id, "Identified Person") if best_person_id else "Identified Person"
                matches.append({
                    'image_name': record['image_name'],
                    'rel_path': record['rel_path'],
                    'image_path': record['image_path'].replace('\\', '/'),
                    'folder_name': record.get('folder_name', 'Default Gallery'),
                    'url': record.get('url', f"/gallery-photos/{record['rel_path']}"),
                    'confidence': round(float(max_sim), 4),
                    'match_percentage': round(float(max_sim * 100), 1),
                    'bounding_box': best_box,
                    'person_id': best_person_id,
                    'matched_person_name': matched_name
                })

        matches.sort(key=lambda x: x['confidence'], reverse=True)
        return matches

if __name__ == '__main__':
    indexer = GalleryIndexer()
    indexer.index_gallery()
