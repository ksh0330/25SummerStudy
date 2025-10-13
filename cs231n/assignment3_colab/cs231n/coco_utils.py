import os, json
import numpy as np
import h5py

dir_path = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = os.path.join(dir_path, "datasets/coco_captioning")
BASE_DIR = os.path.abspath(BASE_DIR)  # Convert to absolute path

def load_coco_data(base_dir=None, max_train=None, pca_features=True):
    if base_dir is None:
        # Use absolute path
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets/coco_captioning"))
    print('base dir ', base_dir)
    data = {}
    
    # Check if we have the sample dataset
    caption_file = os.path.join(base_dir, "coco2014_captions.h5")
    print(f"Looking for caption file: {caption_file}")
    print(f"File exists: {os.path.exists(caption_file)}")
    if os.path.exists(caption_file):
        with h5py.File(caption_file, "r") as f:
            for k, v in f.items():
                data[k] = np.asarray(v)
        
        # Load word mappings
        dict_file = os.path.join(base_dir, "coco2014_vocab.json")
        if os.path.exists(dict_file):
            with open(dict_file, "r") as f:
                dict_data = json.load(f)
                for k, v in dict_data.items():
                    data[k] = v
        
        # Use local image files instead of URLs
        n_train = data["train_captions"].shape[0]
        n_val = data["val_captions"].shape[0]
        data["train_urls"] = np.array([f"cs231n/datasets/coco_captioning/images/train_{i:06d}.jpg" for i in range(n_train)])
        data["val_urls"] = np.array([f"cs231n/datasets/coco_captioning/images/val_{i:06d}.jpg" for i in range(n_val)])
        
        # Create image indices
        data["train_image_idxs"] = np.arange(n_train)
        data["val_image_idxs"] = np.arange(n_val)
        
        # Maybe subsample the training data
        print(f"max_train parameter: {max_train}")
        print(f"Original train_captions shape: {data['train_captions'].shape}")
        if max_train is not None and max_train > 0:
            num_train = data["train_captions"].shape[0]
            print(f"max_train={max_train}, num_train={num_train}")
            if max_train < num_train:  # Only subsample if max_train is smaller than actual data
                print(f"Subsampling from {num_train} to {max_train}")
                mask = np.random.randint(num_train, size=max_train)
                data["train_captions"] = data["train_captions"][mask]
                data["train_image_idxs"] = data["train_image_idxs"][mask]
                data["train_features"] = data["train_features"][mask]
                data["train_urls"] = data["train_urls"][mask]
                
                # Reset image_idxs to be sequential after subsampling
                data["train_image_idxs"] = np.arange(max_train)
            else:
                print(f"No subsampling needed: max_train >= num_train")
        else:
            print(f"No subsampling: max_train is None or <= 0")
        
        return data
    else:
        # Fallback to original implementation if files don't exist
        print("Sample dataset not found, using fallback...")
        # Create minimal data for testing
        np.random.seed(231)
        n_train = max_train if max_train else 50
        n_val = 20
        
        # Create vocab with proper range and diverse words
        vocab_size = 1000
        # Create more diverse vocabulary for better CLIP testing
        diverse_words = [
            'person', 'dog', 'cat', 'car', 'bike', 'tree', 'house', 'sky', 'water', 'food',
            'book', 'phone', 'computer', 'chair', 'table', 'bed', 'window', 'door', 'road', 'grass',
            'flower', 'bird', 'fish', 'horse', 'cow', 'sheep', 'elephant', 'lion', 'tiger', 'bear',
            'red', 'blue', 'green', 'yellow', 'black', 'white', 'big', 'small', 'tall', 'short',
            'running', 'sitting', 'standing', 'walking', 'jumping', 'flying', 'swimming', 'eating', 'sleeping', 'playing',
            'beautiful', 'cute', 'funny', 'happy', 'sad', 'angry', 'tired', 'excited', 'calm', 'busy'
        ]
        
        word_to_idx = {
            '<NULL>': 0,
            '<START>': 1,
            '<END>': 2,
            '<UNK>': 3,
        }
        idx_to_word = {
            0: '<NULL>',
            1: '<START>',
            2: '<END>',
            3: '<UNK>',
        }
        
        # Add diverse words to vocabulary
        for i, word in enumerate(diverse_words):
            word_to_idx[word] = i + 4
            idx_to_word[i + 4] = word
        
        # Fill remaining slots with generic words
        for i in range(len(diverse_words) + 4, vocab_size):
            word_to_idx[f'word_{i}'] = i
            idx_to_word[i] = f'word_{i}'
        
        # Create more diverse captions using the diverse words
        def create_diverse_caption(length=10):
            # Use diverse words more frequently
            diverse_indices = list(range(4, 4 + len(diverse_words)))
            caption = [1]  # <START>
            for _ in range(length - 2):
                if np.random.random() < 0.7:  # 70% chance to use diverse words
                    caption.append(np.random.choice(diverse_indices))
                else:
                    caption.append(np.random.randint(4, vocab_size))
            caption.append(2)  # <END>
            return caption
        
        # Generate diverse captions
        train_captions = np.array([create_diverse_caption() for _ in range(n_train)])
        val_captions = np.array([create_diverse_caption() for _ in range(n_val)])
        
        data = {
            "train_captions": train_captions,
            "val_captions": val_captions,
            "train_features": np.random.randn(n_train, 512),
            "val_features": np.random.randn(n_val, 512),
            "word_to_idx": word_to_idx,
            "idx_to_word": idx_to_word,
            "train_urls": np.array([f"http://example.com/train_{i}.jpg" for i in range(n_train)]),
            "val_urls": np.array([f"http://example.com/val_{i}.jpg" for i in range(n_val)]),
            "train_image_idxs": np.arange(n_train),
            "val_image_idxs": np.arange(n_val)
        }
        return data


def decode_captions(captions, idx_to_word):
    singleton = False
    if captions.ndim == 1:
        singleton = True
        captions = captions[None]
    decoded = []
    N, T = captions.shape
    for i in range(N):
        words = []
        for t in range(T):
            word_idx = int(captions[i, t])  # Convert to int
            # Handle both string and int keys
            if word_idx in idx_to_word:
                word = idx_to_word[word_idx]
            elif str(word_idx) in idx_to_word:
                word = idx_to_word[str(word_idx)]
            else:
                word = idx_to_word.get('<UNK>', 'UNK')  # Fallback to UNK
            if word != "<NULL>":
                words.append(word)
            if word == "<END>":
                break
        decoded.append(" ".join(words))
    if singleton:
        decoded = decoded[0]
    return decoded


def sample_coco_minibatch(data, batch_size=100, split="train"):
    split_size = data["%s_captions" % split].shape[0]
    mask = np.random.choice(split_size, batch_size)
    captions = data["%s_captions" % split][mask]
    image_idxs = data["%s_image_idxs" % split][mask]
    image_features = data["%s_features" % split][image_idxs]
    urls = data["%s_urls" % split][image_idxs]
    return captions, image_features, urls
