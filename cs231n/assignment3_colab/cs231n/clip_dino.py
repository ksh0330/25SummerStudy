import torch
import torch.nn as nn
import numpy as np
import clip
from PIL import Image
from torchvision import transforms as T
from tqdm.auto import tqdm

# cv2 is not needed - we use PIL and numpy instead


def cosine_similarity(text_features, image_features):
    """
    Computes the pairwise cosine similarity between text and image feature vectors.
    """
    # Normalize features
    text_features_norm = text_features / torch.norm(text_features, dim=1, keepdim=True)
    image_features_norm = image_features / torch.norm(image_features, dim=1, keepdim=True)
    
    # Compute cosine similarity matrix
    similarity = torch.mm(text_features_norm, image_features_norm.t())
    return similarity


def get_similarity_no_loop(text_features, image_features):
    """
    Computes the pairwise cosine similarity between text and image feature vectors.

    Args:
        text_features (torch.Tensor): A tensor of shape (N, D).
        image_features (torch.Tensor): A tensor of shape (M, D).

    Returns:
        torch.Tensor: A similarity matrix of shape (N, M), where each entry (i, j)
        is the cosine similarity between text_features[i] and image_features[j].
    """
    return cosine_similarity(text_features, image_features)


@torch.no_grad()
def clip_zero_shot_classifier(clip_model, clip_preprocess, images,
                              class_texts, device):
    """Performs zero-shot image classification using a CLIP model.

    Args:
        clip_model (torch.nn.Module): The pre-trained CLIP model for encoding
            images and text.
        clip_preprocess (Callable): A preprocessing function to apply to each
            image before encoding.
        images (List[np.ndarray]): A list of input images as NumPy arrays
            (H x W x C) uint8.
        class_texts (List[str]): A list of class label strings for zero-shot
            classification.
        device (torch.device): The device on which computation should be
            performed. Pass text_tokens to this device before passing it to
            clip_model.

    Returns:
        List[str]: Predicted class label for each image, selected from the
            given class_texts.
    """
    
    pred_classes = []

    ############################################################################
    # TODO: Find the class labels for images.                                  #
    ############################################################################
    # Process images and get features
    image_features = []
    for img in images:
        # Convert numpy array to PIL Image if needed
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img_tensor = clip_preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            img_feat = clip_model.encode_image(img_tensor)
        image_features.append(img_feat)
    image_features = torch.cat(image_features, dim=0)
    
    # Process text and get features
    text_features = []
    for text in class_texts:
        text_tensor = clip.tokenize([text]).to(device)
        with torch.no_grad():
            text_feat = clip_model.encode_text(text_tensor)
        text_features.append(text_feat)
    text_features = torch.cat(text_features, dim=0)
    
    # Compute similarities and get predictions
    similarities = cosine_similarity(text_features, image_features)
    pred_classes = [class_texts[i] for i in similarities.argmax(dim=0)]

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return pred_classes
  

class CLIPImageRetriever:
    """
    A simple image retrieval system using CLIP.
    """
    
    @torch.no_grad()
    def __init__(self, clip_model, clip_preprocess, images, device):
        """
        Args:
          clip_model (torch.nn.Module): The pre-trained CLIP model.
          clip_preprocess (Callable): Function to preprocess images.
          images (List[np.ndarray]): List of images as NumPy arrays (H x W x C).
          device (torch.device): The device for model execution.
        """
        ############################################################################
        # TODO: Store all necessary object variables to use in retrieve method.    #
        # Note that you should process all images at once here and avoid repeated  #
        # computation for each text query. You may end up NOT using the above      #
        # similarity function for most compute-optimal implementation.#
        ############################################################################
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.device = device
        
        # Process all images at once
        image_tensors = []
        for img in images:
            # Convert numpy array to PIL Image if needed
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img_tensor = clip_preprocess(img).unsqueeze(0).to(device)
            image_tensors.append(img_tensor)
        image_tensors = torch.cat(image_tensors, dim=0)
        
        # Get image features
        with torch.no_grad():
            self.image_features = clip_model.encode_image(image_tensors)
            self.image_features = self.image_features / torch.norm(self.image_features, dim=1, keepdim=True)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    
    @torch.no_grad()
    def retrieve(self, query: str, k: int = 2):
        """
        Retrieves the indices of the top-k images most similar to the input text.
        You may find torch.Tensor.topk method useful.

        Args:
            query (str): The text query.
            k (int): Return top k images.

        Returns:
            List[int]: Indices of the top-k most similar images.
        """
        top_indices = []
        ############################################################################
        # TODO: Retrieve the indices of top-k images.                              #
        ############################################################################
        # Process text query
        text_tensor = clip.tokenize([query]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tensor)
            text_features = text_features / torch.norm(text_features, dim=1, keepdim=True)
        
        # Compute similarities
        similarities = torch.mm(text_features, self.image_features.t())
        
        # Get top-k indices
        _, top_indices = similarities.topk(k, dim=1)
        top_indices = top_indices.squeeze(0).tolist()

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return top_indices

  
class DavisDataset:
    def __init__(self):
        # For local environment, we'll create dummy data instead of using tensorflow_datasets
        self.img_tsfm = T.Compose([
            T.Resize((480, 480)), T.ToTensor(),
            T.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
        ])
        # Create dummy video data for testing
        self.dummy_videos = self._create_dummy_videos()
        
    def _create_dummy_videos(self):
        """Create dummy video data for local testing"""
        dummy_videos = []
        for i in range(10):  # Create 10 dummy videos
            # Create dummy frames (99 frames per video)
            frames = np.random.randint(0, 255, (99, 480, 480, 3), dtype=np.uint8)
            # Create dummy masks (2 classes: background=0, object=1)
            masks = np.random.randint(0, 2, (99, 480, 480), dtype=np.uint8)
            dummy_videos.append({
                'frames': frames,
                'masks': masks,
                'video_name': f'dummy_video_{i}'
            })
        return dummy_videos
      
    def get_sample(self, index):
        if index >= len(self.dummy_videos):
            index = index % len(self.dummy_videos)
        video = self.dummy_videos[index]
        frames, masks = video['frames'], video['masks']
        print(f"video {video['video_name']}  {len(frames)} frames")
        return frames, masks
    
    def process_frames(self, frames, dino_model, device):
        res = []
        for f in frames:
            f = self.img_tsfm(Image.fromarray(f))[None].to(device)
            with torch.no_grad():
              tok = dino_model.get_intermediate_layers(f, n=1)[0]
            res.append(tok[0, 1:])

        res = torch.stack(res)
        return res
    
    def process_masks(self, masks, device):
        res = []
        for m in masks:
            # Use PIL instead of cv2 for resizing
            from PIL import Image
            pil_mask = Image.fromarray(m.astype(np.uint8))
            resized_mask = pil_mask.resize((60, 60), Image.NEAREST)
            m = np.array(resized_mask)
            res.append(torch.from_numpy(m).long().flatten(-2, -1))
        res = torch.stack(res).to(device)
        return res
    
    def mask_frame_overlay(self, processed_mask, frame):
        H, W = frame.shape[:2]
        mask = processed_mask.detach().cpu().numpy()
        mask = mask.reshape((60, 60))
        # Use PIL instead of cv2 for resizing
        from PIL import Image
        pil_mask = Image.fromarray(mask.astype(np.uint8))
        resized_mask = pil_mask.resize((W, H), Image.NEAREST)
        mask = np.array(resized_mask)
        overlay = create_segmentation_overlay(mask, frame.copy())
        return overlay
        


def create_segmentation_overlay(segmentation_mask, image, alpha=0.5):
    """
    Generate a colored segmentation overlay on top of an RGB image.

    Parameters:
        segmentation_mask (np.ndarray): 2D array of shape (H, W), with class indices.
        image (np.ndarray): 3D array of shape (H, W, 3), RGB image.
        alpha (float): Transparency factor for overlay (0 = only image, 1 = only mask).

    Returns:
        np.ndarray: Image with segmentation overlay (shape: (H, W, 3), dtype: uint8).
    """
    assert segmentation_mask.shape[:2] == image.shape[:2], "Segmentation and image size mismatch"
    assert image.dtype == np.uint8, "Image must be of type uint8"

    # Generate deterministic colors for each class using a fixed colormap
    def generate_colormap(n):
        np.random.seed(42)  # For determinism
        colormap = np.random.randint(0, 256, size=(n, 3), dtype=np.uint8)
        return colormap

    colormap = generate_colormap(10)

    # Create a color image for the segmentation mask
    seg_color = colormap[segmentation_mask]  # shape: (H, W, 3)

    # Blend with original image using numpy instead of cv2
    overlay = (image * (1 - alpha) + seg_color * alpha).astype(np.uint8)

    return overlay


def compute_iou(pred, gt, num_classes):
    """Compute the mean Intersection over Union (IoU)."""
    iou = 0
    for ci in range(num_classes):
        p = pred == ci
        g = gt == ci
        iou += (p & g).sum() / ((p | g).sum() + 1e-8)
    return iou / num_classes


class DINOSegmentation:
    def __init__(self, device, num_classes: int, inp_dim : int = 384):
        """
        Initialize the DINOSegmentation model.

        This defines a simple neural network designed to  classify DINO feature
        vectors into segmentation classes. It includes model initialization,
        optimizer, and loss function setup.

        Args:
            device (torch.device): Device to run the model on (CPU or CUDA).
            num_classes (int): Number of segmentation classes.
            inp_dim (int, optional): Dimensionality of the input DINO features.
        """

        ############################################################################
        # TODO: Define a very lightweight pytorch model, optimizer, and loss       #
        # function to train classify each DINO feature vector into a seg. class.   #
        # It can be a linear layer or two layer neural network.                    #
        ############################################################################
        self.device = device
        self.num_classes = num_classes
        
        # Simple two-layer neural network
        self.model = torch.nn.Sequential(
            torch.nn.Linear(inp_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes)
        ).to(device)
        
        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = torch.nn.CrossEntropyLoss()

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def train(self, X_train, Y_train, num_iters=500):
        """Train the segmentation model using the provided training data.

        Args:
            X_train (torch.Tensor): Input feature vectors of shape (N, D).
            Y_train (torch.Tensor): Ground truth labels of shape (N,).
            num_iters (int, optional): Number of optimization steps.
        """
        ############################################################################
        # TODO: Train your model for `num_iters` steps.                            #
        ############################################################################
        self.model.train()
        for i in range(num_iters):
            # Forward pass
            outputs = self.model(X_train)
            loss = self.criterion(outputs, Y_train)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if i % 100 == 0:
                print(f'Iteration {i}, Loss: {loss.item():.4f}')

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    
    @torch.no_grad()
    def inference(self, X_test):
        """Perform inference on the given test DINO feature vectors.

        Args:
            X_test (torch.Tensor): Input feature vectors of shape (N, D).

        Returns:
            torch.Tensor of shape (N,): Predicted class indices.
        """
        pred_classes = None
        ############################################################################
        # TODO: Train your model for `num_iters` steps.                            #
        ############################################################################
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)
            pred_classes = torch.argmax(outputs, dim=1)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return pred_classes