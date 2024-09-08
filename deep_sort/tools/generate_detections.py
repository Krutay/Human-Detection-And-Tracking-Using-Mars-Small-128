import os
import errno
import argparse
import numpy as np
import cv2
import tensorflow as tf

def _run_in_batches(func, data_x, out, batch_size):
    num_batches = (len(data_x) + batch_size - 1) // batch_size
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(data_x))
        batch_data = data_x[start:end]
        out[start:end] = func(batch_data)

def extract_image_patch(image, bbox, patch_shape):
    """Extract image patch from bounding box and ensure it has the correct number of channels."""
    bbox = np.array(bbox)
    if patch_shape is not None:
        # Ensure patch_shape is a tuple of integers
        if not (isinstance(patch_shape, tuple) and len(patch_shape) == 2 and all(isinstance(x, int) for x in patch_shape)):
            raise ValueError(f"Invalid patch_shape: {patch_shape}. Expected a tuple of two integers.")

        # Calculate target aspect ratio and adjust bounding box dimensions
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # Clip bounding box coordinates to be within image bounds
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]

    # Verify the patch_shape values before resizing
    if patch_shape[0] <= 0 or patch_shape[1] <= 0:
        raise ValueError(f"Invalid patch_shape values: {patch_shape}. Both dimensions must be positive integers.")
    
    # Convert patch_shape to integer tuples to avoid any issues
    patch_shape = tuple(map(int, patch_shape))
    image = cv2.resize(image, tuple(patch_shape[::-1]))

    if len(image.shape) == 3 and image.shape[-1] == 3:
        # Convert RGB to grayscale if needed
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image[..., np.newaxis]  # Add channel dimension

    return image







class ImageEncoder(object):
    def __init__(self, checkpoint_filename, input_name=None):
        # Load the Keras model
        self.model = tf.keras.models.load_model(checkpoint_filename)

        # Identify the input and output tensors
        self.input_var = self.model.input
        self.output_var = self.model.output

        # Ensure the model has correct input and output shapes
        self.feature_dim = self.output_var.shape[-1]
        self.image_shape = self.input_var.shape[1:]
        print(self.input_var.shape[1:])

    def __call__(self, data_x, batch_size=32):
        # Use the Keras model to predict features
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        num_batches = (len(data_x) + batch_size - 1) // batch_size

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(data_x))
            batch_data = data_x[start:end]
            batch_features = self.model.predict(batch_data, batch_size=batch_size)
            out[start:end] = batch_features

        return out


def create_box_encoder(model_filename, input_name="images", batch_size=32):
    image_encoder = ImageEncoder(model_filename, input_name)
    image_shape = image_encoder.image_shape

    def encoder(image, boxes):
        image_patches = []
        print(f"Image shape: {image.shape}")
        print(f"Patch shape: {image_shape[:2]}")
        for box in boxes:
            patch_shape = image_shape[:2]
            print(f"Using patch_shape: {patch_shape}")  # Debugging statement
            patch = extract_image_patch(image, box, tuple(map(int, patch_shape)))  # Ensure patch_shape is a tuple of integers
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        return image_encoder(image_patches, batch_size)

    return encoder





def generate_detections(encoder, mot_dir, output_dir, detection_dir=None):
    """Generate detections with features."""
    if detection_dir is None:
        detection_dir = mot_dir
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise ValueError(
                "Failed to create output directory '%s'" % output_dir)

    for sequence in os.listdir(mot_dir):
        print("Processing %s" % sequence)
        sequence_dir = os.path.join(mot_dir, sequence)

        image_dir = os.path.join(sequence_dir, "img1")
        image_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in os.listdir(image_dir)}

        detection_file = os.path.join(
            detection_dir, sequence, "det/det.txt")
        detections_in = np.loadtxt(detection_file, delimiter=',')
        detections_out = []

        frame_indices = detections_in[:, 0].astype(np.int)
        min_frame_idx = frame_indices.astype(np.int).min()
        max_frame_idx = frame_indices.astype(np.int).max()
        for frame_idx in range(min_frame_idx, max_frame_idx + 1):
            print("Frame %05d/%05d" % (frame_idx, max_frame_idx))
            mask = frame_indices == frame_idx
            rows = detections_in[mask]

            if frame_idx not in image_filenames:
                print("WARNING: could not find image for frame %d" % frame_idx)
                continue
            bgr_image = cv2.imread(
                image_filenames[frame_idx], cv2.IMREAD_COLOR)
            features = encoder(bgr_image, rows[:, 2:6].copy())
            detections_out += [np.r_[(row, feature)] for row, feature
                               in zip(rows, features)]

        output_filename = os.path.join(output_dir, "%s.npy" % sequence)
        np.save(
            output_filename, np.asarray(detections_out), allow_pickle=False)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Re-ID feature extractor")
    parser.add_argument(
        "--model",
        default="resources/networks/mars-small128.h5",
        help="Path to the H5 model file.")
    parser.add_argument(
        "--mot_dir", help="Path to MOTChallenge directory (train or test)",
        required=True)
    parser.add_argument(
        "--detection_dir", help="Path to custom detections. Defaults to "
        "standard MOT detections Directory structure should be the default "
        "MOTChallenge structure: [sequence]/det/det.txt", default=None)
    parser.add_argument(
        "--output_dir", help="Output directory. Will be created if it does not"
        " exist.", default="detections")
    return parser.parse_args()

def main():
    args = parse_args()
    encoder = create_box_encoder(args.model, batch_size=32)
    generate_detections(encoder, args.mot_dir, args.output_dir,
                        args.detection_dir)

if __name__ == "__main__":
    main()
