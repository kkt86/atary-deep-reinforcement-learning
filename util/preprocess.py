from skimage.color import rgb2gray
from skimage.transform import resize


def preprocess_frame(image, target_shape=(110, 84)):
    grayscaled = rgb2gray(image)
    resized = resize(grayscaled, output_shape=target_shape)
    return resized


def stack_frames(stacked_frames, new_frame, is_empty):
    if is_empty:
        for i in range(4):
            stacked_frames.append(new_frame)
    else:
        stacked_frames.append(new_frame)
    return stacked_frames
