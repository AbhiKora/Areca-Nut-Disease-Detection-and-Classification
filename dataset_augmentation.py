import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img

# Input and output directories
input_dir = "dataset/mild"     # put your original images here
output_dir = "dataset/mild_augmented"  # where augmented images will be saved

os.makedirs(output_dir, exist_ok=True)

# Define augmentation
datagen = ImageDataGenerator(
    shear_range=20,          # shear angle in degrees
    horizontal_flip=True,
    vertical_flip=True
)

# Loop over all images
for img_name in os.listdir(input_dir):
    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(input_dir, img_name)

        # Load image
        img = load_img(img_path)  # loads as PIL image
        x = img_to_array(img)     # to numpy array
        x = x.reshape((1,) + x.shape)

        # Generate 3 augmented versions per image
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=output_dir,
                                  save_prefix=img_name.split(".")[0],
                                  save_format="jpg"):
            i += 1
            if i >= 3:   # adjust if you want more per image
                break
