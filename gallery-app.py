import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO  # Import BytesIO for binary data conversion
from cv2 import dnn
from pymongo import MongoClient
import base64

connection_string = "mongodb+srv://ravitarun2103:kF1SLaoKVkwBnQzu@cluster0.brb5i.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# MongoDB setup
def get_mongo_client():
    return MongoClient(connection_string)

# Fetch all images from MongoDB
def fetch_images_from_db(collection):
    images = []
    for document in collection.find():
        img_data = base64.b64decode(document['image'])
        img = Image.open(BytesIO(img_data))
        images.append((document['_id'], img, document['name']))
    return images

# Grayscale conversion
def convert_image_to_grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

# Noise reduction
def noise_reduction(image):
    blurred_img = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred_img

# Contrast enhancement using CLAHE
def contrast_enhancement(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(image)
    return enhanced_img

# Scratch mask creation
def create_scratch_mask(image, threshold=240):
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image

    _, mask = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
    return mask

# Scratch removal using OpenCV inpainting
def remove_scratches(image):
    mask = create_scratch_mask(image)
    inpainted_img = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return inpainted_img

# Colorize image
def colorize_image(img_array):
    proto_file = 'models/colorization_deploy_v2.prototxt'
    model_file = 'models/colorization_release_v2.caffemodel'
    hull_pts = 'models/pts_in_hull.npy'
    
    net = dnn.readNetFromCaffe(proto_file, model_file)
    kernel = np.load(hull_pts)

    scaled = img_array.astype("float32") / 255.0
    lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = kernel.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    resized = cv2.resize(lab_img, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_channel = cv2.resize(ab_channel, (img_array.shape[1], img_array.shape[0]))

    L = cv2.split(lab_img)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    colorized = (255 * colorized).astype("uint8")
    return colorized

# Function to process the image pipeline
def process_image(image):
    bw_image = convert_image_to_grayscale(image)
    noise_reduced_image = noise_reduction(bw_image)
    contrast_enhanced = contrast_enhancement(noise_reduced_image)
    enhanced_bw_image = remove_scratches(contrast_enhanced)
    enhanced_bw_image_bgr = cv2.cvtColor(enhanced_bw_image, cv2.COLOR_GRAY2BGR)
    color_image = colorize_image(enhanced_bw_image_bgr)
    return enhanced_bw_image, color_image

# Function to convert PIL image to binary
def convert_image_to_bytes(image_pil):
    buf = BytesIO()
    image_pil.save(buf, format="PNG")
    byte_data = buf.getvalue()
    return byte_data

# Streamlit app
st.title("Old Photo Enhancer")

# Connect to MongoDB and fetch images
client = get_mongo_client()
db = client['image_database']  # Use your database name
collection = db['images']      # Use your collection name
images = fetch_images_from_db(collection)

# Sidebar for image uploader and original image display
if images:
    st.sidebar.header("Select an Image")
    selected_image_id = None

    # Create two-column layout within the sidebar
    for idx, (image_id, img, name) in enumerate(images):
        if idx % 2 == 0:
            col1, col2 = st.sidebar.columns(2)  # Create two new columns in the sidebar

        # Add the image and button to the appropriate column with a fixed height for the image
        with (col1 if idx % 2 == 0 else col2):
            # Convert the PIL image to base64
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            # Display the image with uniform height
            st.markdown(
                f"""
                <div style="height: 150px; overflow: hidden; display: flex; flex-direction: column; align-items: center; justify-content: center; margin-bottom: 20px;">
                    <img src="data:image/png;base64,{img_base64}" style="max-height: 150px; margin-bottom: 10px;">
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Select", key=f"select_{idx}"):
                selected_image_id = image_id

    # Process the selected image
    if selected_image_id:
        selected_image = next(img for img_id, img, name in images if img_id == selected_image_id)
         # Convert the selected image to base64 for HTML embedding
        buffered = BytesIO()
        selected_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")


        print(type(selected_image))
        # Render the image in the center with reduced size
        st.markdown(
        f"""
        <div style="text-align: center; margin-bottom: 30px;">
            <img src="data:image/png;base64,{img_base64}" style="max-width: 300px;">
        </div>
        """,
        unsafe_allow_html=True,
    )

        # Convert to OpenCV format and process
        input_image_cv = cv2.cvtColor(np.array(selected_image), cv2.COLOR_RGB2BGR)
        bw_image, color_image = process_image(input_image_cv)

        # Display processed images
        bw_image_pil = Image.fromarray(bw_image)
        color_image_pil = Image.fromarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

        # Convert to bytes for download
        bw_image_bytes = convert_image_to_bytes(bw_image_pil)
        color_image_bytes = convert_image_to_bytes(color_image_pil)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button("Download Enhanced Black and White Image", bw_image_bytes, file_name="enhanced_bw_image.png", mime="image/png")
            st.image(bw_image_pil, caption="Enhanced Black and White Image", use_container_width=True)
        with col2:
            st.download_button("Download Enhanced Color Image", color_image_bytes, file_name="enhanced_color_image.png", mime="image/png")
            st.image(color_image_pil, caption="Enhanced Color Image", use_container_width=True)
else:
    st.write("No images found in the MongoDB collection.")