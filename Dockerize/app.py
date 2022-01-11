import os
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from utils import load_graph, run_inference_for_single_image


PATH_TO_FROZEN_GRAPH = os.path.join(os.getcwd(),
                                 "models",
                                 "research",
                                 "cup_maskrcnn_model_export",
                                 "frozen_inference_graph.pb")

PATH_TO_LABELS = os.path.join(os.getcwd(),
                              "models", 
                              "research", 
                              "object_detection",
                              "data",
                              "cup_label_map.pbtxt")


if __name__ == "__main__":

    # Load tensorflow graph
    detection_graph = load_graph(PATH_TO_FROZEN_GRAPH)

    # Load label map
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    st.title("Cup Mask Segmentation")
    st.subheader("Home")
    IMAGE_EXTENSIONS = ["jpeg", "jpg", "png"]
    image_file = st.file_uploader("Upload the image", type=IMAGE_EXTENSIONS)

    if image_file is not None:
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        stframe = st.empty()

        image_expanded = np.expand_dims(image, axis=0)
        output_dict = run_inference_for_single_image(image, detection_graph)
    
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)
        stframe.image(image)
