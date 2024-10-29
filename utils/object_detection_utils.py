import numpy as np

def class_wise_nms(bboxes, scores, labels, iou_threshold=0.5):
    """
    Applies Non-Maximum Suppression (NMS) independently for each class label.

    Parameters:
        bboxes (numpy.ndarray): Array of bounding boxes (x_min, y_min, x_max, y_max).
        scores (numpy.ndarray): Array of confidence scores for each bounding box.
        labels (numpy.ndarray): Array of class labels for each bounding box.
        iou_threshold (float): IoU threshold for filtering overlapping boxes.

    Returns:
        numpy.ndarray: Filtered bounding boxes.
        numpy.ndarray: Filtered scores.
        numpy.ndarray: Filtered labels.
    """
    # Lists to hold the final results after NMS
    filtered_bboxes = []
    filtered_scores = []
    filtered_labels = []

    # Get unique labels
    unique_labels = np.unique(labels)

    # Perform NMS independently for each class label
    for label in unique_labels:
        # Select boxes, scores, and indices for the current label
        label_indices = np.where(labels == label)[0]
        label_bboxes = bboxes[label_indices]
        label_scores = scores[label_indices]

        # Sort by score
        order = label_scores.argsort()[::-1]
        label_bboxes = label_bboxes[order]
        label_scores = label_scores[order]

        keep_indices = []

        while len(label_bboxes) > 0:
            # Select the box with the highest score
            i = 0
            keep_indices.append(label_indices[order[i]])

            # Calculate IoU of this box with the rest
            xx1 = np.maximum(label_bboxes[i][0], label_bboxes[1:, 0])
            yy1 = np.maximum(label_bboxes[i][1], label_bboxes[1:, 1])
            xx2 = np.minimum(label_bboxes[i][2], label_bboxes[1:, 2])
            yy2 = np.minimum(label_bboxes[i][3], label_bboxes[1:, 3])

            # Calculate the area of overlap
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap_area = w * h

            # Calculate IoU
            areas = (label_bboxes[:, 2] - label_bboxes[:, 0] + 1) * (label_bboxes[:, 3] - label_bboxes[:, 1] + 1)
            iou = overlap_area / (areas[i] + areas[1:] - overlap_area)

            # Keep only boxes with IoU less than the threshold
            below_threshold = np.where(iou <= iou_threshold)[0]
            label_bboxes = label_bboxes[below_threshold + 1]
            label_scores = label_scores[below_threshold + 1]
            order = order[below_threshold + 1]

        # Append filtered results for this label
        filtered_bboxes.extend(bboxes[keep_indices])
        filtered_scores.extend(scores[keep_indices])
        filtered_labels.extend(labels[keep_indices])

    return np.array(filtered_bboxes), np.array(filtered_scores), np.array(filtered_labels)

def filter_bboxes_by_score(bboxes, scores, labels, score_threshold=0.5):
    """
    Filters out bounding boxes with scores below a specified threshold.

    Parameters:
        bboxes (numpy.ndarray): Array of bounding boxes (x_min, y_min, x_max, y_max).
        scores (numpy.ndarray): Array of confidence scores for each bounding box.
        labels (numpy.ndarray): Array of labels.
        score_threshold (float): Minimum score threshold to keep a bounding box.

    Returns:
        numpy.ndarray: Filtered bounding boxes.
        numpy.ndarray: Filtered scores.
    """
    # Get indices where scores are above the threshold
    keep_indices = np.where(scores >= score_threshold)[0]

    # Filter bounding boxes and scores using the indices
    filtered_bboxes = bboxes[keep_indices]
    filtered_scores = scores[keep_indices]
    filtered_labels = labels[keep_indices]

    return np.array(filtered_bboxes), np.array(filtered_scores), np.array(filtered_labels)

def remove_large_containing_boxes(bboxes, scores, labels):
    """
    Removes larger bounding boxes that contain smaller ones with higher scores.

    Parameters:
        bboxes (numpy.ndarray): Array of bounding boxes (x_min, y_min, x_max, y_max).
        scores (numpy.ndarray): Array of confidence scores for each bounding box.
        labels (numpy.ndarray): Array of labels.

    Returns:
        numpy.ndarray: Filtered bounding boxes.
        numpy.ndarray: Filtered scores.
    """
    keep_indices = []

    # Iterate over each bounding box
    for i in range(len(bboxes)):
        keep_box = True
        for j in range(len(bboxes)):
            if i == j:
                continue

            # Check if bbox[j] is contained within bbox[i]
            if (bboxes[j][0] >= bboxes[i][0] and bboxes[j][1] >= bboxes[i][1] and
                bboxes[j][2] <= bboxes[i][2] and bboxes[j][3] <= bboxes[i][3]):

                # If inner box has a higher score, mark outer box for removal
                if scores[j] > scores[i]:
                    keep_box = False
                    break

        if keep_box:
            keep_indices.append(i)

    # Filter bounding boxes and scores using the indices in `keep`
    filtered_bboxes = bboxes[keep_indices]
    filtered_scores = scores[keep_indices]
    filtered_labels = labels[keep_indices]

    return np.array(filtered_bboxes), np.array(filtered_scores), np.array(filtered_labels)
