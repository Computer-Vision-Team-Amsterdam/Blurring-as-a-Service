from blurring_as_a_service.metrics.metrics_utils import (
    generate_binary_mask,
    parse_labels,
)


def process_image_labels(labels):

    true_classes, true_bboxes = parse_labels(labels["true"])
    pred_classes, pred_bboxes = parse_labels(labels["predicted"])

    tba_true_mask = generate_binary_mask(true_bboxes)
    tba_pred_mask = generate_binary_mask(pred_bboxes)

    # discard true and pred classes which are licence_plates
    person_true_bboxes_filtered = [
        true_bboxes[i] for i in range(len(true_bboxes)) if true_classes[i] == 0
    ]
    person_pred_bboxes_filtered = [
        pred_bboxes[i] for i in range(len(pred_bboxes)) if pred_classes[i] == 0
    ]

    uba_true_mask = generate_binary_mask(
        person_true_bboxes_filtered, consider_upper_half=True
    )
    uba_pred_mask = generate_binary_mask(
        person_pred_bboxes_filtered, consider_upper_half=True
    )

    return tba_true_mask, tba_pred_mask, uba_true_mask, uba_pred_mask
