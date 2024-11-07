

def transform_box(box, height, width):
    
    x_center = box[0] * width
    y_center = box[1] * height
    box_width = box[2] * width
    box_height = box[3] * height

    x_min = x_center - box_width / 2
    y_min = y_center - box_height / 2
    x_max = x_center + box_width / 2
    y_max = y_center + box_height / 2

    # Chuẩn hóa tọa độ sang (0, 1)
    x_min_norm = x_min / width
    y_min_norm = y_min / height
    x_max_norm = x_max / width
    y_max_norm = y_max / height

    return [x_min_norm, y_min_norm, x_max_norm, y_max_norm]
    