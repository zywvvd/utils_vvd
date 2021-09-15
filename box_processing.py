import numpy as np
from .utils import current_system
from .utils import is_number


def is_box(box):
    """
    check if the input box is a box
    """
    def is_iter_box(iter_box):
        if len(iter_box) != 4:
            return False
        else:
            for num in iter_box:
                if not is_number(num):
                    return False
        return True

    if isinstance(box, list):
        return is_iter_box(box)
    elif isinstance(box, tuple):
        return is_iter_box(box)
    elif isinstance(box, np.ndarray):
        return is_iter_box(box)
    else:
        return False


def compute_box_area(box):
    """
    compute area of input box
    """
    assert is_box(box)
    area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    return area


def compute_box_box_iou(box1, box2):
    """
    compute iou of input boxes
    """
    area1 = compute_box_area(box1)
    area2 = compute_box_area(box2)
    u_x_min = max(box1[0], box2[0])
    u_x_max = min(box1[2], box2[2])
    u_y_min = max(box1[1], box2[1])
    u_y_max = min(box1[3], box2[3])
    new_box = [u_x_min, u_y_min, u_x_max, u_y_max]
    u_area = compute_box_area(new_box)
    iou = u_area / max(area1 + area2 - u_area, 1)
    return iou


def get_xyxy(polygon_xy):
    """
    get xyxy of a ploygon
    """
    polygon_array = np.array(polygon_xy)
    assert polygon_array.shape[0] > 1
    assert polygon_array.shape[1] == 2
    x1, y1, x2, y2 = polygon_array[:, 0].min(), polygon_array[:, 1].min(), polygon_array[:, 0].max(), polygon_array[:, 1].max()
    return [x1, y1, x2, y2]


def make_box(center_point, box_x, box_y=None):
    """ build box for a given center-point"""
    box_x = int(box_x)
    if box_y is None:
        box_y = box_x
    else:
        box_y = int(box_y)
    assert box_x > 0 and box_y > 0
    center_x, center_y = center_point

    left = int(round(center_x - box_x // 2))
    right = left + box_x
    top = int(round(center_y - box_y // 2))
    bottom = top + box_y

    box = [left, top, right, bottom]
    return box


def boxes_painter(rgb_image, box_list, label_list=None, score_list=None, color_list=None, color=None, class_name_dict=None, line_thickness=3):
    """[paint boxex and labels on image]

    Args:
        rgb_image ([np.array(uint8)]): [np array image as type uint8]
        box_list ([list of list of 4 int]): [list of box like [10(xmin), 20(ymin), 50(xmax), 60(ymax)]]
        label_list ([list of int]): [class indexes of boxes in box_list] (could be none)
        class_name_dict ([dict - index: class_name]): [key is index and value is the name in type of str] (could be none)
    Returns:
        [rgb image]: [image with boxes and labels]
    """

    if rgb_image.ndim == 2:
        rgb_image = (np.repeat(rgb_image[:, :, None], 3, axis=2))

    rgb_image = rgb_image.astype('uint8')

    color_input = color

    if label_list is not None:
        assert len(label_list) == len(box_list)
        if class_name_dict is not None:
            for item in label_list:
                assert item in class_name_dict

    if score_list is not None:
        assert len(score_list) == len(box_list)

    if color_list is not None:
        assert len(color_list) == len(box_list)

    from PIL import ImageFont, ImageDraw, Image
    import matplotlib.font_manager as fm

    color_list_default = [(159, 2, 98), (95, 32, 219), (222, 92, 189), (56, 233, 120), (23, 180, 100), (78, 69, 20), (97, 202, 39), (65, 179, 135), (163, 159, 219)]

    pil_image = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(pil_image)

    fontsize = 24

    try:
        if current_system() == 'Windows':
            font = ImageFont.truetype('arial.ttf', fontsize)
        else:
            font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')),fontsize)
    except IOError:
        font = ImageFont.load_default()

    text_height = 22

    # draw boxes
    for index, bbox in enumerate(box_list):
        if not bbox:
            continue
        left, top, right, bottom = np.array(bbox).astype('int').tolist()
        if color_list is not None:
            color = color_list[index]
        else:
            if label_list:
                color = color_list_default[label_list[index] % len(color_list_default)]
            else:
                color = (255, 255, 0)

        # draw box
        if color_input:
            color = tuple(color_input)

        draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=line_thickness, fill=color)

        # draw text
        display_str = ""

        if label_list:
            if class_name_dict:
                display_str += class_name_dict[label_list[index]]
            else:
                display_str += str(label_list[index])

        if score_list:
            if display_str != "":
                display_str += ' '
            score = score_list[index]
            display_str += str(format(score, '.3f'))

        text_width, text_height = font.getsize(display_str)

        text_bottom = top

        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left - 1, text_bottom - text_height - 2 * margin), (right + 1, text_bottom)], fill=color)
        if np.mean(np.array(color)) < 250:
            font_color = 'yellow'
        else:
            font_color = 'red'
        draw.text((int(left + (right - left)/2 - text_width/2), text_bottom - text_height - margin), display_str, fill=font_color, font=font)

    # get image with box and index
    array_image_with_box = np.asarray(pil_image)

    return array_image_with_box