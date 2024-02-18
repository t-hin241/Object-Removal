import cv2
from sd_inpaint import inpaint
from utils import load_img_to_array, dilate_mask,save_array_to_img,show_mask
from matplotlib import pyplot as plt
from segment_pipeline import *
import torch

def get_clicked_point(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (512,512))
    cv2.namedWindow("image") 
    cv2.imshow("image", img)
    cv2.resizeWindow("image", 512, 512)

    last_point = []
    keep_looping = True

    def mouse_callback(event, x, y, flags, param):
        nonlocal last_point, keep_looping, img

        if event == cv2.EVENT_LBUTTONDOWN:
            if last_point:
                cv2.circle(img, tuple(last_point), 5, (0, 0, 0), -1)
            last_point = [x, y]
            cv2.circle(img, tuple(last_point), 5, (0, 0, 255), -1)
            cv2.imshow("image", img)
        elif event == cv2.EVENT_RBUTTONDOWN:
            keep_looping = False

    cv2.setMouseCallback("image", mouse_callback)

    while keep_looping:
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    return last_point

if __name__ == "__main__":
    img_path = "./images/1.jpg"
    mask_p = "./outputs/mask_of_" + img_path.split("/")[-1]
    output_path = "./outputs/result_of_" + img_path.split("/")[-1]
    input_img = load_img_to_array(img_path)
    selected_point = get_clicked_point(img_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam_ckpt = "./segment_model/pretrained_weights/sam_vit_h_4b8939.pth"
    dilate_kernel_size = 15

    mask = predict_masks_with_sam(
        input_img,
        np.array([selected_point]),
        point_labels=np.array([1]),
        model_type='vit_h',
        ckpt_p=sam_ckpt,
        device=device,
    )
    mask = mask.astype(np.uint8) * 255
    mask = dilate_mask(mask, dilate_kernel_size)

    save_array_to_img(mask, mask_p)
    print("Segment Finish!")

    input_mask = load_img_to_array(mask_p)
    img_filled = inpaint(input_img, input_mask, device=device)
    save_array_to_img(img_filled, output_path)


