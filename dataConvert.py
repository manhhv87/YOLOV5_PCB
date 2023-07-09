import glob

# address where the picture is stored
train_image_path = "datasets/pcb-yolo-en-clean-v2/images/train/"
valid_image_path =  "datasets/pcb-yolo-en-clean-v2/images/val/"
test_image_path =  "datasets/pcb-yolo-en-clean-v2/images/test/"
# path to the generated txt
txt_path = "datasets/pcb-yolo-en-clean-v2"


def generate_train_and_val(image_path, txt_file):
    with open(txt_file, 'w') as tf:
        for jpg_file in glob.glob(image_path + '/' + '*.jpg'):
            tf.write(jpg_file + '\n')


generate_train_and_val(train_image_path, txt_path + 'train.txt')
generate_train_and_val(valid_image_path, txt_path + 'valid.txt')
generate_train_and_val(test_image_path, txt_path + 'test.txt')
