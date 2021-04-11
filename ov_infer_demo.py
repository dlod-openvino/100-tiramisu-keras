from openvino.inference_engine import IECore
import numpy as np
import time
import cv2 as cv
import os,sys 
from camvid.mapping import decode

import argparse

def parse_args(args):
    parser = argparse.ArgumentParser(description='Script for inference with models trained on CamVid data and optimized by OpenVINO')

    parser.add_argument('-d',
                        help='CPU|GPU',
                        default='cpu')
    

    return parser.parse_args(args)

def color_label(img, id2code):
    rows, cols = img.shape
    result = np.zeros((rows, cols, 3), 'uint8')
    for j in range(rows):
        for k in range(cols):
            result[j, k] = id2code[img[j, k]]
    return result
    
def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    ie = IECore()
    for device in ie.available_devices:
        print(device)

    model_xml = "models/my_tiramisu.xml"
    model_bin = "models/my_tiramisu.bin"
    image_folder = "test"
    test_image_files = os.listdir(image_folder)

    label_codes, label_names, code2id = decode('camvid-master/label_colors.txt')
    id2code = {val: key for (key, val) in code2id.items()}
    print(id2code)

    net = ie.read_network(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))

    n, h, w, c = net.input_info[input_blob].input_data.shape
    print(n, h, w, c)

    exec_net = ie.load_network(network=net, device_name=args.d)

    infer_time_list = []
    for image_file in test_image_files:
        frame = cv.imread(os.path.join(image_folder, image_file))
        
        image = cv.resize(frame, (w, h))
        # image = image.transpose(2, 0, 1)
        inf_start = time.time()
        res = exec_net.infer(inputs={input_blob:[image]})
        inf_end = time.time() - inf_start
        infer_time_list.append(inf_end)
        res = res[out_blob].reshape((n, h, w, 32))
        res = np.squeeze(res, 0)
        res = np.argmax(res, axis=-1)
        hh, ww = res.shape
        print(res.shape)
        mask = color_label(res,id2code)    
        mask = cv.resize(mask, (frame.shape[1], frame.shape[0]))
        result = cv.addWeighted(frame, 0.5, mask, 0.5, 0)
        cv.putText(result, "infer time(ms): %.3f, FPS: %.2f"%(inf_end*1000, 1/(inf_end+0.0001)), (10, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2, 8)
        cv.imshow("semantic segmentation benchmark", result)
        cv.waitKey(0) # wait for the image show

    infer_times = np.array(infer_time_list)
    avg_infer_time = np.mean(infer_times)
    print("infer time(ms): %.3f, FPS: %.2f"%(avg_infer_time*1000, 1/(avg_infer_time+0.0001)))
    cv.destroyAllWindows()

if __name__ == "__main__":
    
    main()
