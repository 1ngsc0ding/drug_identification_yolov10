import gradio as gr
import cv2
import supervision as sv
from ultralytics import YOLOv10
# https://www.youtube.com/watch?v=h53XYgKzYYE&t=1534s
# https://github.com/Codewello/Yolo-v8-and-hugginface/blob/main/hugginface/app.py
# https://www.youtube.com/watch?v=29tnSxhB3CY&t=564s
# https://colab.research.google.com/drive/1Sv3_0S4zhZT763bBMjYxf0HhTit4Bvh6?usp=sharing#scrollTo=kAi4PvrItTCf
# https://github.com/roboflow/supervision/blob/develop/demo.ipynb
def yoloV10_func(image: gr.Image() = None,
                image_size: gr.Slider() = 640,
                conf_threshold: gr.Slider() = 0.25,
                iou_threshold: gr.Slider() = 0.45):
    """This function performs YOLOv10 object detection on the given image.

    Args:
        image (gr.Image, optional): Input image to detect objects on. Defaults to None.
        image_size (gr.Slider, optional): Desired image size for the model. Defaults to 640.
        conf_threshold (gr.Slider, optional): Confidence threshold for object detection. Defaults to 0.4.
        iou_threshold (gr.Slider, optional): Intersection over Union threshold for object detection. Defaults to 0.50.
    """
    # Load the YOLOv10 model from the 'best.pt' checkpoint
    model_path = './pills_yolov10.pt'
    model = YOLOv10(model_path)

    # Perform object detection on the input image using the YOLOv10 model
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.predict(image,
                            conf=conf_threshold,
                            iou=iou_threshold,
                            imgsz=image_size)

    # Print the detected objects' information (class, coordinates, and probability)
    box = results[0].boxes
    print("Object type:", box.cls)
    print("Coordinates:", box.xyxy)
    print("Probability:", box.conf)

    # Render the output image with bounding boxes around detected objects
    # render = render_result(model=model, image=image, result=results[0])
    # results = model(image)[0]
    detections = sv.Detections.from_ultralytics(results[0])
    box_annotator = sv.BoxAnnotator()
    labels = [
        f"{model.model.names[class_id]} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]
    annotated_image = box_annotator.annotate(
        image.copy(), detections=detections, labels=labels
    )
    return annotated_image
inputs = [
    gr.Image(type="filepath", label="Input Image"),
    gr.Slider(minimum=120, maximum=1280, value=640,
                     step=32, label="Image Size"),
    gr.Slider(minimum=0.0, maximum=1.0, value=0.25,
                     step=0.05, label="Confidence Threshold"),
    gr.Slider(minimum=0.0, maximum=1.0, value=0.45,
                     step=0.05, label="IOU Threshold"),
]
outputs = gr.Image(type="filepath", label="Output Image")

title = "YOLOv10 101: Custom Object Detection on Pill Types"


examples = [['RXNAV-600_13668-0095-90_RXNAVIMAGE10_D145E8EF.jpg', 640, 0.2, 0.7],
            ['RXBASE-600_00071-1014-68_NLMIMAGE10_5715ABFD.jpg', 280, 0.2, 0.6],
            ['RXBASE-600_00074-7126-13_NLMIMAGE10_C003606B.jpg', 640, 0.2, 0.8]]
yolo_app = gr.Interface(
    fn=yoloV10_func,
    inputs=inputs,
    outputs=outputs,
    title=title,
    examples=examples,
    cache_examples=True,
)
# Launch the Gradio interface in debug mode with queue enabled
yolo_app.launch(debug=True)
