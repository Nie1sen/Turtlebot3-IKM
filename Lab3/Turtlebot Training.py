from ultralytics import YOLO

def main():

    model = YOLO("yolov8n.pt")

    model.train(
        data="Turtlebots.yolov8/data.yaml",
        epochs=50,
        imgsz=640,
        batch=32,
        device=0,   # use GPU
        workers = 10, # use cpu to load images
        cache = True, # cache images for faster training
        amp = True, # use mixed precision training for faster training and less memory usage
        exist_ok = True, # overwrite existing runs
    )

if __name__ == "__main__":
    main()
