from ultralytics import YOLO

def train_asl():
    # 1. 数据集配置文件路径
    data_yaml_path = "datasets/abc/data.yaml"

    # 2. 加载预训练模型
    model = YOLO("yolov8n.pt")  # 可换成 yolov8s/m/l/x 等模型

    # 3. 开始训练
    model.train(
        data=data_yaml_path,
        epochs=100,
        batch=16,
        imgsz=640,
        workers = 0,
        device="0"  # GPU训练，CPU改为"cpu"
    )
    print("训练完成！")


if __name__ == '__main__':
    train_asl()