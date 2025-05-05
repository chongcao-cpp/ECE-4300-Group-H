import cv2
import time
import numpy as np
from ultralytics import YOLO

MODELS = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
VIDEO_PATH = 'vid1.mp4'     
NUM_FRAMES = 100                 
SHOW_WINDOW = False

benchmark_results = []

for model_path in MODELS:
    print(f"\nStarting Benchmark for Model: {model_path}\n")

    model = YOLO(model_path)
    cap = cv2.VideoCapture(VIDEO_PATH)

    frame_count = 0
    inference_times = []
    start_time = time.time()

    while frame_count < NUM_FRAMES:
        ret, frame = cap.read()
        if not ret:
            print("\nEnd of video reached before getting enough frames.")
            break

        t0 = time.time()
        results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)
        t1 = time.time()

        inference_times.append(t1 - t0)
        frame_count += 1

        if SHOW_WINDOW:
            annotated_frame = results[0].plot()
            cv2.imshow(f"Detection - {model_path}", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        elapsed_time = time.time() - start_time
        avg_time_per_frame = elapsed_time / frame_count
        frames_left = NUM_FRAMES - frame_count
        estimated_time_left = frames_left * avg_time_per_frame
        current_fps = 1 / avg_time_per_frame if avg_time_per_frame > 0 else 0

        m, s = divmod(estimated_time_left, 60)
        h, m = divmod(m, 60)
        time_left_str = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

        print(f"Model: {model_path} | Frames: {frame_count}/{NUM_FRAMES} | FPS: {current_fps:.2f} | ETA: {time_left_str}", end='\r')

    cap.release()
    cv2.destroyAllWindows()

    end_time = time.time()

    total_time = end_time - start_time
    avg_inference_time = np.mean(inference_times)
    avg_fps = len(inference_times) / total_time
    min_inference_time = np.min(inference_times)
    max_inference_time = np.max(inference_times)
    std_dev_inference_time = np.std(inference_times)
    median_inference_time = np.median(inference_times)

    benchmark_results.append({
        "Model": model_path,
        "Frames": frame_count,
        "Total Time (s)": total_time,
        "Avg Inference Time (ms)": avg_inference_time * 1000,
        "Median Inference Time (ms)": median_inference_time * 1000,
        "Min Inference Time (ms)": min_inference_time * 1000,
        "Max Inference Time (ms)": max_inference_time * 1000,
        "Std Dev (ms)": std_dev_inference_time * 1000,
        "Avg FPS": avg_fps
    })

print("\n\n====== YOLOv8 Benchmark Summary ======")
for result in benchmark_results:
    print(f"\nModel: {result['Model']}")
    print(f"Frames Processed: {result['Frames']}")
    print(f"Total Time: {result['Total Time (s)']:.2f} seconds")
    print(f"Avg Inference Time: {result['Avg Inference Time (ms)']:.2f} ms")
    print(f"Median Inference Time: {result['Median Inference Time (ms)']:.2f} ms")
    print(f"Min Inference Time: {result['Min Inference Time (ms)']:.2f} ms")
    print(f"Max Inference Time: {result['Max Inference Time (ms)']:.2f} ms")
    print(f"Std Dev: {result['Std Dev (ms)']:.2f} ms")
    print(f"Average FPS: {result['Avg FPS']:.2f}")
print("\n=======================================\n")
