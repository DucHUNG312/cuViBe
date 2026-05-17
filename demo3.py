##############################################
# Demo3: ViBe CPU vs cuViBe GPU comparison
# Usage:
#   python demo3.py [input_video]
# Shows side-by-side: [Input | ViBe CPU mask | cuViBe GPU mask]
# Press 'q' to quit
##############################################

import sys
import time
import numpy as np
import cv2
import cupy as cp
import pybgs as bgs

import faulthandler

faulthandler.enable()

print("OpenCV:", cv2.__version__)

input_src = sys.argv[1] if len(sys.argv) > 1 else "dataset/video.avi"

capture = cv2.VideoCapture(input_src)
if not capture.isOpened():
    print("Cannot open:", input_src)
    sys.exit(1)

W, H = 640, 480

print("Layout: [Input | ViBe CPU mask | cuViBe GPU mask]")
print("Press 'q' to quit")

cpu_bgs = bgs.ViBe()
gpu_bgs = bgs.cuda.ViBe()

frame_count = 0
cpu_total_ms = 0.0
gpu_total_ms = 0.0

while True:
    ok, frame = capture.read()
    if not ok or frame is None:
        break

    frame = cv2.resize(frame, (W, H))
    frame_count += 1

    # ---- CPU ViBe ----
    t0 = time.perf_counter()
    fg_cpu = cpu_bgs.apply(frame)
    cpu_ms = (time.perf_counter() - t0) * 1000.0
    cpu_total_ms += cpu_ms

    if fg_cpu is None or fg_cpu.size == 0:
        fg_cpu = np.zeros((H, W), dtype=np.uint8)

    # ---- CUDA ViBe ----
    frame_gpu = cp.asarray(frame)

    t1 = time.perf_counter()
    fg_gpu_cp = gpu_bgs.apply(frame_gpu)
    cp.cuda.Stream.null.synchronize()  # sync before stopping timer
    gpu_ms = (time.perf_counter() - t1) * 1000.0
    gpu_total_ms += gpu_ms

    fg_gpu = cp.asnumpy(fg_gpu_cp)  # download for display

    # Convert grayscale masks to BGR
    cpu_disp = cv2.cvtColor(fg_cpu, cv2.COLOR_GRAY2BGR)
    gpu_disp = cv2.cvtColor(fg_gpu, cv2.COLOR_GRAY2BGR)

    # Annotate
    cv2.putText(frame, "Input", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(
        cpu_disp,
        f"ViBe CPU  {cpu_ms:.1f} ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 220, 255),
        2,
    )
    cv2.putText(
        gpu_disp,
        f"cuViBe GPU {gpu_ms:.1f} ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 220, 255),
        2,
    )

    cv2.imshow("ViBe CPU vs cuViBe GPU", np.hstack([frame, cpu_disp, gpu_disp]))

    if frame_count % 30 == 0:
        print(
            f"frame {frame_count:4d} | CPU avg {cpu_total_ms/frame_count:.2f} ms "
            f"| GPU avg {gpu_total_ms/frame_count:.2f} ms "
            f"| speedup {cpu_total_ms/gpu_total_ms:.2f}x"
        )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()

if frame_count > 0:
    print(f"\n=== Results over {frame_count} frames ===")
    print(f"CPU ViBe  avg: {cpu_total_ms/frame_count:.2f} ms/frame")
    print(f"CUDA ViBe avg: {gpu_total_ms/frame_count:.2f} ms/frame")
    print(f"Speedup:       {cpu_total_ms/gpu_total_ms:.2f}x")
