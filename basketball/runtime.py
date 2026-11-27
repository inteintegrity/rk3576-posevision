"""Bound native CPU thread pools to avoid oversubscribing the ARM cores."""
import os
import cv2
from threadpoolctl import threadpool_limits

def configure_cpu():
    threads=int(os.environ.get('BASKETBALL_CPU_THREADS','2'))
    if threads<1:raise ValueError('BASKETBALL_CPU_THREADS must be positive')
    cv2.setNumThreads(threads)
    # Kalman matrices and per-track cosine products are small. Large BLAS pools
    # spend more time synchronizing than computing these products.
    return threadpool_limits(limits=1,user_api='blas')
