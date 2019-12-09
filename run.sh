echo Each experiment runs five times, shows the fastest/average runtime below:
echo Run Python CPU:
python3 benchmark.py py cpu

echo Run C++ CPU:
python3 benchmark.py cpp cpu

echo Run Python GPU:
python3 benchmark.py py gpu

echo Run C++ GPU:
python3 benchmark.py cpp gpu

echo Run CUDA GPU:
python3 benchmark.py cuda gpu
