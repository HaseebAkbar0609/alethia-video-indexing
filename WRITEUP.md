# Write-up

<!-- Answers to be written by the candidate, without AI (brief, section 8). -->

## 1. Your timings (≤ 100 words)
How long did your first working version take on the full 50 files, and how long did your final version take? Which single change saved the most time?

I did not process all 50 files for the initial processing. Shard 49 was processed such that YOLO processed every frame (30fps). This is very slow, and processing all the files in this way would take over 6 hours. However, to speed it up, a sampling rate of 2fps was used, increasing to 8fps if there was discrepancy between the two samples. Using this method on shard 49 gave agreement with the perfect answer 99% of the time. Processing all 50 files this way took 2h 25m. Frame skipping was highly effective in speeding things up. Only 12% of frames were processed by YOLO.

## 2. Your pipeline (≤ 150 words)
Describe what your script does, step by step, from opening a Parquet file to writing an output line.

Open the parquet file containing all videos. For each video, the worker will extract 2 frames per second of video. It will then resize these frames and pass them to the main program. Once 64 frames are received, they are sent to the GPU. We can then use YOLO, the object detection algorithm, to know which objects (from the list of 5) were present in the video, and how many people. If there are the same results in both frames of the second, we finish processing this second. If they differ, then we take 8 extra frames from this second. For each second, we write a line containing the name of the video, the second number, objects detected, and number of people. If a video was corrupted, we simply log an error and continue.

## 3. Where the time goes (≤ 150 words)
Which part of your final version takes the most time? How did you measure this?

The decoding is very slow. However, the decoding steps are done on the CPU. The YOLO inference is quick, and is done on the GPU.
I have added in a small timer to run.py. This measures how long the GPU is running, and how long the main program has to wait for images to be returned from workers. For each shard, this data is written to a small stats file.
For one shard, the total runtime was 356 seconds, with only 114 seconds being spent using the GPU. For all 50 shards, the usage pattern was very similar, and in general only around a third of the time was spent using the GPUs.
As the Kaggle machine has only 4 slow CPU cores, while having 2 fast GPUs, the decoding step becomes a bottleneck

## 4. Checking correctness (≤ 100 words)
How did you check that your results stayed correct after you made the script faster?

10 simple automated tests were written which are run every time there is an alteration. These ensure that the resize and GPU detector are working correctly with real frames, to match Ultralytics. Speed was tested by comparing to a known correct result. For shard 49, YOLO was run on each frame using my code and the official Ultralytics YOLO. Comparing detection results second by second, my results match correctly. Objects were detected with 0.993 confidence. People were counted correctly in 98.5% of cases and within 1 person in 99.96% of cases.

## 5. Working with AI (≤ 100 words)
Describe one suggestion from an AI tool that was wrong or that you decided not to use. How did you find out it was wrong, or why did you reject it?

The AI was poorly set up. We were running on Kaggle (4 cores), however, in the companies computer there are 32 cores and 64GB RAM. In run.py, the AI has started 1 video helper for each core. Assuming each video helper requires 0.5GB RAM, we should be able to run 34 video helpers. However, measuring real video helper RAM usage, we find that they require 2.6GB RAM (5x more). Therefore, we would require 90GB to run 34 video helpers. Since we have only 64GB RAM, we would get some crashes. We have fixed this. run.py now checks the available RAM and only creates as many video helpers as possible (about 20). Repeating this, we get the same results and no crashes.

## 6. Next steps (≤ 100 words)
If you had one more week, what would you change, and why?

If I had more time, I would improve my code further. Firstly, using the GPU for decoding (NVIDIA have an extra chip for decoding videos in GPUs called NVDEC) will significantly speed up the process, as it means that the CPU is no longer causing a bottleneck. Secondly, I would run my code on a wider variety of files. As I chose my parameters based on data from shard 49, I could check that they work well on longer videos such as those in the company (13 hours). Additionally, I would improve performance on the poor results. In particular, I would only take extra frames in these periods in phones or busy scenes.

