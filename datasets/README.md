## Dataloading 

To try to make the framework as easy to install as possible we use a custom dataloader for videos that only requires installing ffmpeg-python from pip. This is simply python bindings for a ffmpeg binary. This allows us to load whole videos without extracting frames first with minimial latency on most systems.

## Instructions for installing different datasets


## Charades

https://allenai.org/plato/charades/

1. Download the RGB frames
2. Download the Annotations zip file from https://allenai.org/plato/charades/
3. Update the paths in the experiment file to the directories where you unzip those

## CharadesEgo

https://allenai.org/plato/charades/

1. Download the RGB frames
2. Download the Annotations zip file from https://allenai.org/plato/charades/
3. Update the paths in the experiment file to the directories where you unzip those

## Kinetics

https://deepmind.com/research/open-source/open-source-datasets/kinetics/

1. Download the Training and Validation zip files, those provide the URLs to the training data.
2. Run a YouTube scraping program such as youtube-dl ( https://rg3.github.io/youtube-dl/ ) to download each video
3. Since Kinetics only uses short snippets from these videos, we recommend trimming the video as they are downloaded. ffmpeg can be used for this purpose. We provide a download script under `datasets/dataset_scripts/download_kinetics_trim.py` that might help.
4. The final videos should be organized as `data_dir/action_name/youtubeid_000000_000000.mp4`

## AVA

https://research.google.com/ava/

1. Download the AVA train and test csv files, those provide the URLs to the training data.
2. Run a YouTube scraping program such as youtube-dl ( https://rg3.github.io/youtube-dl/ ) to download each video. We provide a download script under `datasets/dataset_scripts/download_ava.py` that might help.
3. Since the AVA videos are very long, we process the videos by chopping them into 3 second snippets. The dataloader then loads only the snippets needed for loading. We provide a processing script under `datasets/dataset_scripts/process_ava.py` that might help.
4.  The final videos should be organized as `data_dir/youtubeid_902_905.mp4` etc.

## ActivityNet

http://activity-net.org

1. Download the json annotation file, those provide the URLs to the training data.
2. Run a YouTube scraping program such as youtube-dl ( https://rg3.github.io/youtube-dl/ ) to download each video
3. We use ffmpeg to extract frames at 4fps, and resize the frames such that they have maximum dimension of 320px. We provide a bash script under `datasets/dataset_scripts/extract_frames.py` that might help.

## Something Something

https://20bn.com/datasets/something-something

1. Sign up and download the data and json annotation files.
2. This codebase directly uses the webm videos so no processing needed.

## Jester

https://20bn.com/datasets/jester

TODO
