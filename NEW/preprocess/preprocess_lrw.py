import argparse
import os
from os import path

from data_module import AVSRDataLoader
from utils import save2vid
from tqdm import tqdm
import torch
import random
import gc

import glob
import numpy as np
import torchvision


def load_video_text_data(data_folder, group):
    """
    Parses YAML files found in the mTEDx dataset and returns video and text
    samples.

    Arguments
    ---------
    data_folder: str
        The absolute/relative path to the directory where the video file is
        located.
    group : list
        The group to be processed, e.g "test".
    """

    video_samples = glob.glob(f"{data_folder}/*/{group}/*.mp4", recursive=True)
    text_samples = [sample.split('/')[-1].split('_')[0] for sample in video_samples]
    id_samples = list(np.arange(0, len(video_samples)))

    return video_samples, text_samples, id_samples



def process_video_text_sample(i, video_input_filepath, text, dst_vid_dir):
    """
    Process one data sample.

    Arguments
    ---------
    i: int
        The index of the video file.
    video: dict
        A dictionary describing info about the video segment like:
        speaker_id, duration, offset, ... etc.
    text: str
        The text of the video segment.
    data_folder: str
        The absolute/relative path where the mTEDx data can be found.
    save_folder: str
        The absolute/relative path where the mTEDx data will be saved.
    
    Returns
    -------
    dict:
        A dictionary of audio-text segment info. 
    """

    text = text.lower()

    if text is None:
        return None
    
    
    video_output_filepath = (
        f"{dst_vid_dir}/{video_input_filepath.split('/')[-1]}"
    )

    

    if os.path.exists(video_output_filepath) or not os.path.exists(video_input_filepath):
        return None
    
    video = torchvision.io.read_video(video_input_filepath, pts_unit="sec")[0].numpy()
    # print(video)
    try:
        landmarks = vid_dataloader.landmarks_detector(video)
        # print('in here')
        # print(landmarks)
        if landmarks is not None:
            video_data = vid_dataloader.video_process(video, landmarks)
            # print(video_data)
            if video_data is None:
                return None
        else: return None
        video_length = len(video_data)
        # print(video_length)
        # if video_length <= seg_duration * fps and video_length >= OUT_FPS:
        # if video_length <= seg_duration * fps:

        save2vid(video_output_filepath, video_data, fps)
        return video_output_filepath, video_data.shape[0], text
    except:
        return None



def preprocess(args, subset):
    """
    Preprocess the mTEDx data found in the given language and the given group.
    Also, it writes the video-text information in a json file.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original mTEDx dataset is stored.
    save_folder : str
        Location of the folder for storing the csv.
    lang : str
        The language code.
    group : list
        The group to be processed, e.g "test".
    """

    label_filename = os.path.join(
    args.root_dir,
    "labels",
    f"{args.dataset}_{subset}_transcript_lengths_seg{args.seg_duration}s.csv"
    if args.groups <= 1
    else f"{args.dataset}_{subset}_transcript_lengths_seg{args.seg_duration}s_{args.groups}.{args.job_index}.csv",
    )

    
    print(label_filename)

    

    os.makedirs(os.path.dirname(label_filename), exist_ok=True)
    print(f"Directory {os.path.dirname(label_filename)} created")

    if path.exists(label_filename):
        f = open(label_filename, 'a')
        l = open(label_filename, 'r').readlines()
        print(f"File {label_filename} exists")
    else:
        f = open(label_filename, "w")
        l= []
        print(f"File {label_filename} created")

    # l = f.readlines()
    
    flag_open_labels = True

    dst_vid_dir = os.path.join(
        args.root_dir, args.dataset, args.dataset + f"_video_seg{args.seg_duration}s"
    )

    os.makedirs(dst_vid_dir, exist_ok=True)


    words_list = os.listdir(args.data_dir)

    # print(args.data_dir, lang, subset)
    video_samples, text_samples, id_samples = load_video_text_data(args.data_dir, subset)
    
    # processed_samples = 
    # i_samples = list(np.arange(0, len(video_samples)))
    if subset == 'train':
        print('Shuffle for train')
        zipped = list(zip(video_samples, text_samples, id_samples))
        random.seed(11)
        random.shuffle(zipped)
        video_samples, text_samples, id_samples = zip(*zipped)
    print(len(video_samples))
    if len(l) != 0:
        print(f'There are some lines in file: {len(l)}')
        last = int(l[-1].split(',')[1][-9:-4])
        print(last)
        print(id_samples.index(last))
        print(id_samples[last])
        print(id_samples[id_samples.index(last)])
        last_ind = id_samples.index(last)
        video_samples, text_samples, id_samples = video_samples[last_ind+1:], text_samples[last_ind+1:], id_samples[last_ind+1:]

    print(len(video_samples))
    for i, (video, text, id) in tqdm(enumerate(zip(video_samples, text_samples, id_samples))):
        print('here')
        line = process_video_text_sample(id, video, text, dst_vid_dir)
        if line is not None:
            basename = os.path.relpath(line[0], start=os.path.join(args.root_dir, args.dataset))
            video_len, content = line[1], line[2]
            # token_id_str = " ".join(map(str, [_.item() for _ in text_transform.tokenize(content)]))
            if not flag_open_labels:
                        f = open(label_filename, "a")
                        flag_open_labels = True

            if flag_open_labels:
                f.write("{}\n".format(f"{args.dataset},{basename},{video_len},{content}"))
                f.close()
                flag_open_labels = False
            torch.cuda.empty_cache()
            gc.collect()



if __name__ == "__main__":
    # Argument Parsing
    parser = argparse.ArgumentParser(description="mTEDx Preprocessing")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory of original dataset",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="retinaface",
        help="Type of face detector. (Default: retinaface)",
    )
    parser.add_argument(
        "--landmarks-dir",
        type=str,
        default=None,
        help="Directory of landmarks",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="Root directory of preprocessed dataset",
    )
    parser.add_argument('--subset', nargs='+', default="test valid train",
                help='List of groups separated by space, e.g. "valid train".')
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of dataset",
    )
    parser.add_argument(
        "--seg-duration",
        type=int,
        default=24,
        help="Max duration (second) for each segment, (Default: 24)",
    )
    parser.add_argument(
        "--groups",
        type=int,
        default=1,
        help="Number of threads to be used in parallel.",
    )
    parser.add_argument(
        "--job-index",
        type=int,
        default=0,
        help="Index to identify separate jobs (useful for parallel processing).",
    )
    args = parser.parse_args()

    seg_duration = args.seg_duration
    dataset = args.dataset
    fps = 25

    # if "lrw" in dataset.lower():
    seg_duration = 1.16
    args.seg_duration = 1.6
    #     landmarks_detector = None
    # else: 
    if args.detector == "retinaface" : from detectors.retinaface.detector import LandmarksDetector
    elif args.detector == "mediapipe": from detectors.mediapipe.detector import LandmarksDetector
    landmarks_detector = LandmarksDetector()
        # text_transform = TextTransform()


    # Load Data directory
    args.data_dir = os.path.normpath(args.data_dir)

    vid_dataloader = AVSRDataLoader(
        modality="video", detector=args.detector, convert_gray=False
    )

    for subset in args.subset:
        print(subset)
        preprocess(args, subset)



