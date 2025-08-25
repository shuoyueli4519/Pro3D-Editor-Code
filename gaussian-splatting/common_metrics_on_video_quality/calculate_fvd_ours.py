import numpy as np
import torch
from tqdm import tqdm
import os
import torchvision.transforms as T
import torchvision.io as io

def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)

    # permute BTCHW -> BCTHW
    x = x.permute(0, 2, 1, 3, 4) 

    return x

def calculate_fvd(videos1, videos2, device, method='styleganv', only_final=False):

    if method == 'styleganv':
        from fvd.styleganv.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
    elif method == 'videogpt':
        from fvd.videogpt.fvd import load_i3d_pretrained, frechet_distance
        from fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats

    print("calculate_fvd...")

    # videos [batch_size, timestamps, channel, h, w]
    
    assert videos1.shape == videos2.shape

    i3d = load_i3d_pretrained(device=device)
    fvd_results = []

    # support grayscale input, if grayscale -> channel*3
    # BTCHW -> BCTHW
    # videos -> [batch_size, channel, timestamps, h, w]

    videos1 = trans(videos1)
    videos2 = trans(videos2)

    fvd_results = []

    if only_final:

        assert videos1.shape[2] >= 10, "for calculate FVD, each clip_timestamp must >= 10"

        # videos_clip [batch_size, channel, timestamps, h, w]
        videos_clip1 = videos1
        videos_clip2 = videos2

        # get FVD features
        feats1 = get_fvd_feats(videos_clip1, i3d=i3d, device=device)
        feats2 = get_fvd_feats(videos_clip2, i3d=i3d, device=device)

        # calculate FVD
        fvd_results.append(frechet_distance(feats1, feats2))
    
    else:

        # for calculate FVD, each clip_timestamp must >= 10
        for clip_timestamp in tqdm(range(10, videos1.shape[-3]+1)):
        
            # get a video clip
            # videos_clip [batch_size, channel, timestamps[:clip], h, w]
            videos_clip1 = videos1[:, :, : clip_timestamp]
            videos_clip2 = videos2[:, :, : clip_timestamp]

            # get FVD features
            feats1 = get_fvd_feats(videos_clip1, i3d=i3d, device=device)
            feats2 = get_fvd_feats(videos_clip2, i3d=i3d, device=device)
        
            # calculate FVD when timestamps[:clip]
            fvd_results.append(frechet_distance(feats1, feats2))

    result = {
        "value": fvd_results,
    }

    return result

# test code / using example

def load_videos_from_folder(folder_path):
    video_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ])
    if not video_files:
        raise ValueError(f"文件夹 {folder_path} 中没有找到视频文件。")

    transform = T.Compose([
        T.Resize((512, 512))  # 统一缩放到 512x512
    ])

    all_videos = []
    for filename in video_files:
        video_path = os.path.join(folder_path, filename)
        vid, _, _ = io.read_video(video_path, pts_unit='sec')  # [T, H, W, C]
        vid = vid.permute(0, 3, 1, 2) / 255.0  # [T, C, H, W], 转成 0-1

        # 对每一帧做 resize
        vid_resized = torch.stack([transform(frame) for frame in vid])

        all_videos.append(vid_resized)

    # 统一帧数（取最短的）
    min_frames = min(v.shape[0] for v in all_videos)
    all_videos = [v[:min_frames] for v in all_videos]

    videos_tensor = torch.stack(all_videos)  # [N, T, C, H, W]
    videos_tensor = videos_tensor.permute(0, 1, 2, 3, 4)  # [N, C, T, H, W]

    return videos_tensor

def main():
    videos1 = load_videos_from_folder("./naive")
    videos2 = load_videos_from_folder("./naive_source")
    print(videos1.shape)
    device = torch.device("cuda")
    # device = torch.device("cpu")

    result = calculate_fvd(videos1, videos2, device, method='videogpt', only_final=True)
    print("[fvd-videogpt ]", result["value"])

    result = calculate_fvd(videos1, videos2, device, method='styleganv', only_final=True)
    print("[fvd-styleganv]", result["value"])

if __name__ == "__main__":
    main()
