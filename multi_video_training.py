import argparse
from pathlib import Path
import os
import re


def extract_stats_from_file(stat_file_path):
    """
    Extract average QP, PSNR, SSIM, and bitrate from SVT-AV1 stat file.
    Returns a dictionary with extracted values.
    """
    try:
        with open(stat_file_path, 'r') as f:
            content = f.read()
        
        # Look for the SUMMARY line and the statistics line after it
        summary_match = re.search(r'SUMMARY.*?\n.*?\n.*?\n\s*(\d+)\s+([0-9.]+)\s+([0-9.]+) dB\s+([0-9.]+) dB\s+([0-9.]+) dB.*?\|\s+([0-9.]+) dB\s+([0-9.]+) dB\s+([0-9.]+) dB.*?\|\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+).*?\|\s+([0-9.]+) kbps', content, re.DOTALL)
        
        if summary_match:
            stats = {
                'total_frames': int(summary_match.group(1)),
                'average_qp': float(summary_match.group(2)),
                'avg_psnr_y': float(summary_match.group(3)),
                'avg_psnr_u': float(summary_match.group(4)),
                'avg_psnr_v': float(summary_match.group(5)),
                'overall_psnr_y': float(summary_match.group(6)),
                'overall_psnr_u': float(summary_match.group(7)),
                'overall_psnr_v': float(summary_match.group(8)),
                'avg_ssim_y': float(summary_match.group(9)),
                'avg_ssim_u': float(summary_match.group(10)),
                'avg_ssim_v': float(summary_match.group(11)),
                'bitrate_kbps': float(summary_match.group(12))
            }
            return stats
        else:
            print(f"Could not parse summary statistics from {stat_file_path}")
            return None
            
    except FileNotFoundError:
        print(f"Stat file not found: {stat_file_path}")
        return None
    except Exception as e:
        print(f"Error parsing stat file {stat_file_path}: {e}")
        return None


def test_for_tbr(SVT_path, local_video_path, output_path, name, stat_file_path, tbr=600):
    """
    Test encoding with a specific target bitrate (tbr) and extract stats.
    """
    while True:
        command = f"{SVT_path} -i {local_video_path} -b {output_path / (name + '_test.ivf')} \
            --rc 2 --tbr {tbr} --pred-struct 1 --enable-stat-report 1 --stat-file {stat_file_path}"
            
        print(f"Running command: {command}")
        os.system(command)
        
        stats = extract_stats_from_file(stat_file_path)
        assert(stats is not None)
        
        qp = stats['average_qp'] if stats else None
        if qp is None:
            print("Failed to extract QP, exiting test.")
            break
        if 28 <= qp <= 32:
            print(f"Found suitable tbr: {tbr} with average QP: {qp}")
            break
        # Adjust tbr based on how far qp is from the target range
        if qp < 28:
            diff = 28 - qp
            # Decrease tbr, larger step if far, minimum step 100
            step = max(int(tbr * (diff / 32)), 100)
            tbr = max(100, tbr - step)
        elif qp > 32:
            diff = qp - 32
            # Increase tbr, larger step if far, minimum step 100
            step = max(int(tbr * (diff / 32)), 100)
            tbr = tbr + step
        else:
            # Should not reach here, but break just in case
            break
        
    return tbr, stats


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-video training script")
    parser.add_argument("--video_paths", type=str, nargs="+", help="A file that contains paths to the input videos")
    parser.add_argument("--output_dir", type=Path, help="Directory to save the output for all videos")
    parser.add_argument("--SVTAV1_path", type=Path, help="Path to the original SVT-AV1 encoder executable")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)
    
    # check if output directory exists, if not create it
    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # Dictionary to store all video statistics
    all_video_stats = {}

    # read each line in the video paths file
    video_paths_file = args.video_paths[0]
    with open(video_paths_file, 'r') as f:
        video_files = [line.strip() for line in f.readlines()]
        print(f"Found {len(video_files)} video files.")
        
        for video_file in video_files:
            print(f"Processing video: {video_file}")
            # the path will be something like https://media.xiph.org/video/derf/y4m/bus_cif.y4m, check if it is a url
            if video_file.startswith("http"):
                print(f"Video file is a URL: {video_file}")
                
            name = video_file.split('/')[-1].split('.')[0]
            print(f"Video name: {name}")
            
            # create output directory for each video
            video_output_dir = args.output_dir / name
            if not video_output_dir.exists():
                video_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Output directory: {video_output_dir}")
            
            # check if the video file is in output directory, if not download it
            local_video_path = video_output_dir / (name + ".y4m")
            if not local_video_path.exists():
                import os
                print(f"Downloading video from {video_file}")
                os.system(f"wget -O {local_video_path} {video_file}")

            # run original SVT-AV1 encoder to get the original output
            original_output = video_output_dir / "original"
            if not original_output.exists():
                original_output.mkdir(parents=True, exist_ok=True)
            
            stat_file_path = original_output / (name + '_stat.txt')
            
            tbr = 1000
            # Only run encoding if stat file doesn't exist
            if not stat_file_path.exists():
                tbr, stat = test_for_tbr(
                    SVT_path=str(args.SVTAV1_path),
                    local_video_path=str(local_video_path),
                    output_path=original_output,
                    name=name,
                    stat_file_path=stat_file_path,
                    tbr=tbr
                )
                print("The final tbr is:", tbr)
                all_video_stats[name] = stat
            else:
                print(f"Stat file already exists: {stat_file_path}")
                # extract stats from existing file
                stat = extract_stats_from_file(stat_file_path)
                all_video_stats[name] = stat
                tbr = stat['bitrate_kbps'] if stat else None
                print("The existing tbr is:", tbr)
            print(f"Stats for {name}: {stat}")
            # round tbr to nearest 100
            if tbr is not None:
                tbr = round(tbr / 100) * 100
            print(f"Using tbr: {tbr} for training.")
            
            if tbr is None:
                print(f"Could not determine tbr for video {name}, skipping training.")
                continue
            command = f"python av1env-training/src/train_refactor.py \
                --file {local_video_path} \
                --output_dir {video_output_dir} \
                --total_iteration 100 \
                --wandb True \
                --batch_size 32 \
                --video_name {name} \
                --tbr {tbr}"
            print(f"Running training command: {command}")
            os.system(command)