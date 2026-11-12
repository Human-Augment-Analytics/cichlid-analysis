# Training on PACE ICE (GPU)

ICE is only reachable on the GT VPN (<https://vpn.gatech.edu>). The quotas below come from
community guides and change over time, so check yours with `pace-quota`.

| Storage | Size | Use for |
|---|---|---|
| `~` (home) | ~15 GB | nothing large |
| `~/scratch` | ~300 GB, **wiped after the semester** | repo, `.venv`, videos, outputs |
| `$TMPDIR` | ~1 TB, deleted when the job ends | fast per-job copies |

## 1. Set up once (login node)

```sh
ssh <gtuser>@login-ice.pace.gatech.edu
cd ~/scratch
git clone <repo-url> cichlid-analysis && cd cichlid-analysis
# Keep caches out of the 15 GB home
mkdir -p ~/scratch/.cache && rm -rf ~/.cache && ln -s ~/scratch/.cache ~/.cache
module load uv        # or: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --locked
```

## 2. Get the videos to ICE (~33 GB each, ~100 GB total)

**Pull them on ICE instead of uploading them from your laptop.** Campus bandwidth is much
faster than a home uplink. `download_data.py` fetches every link in `video_source.py`,
downloading many pieces of each file at once, and resumes where it stopped if you rerun it:

```sh
cd ~/scratch/cichlid-analysis
tmux new -s dl                                 # survives an SSH disconnect
python3 download_data.py --tables              # videos + parquet tables into data/
python3 download_data.py --urls extra.txt      # more files: "<name> <dropbox url ending dl=1>" per line
```

`0018_vid.mp4` and `0018_vid.parquet` have no URL yet. Either add them to the shared
Dropbox folder (then to `video_source.py`), or push them from a machine on campus or the VPN:

```sh
rsync -a --partial --progress data/0018_vid.* <gtuser>@login-ice.pace.gatech.edu:~/scratch/cichlid-analysis/data/
```

**Tips:**

- Models that only need features can train from the `.parquet` files (~200 MB each), so
  only download the `.mp4`s if you need pixels.
- Back up anything in scratch you want to keep. It is wiped at the end of the semester.

## 3. Run on a GPU

**Interactive:** use the OnDemand VS Code or Jupyter app at
<https://ondemand-ice.pace.gatech.edu> with a GPU node type, or from SSH:

```sh
salloc --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=2:00:00
nvidia-smi
```

**Batch** (`train.sbatch`):

```sh
#!/bin/bash
#SBATCH -J cichlid-train
#SBATCH --gres=gpu:1             # or a specific type, e.g. gpu:V100:1 / gpu:A100:1 / gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -o logs/%x-%j.out

cd ~/scratch/cichlid-analysis
uv run python -m models.logistic_regression.train   # replace with your training entry point
```

```sh
mkdir -p logs && sbatch train.sbatch
squeue -u $USER      # check status
scancel <jobid>      # cancel
```

**Notes:**

- Don't pass a partition. ICE picks one from your course.
- Smaller, shorter requests start sooner.
- Jobs keep running, and keep using allocation, after you close the browser. Cancel them when
  you're done.
- For heavy random-access reads of a video, copy it to `$TMPDIR` at the start of the job.
