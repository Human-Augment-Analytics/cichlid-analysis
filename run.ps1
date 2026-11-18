<#
.SYNOPSIS
    Full circling pipeline: data preparation, training, prediction, evaluation and report figures
    for every model, top to bottom.

.EXAMPLE
    .\run.ps1                                   # expanded run: every video in the ground truth, windowed features
    .\run.ps1 -NumVideos 8 -GroundTruth D:\gt\merged_spans_8videos.csv
    .\run.ps1 -Videos 0028_vid,0031_vid,0040_vid -Epochs 50
    .\run.ps1 -NumVideos 8 -SkipVideoMAE -SkipBaselines
    .\run.ps1 -Labels baseline                  # added models on the baselines' 15 spans, 0028/0031
    .\run.ps1 -Fresh                            # redo every step instead of resuming
    .\run.ps1 -MinSeconds 2 -MergeSeconds 3     # keep short events, join pieces split by gaps < 3 s
    .\run.ps1 -MinSeconds 4 -MergeSeconds 1 -HardNegatives 0.5   # half the negatives: close, not circling

    Re-running with the same -Labels, -GroundTruth (same file contents), videos and -DataDir resumes:
    steps that already finished are skipped (progress in outputs\runs\<run>\progress.json). Changing
    -Epochs retrains the MS-TCN; changing -Threshold/-MinSeconds/-MinFraction needs no retraining
    (predict steps always re-run; they take seconds).
    Each combination of those settings keeps its own results in
    outputs\runs\<run>\results_thr<T>_min<S>s_frac<F>[_merge<M>s]_ep<E>\ (metrics, figures, predictions, model
    reports), so earlier results are not overwritten.
#>
[CmdletBinding()]
param(
    # Ground-truth spans (VideoID, StartFrame, EndFrame): the evaluation reference for every model and,
    # with -Labels ground_truth, the training labels of the added models.
    [string]$GroundTruth = "ground_truth_annotations\finalized_annotations\merged_spans.csv",
    # First N video IDs of the ground truth, sorted; 0 = all (the expanded-dataset default).
    # Ignored when -Videos is set.
    [int]$NumVideos = 0,
    # Explicit video IDs (e.g. 0028_vid,0031_vid). Overrides -NumVideos.
    [string[]]$Videos = @(),
    # Training labels of the added models:
    #   ground_truth  the -GroundTruth file on the selected videos (model names as usual)
    #   baseline      the logistic regression's 15 hard-coded spans on 0028_vid/0031_vid, so the added models
    #                 train like logistic_regression and rule_based; names get the suffix _15spans and the
    #                 report compares only these with the two baselines. -NumVideos/-Videos are ignored.
    [ValidateSet("ground_truth", "baseline")]
    [string]$Labels = "ground_truth",
    # Folder with <video>.parquet, <video>_pairs.parquet and <video>.mp4. Default: .\data, else ..\data.
    [string]$DataDir = "",
    # Python of the environment with CUDA torch, dlc2action, xgboost and transformers.
    # Default: the uv environment .venv, else .venv-mstcn.
    [string]$Python = "",
    # uv sync, then reinstall CUDA torch from -TorchIndex (uv sync installs the CPU wheel on Windows).
    [switch]$Setup,
    [string]$TorchIndex = "https://download.pytorch.org/whl/cu121",

    [int]$Epochs = 100,
    # ActionFormer uses native temporal proposals and a separate epoch/confidence setting.
    [int]$ActionFormerEpochs = 60,
    # Native confidence/duration default to training-only validation selection.
    # Explicit values override the saved settings.
    [double]$ActionFormerMinScore = 0.05,
    [double]$ActionFormerMinSeconds = 0,
    # Windowed pair-motion variants are the expanded-dataset defaults. Original
    # non-baseline feature sets remain available for ablations.
    [string[]]$MstcnFeatures = @("pairmotion", "pairmotionvmae"),
    [string[]]$FrameFeatures = @("pairmotion", "pairmotionvmae"),
    [double]$Threshold = 0.5,
    [double]$MinSeconds = 7.5,
    [double]$MinFraction = 0.9,
    # Join predicted spans separated by gaps shorter than this many seconds (0 = off). Applies to the added
    # models only; logistic_regression keeps its own span rule.
    [double]$MergeSeconds = 0,
    # Moving average of the per-frame scores (seconds) before the span rule, added models only (0 = off).
    # Fills short dips inside a bout without joining separate spans.
    [double]$SmoothSeconds = 0,
    # Share (0-1) of the negative training segments that are hard negatives: male and partner within 160 px
    # for at least half the segment but no circling. Retrains every added model; names get the suffix _hn<percent>.
    [double]$HardNegatives = 0,

    [switch]$SkipMstcn,
    [switch]$SkipActionFormer,
    [switch]$SkipXgboost,
    [switch]$SkipLrcv,
    [switch]$SkipSartab,
    # Disable video-level out-of-fold postprocessing selection for XGBoost/LRCV.
    [switch]$SkipPostprocessTuning,
    [switch]$SkipVideoMAE,
    [switch]$SkipDinov2,
    [switch]$SkipBaselines,
    [switch]$SkipReport,
    # Ignore saved progress and redo every step (also retrains MS-TCN episodes that are already trained).
    [switch]$Fresh
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = $Root

function Resolve-Full([string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) { return [System.IO.Path]::GetFullPath($Path) }
    return [System.IO.Path]::GetFullPath((Join-Path $Root $Path))
}

$script:Done = @{}

function Save-Progress {
    @($script:Done.Keys | Sort-Object) | ConvertTo-Json | Set-Content $script:ProgressFile -Encoding UTF8
}

function Step([string]$Title, [scriptblock]$Body, [string]$Key = $Title, [switch]$Always) {
    if (-not $Always -and $script:Done.ContainsKey($Key)) {
        Write-Host ("[skip] already done: {0}" -f $Title) -ForegroundColor DarkYellow
        return $false
    }
    $started = Get-Date
    Write-Host ""
    Write-Host ("=" * 100) -ForegroundColor Cyan
    Write-Host ("[{0:HH:mm:ss}] {1}" -f $started, $Title) -ForegroundColor Cyan
    Write-Host ("=" * 100) -ForegroundColor Cyan
    & $Body | Out-Host
    if (-not $Always) {
        $script:Done[$Key] = $true
        Save-Progress
    }
    Write-Host ("[{0:HH:mm:ss}] done in {1:hh\:mm\:ss}: {2}" -f (Get-Date), ((Get-Date) - $started), $Title) -ForegroundColor Green
    return $true
}

function Py {
    Write-Host "> python $($args -join ' ')" -ForegroundColor DarkGray
    & $script:PythonExe @args
    if ($LASTEXITCODE -ne 0) { throw "python $($args -join ' ') failed with exit code $LASTEXITCODE" }
}

$NameOf = @{
    mstcn   = @{ pairemb = "mstcn_dinov2"; pairvmae = "mstcn_videomae"; pair = "mstcn_pose_pair"; pose = "mstcn_pose"
                 pairmotion = "mstcn_pairmotion"; pairmotionemb = "mstcn_pairmotion_dinov2"
                 pairmotionvmae = "mstcn_pairmotion_videomae" }
    actionformer = @{ pairmotion = "actionformer_pairmotion" }
    xgboost = @{ pair = "xgboost_pose_pair"; pairemb = "xgboost_dinov2"; pairvmae = "xgboost_videomae"
                 pairturns = "xgboost_turns"; pairturnsvmae = "xgboost_turns_videomae"
                 pairmotion = "xgboost_pairmotion"; pairmotionemb = "xgboost_pairmotion_dinov2"
                 pairmotionvmae = "xgboost_pairmotion_videomae" }
    lrcv    = @{ pair = "lrcv_pose_pair"; pairemb = "lrcv_dinov2"; pairvmae = "lrcv_videomae"
                 pairturns = "lrcv_turns"; pairturnsvmae = "lrcv_turns_videomae"
                 pairmotion = "lrcv_pairmotion"; pairmotionemb = "lrcv_pairmotion_dinov2"
                 pairmotionvmae = "lrcv_pairmotion_videomae" }
}

# ------------------------------------------------------------------------------------------- settings
$GroundTruth = Resolve-Full $GroundTruth
if (-not (Test-Path $GroundTruth)) { throw "Ground truth not found: $GroundTruth" }

if (-not $DataDir) {
    $DataDir = @((Join-Path $Root "data"), (Join-Path (Split-Path $Root -Parent) "data")) |
        Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $DataDir) { throw "No data folder found: pass -DataDir" }
}
$DataDir = Resolve-Full $DataDir

if (-not $Python) {
    $Python = @((Join-Path $Root ".venv\Scripts\python.exe"), (Join-Path $Root ".venv-mstcn\Scripts\python.exe")) |
        Where-Object { Test-Path $_ } | Select-Object -First 1
}

if ($SkipDinov2) {
    $MstcnFeatures = @($MstcnFeatures | Where-Object { $_ -notlike "*emb*" })
    $FrameFeatures = @($FrameFeatures | Where-Object { $_ -notlike "*emb*" })
}
if ($SkipVideoMAE) {
    $MstcnFeatures = @($MstcnFeatures | Where-Object { $_ -notlike "*vmae*" })
    $FrameFeatures = @($FrameFeatures | Where-Object { $_ -notlike "*vmae*" })
}

# ------------------------------------------------------------------------------------------- setup
if ($Setup) {
    $null = Step "Setup: uv sync and CUDA torch" -Always {
        uv sync
        if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }
        uv pip install --reinstall torch torchvision torchaudio --index-url $TorchIndex
        if ($LASTEXITCODE -ne 0) { throw "CUDA torch install failed" }
    }
    if (-not $Python) { $Python = Join-Path $Root ".venv\Scripts\python.exe" }
}
if (-not $Python -or -not (Test-Path $Python)) {
    throw "No Python environment found. Run with -Setup, or pass -Python <path to python.exe>."
}
$script:PythonExe = $Python

# ------------------------------------------------------------------------------------------- videos
$rows = Import-Csv $GroundTruth
if (-not $rows) { throw "$GroundTruth has no rows" }
$videoColumn = $rows[0].PSObject.Properties.Name |
    Where-Object { ($_ -replace '[^a-zA-Z]', '').ToLower() -eq "videoid" } | Select-Object -First 1
if (-not $videoColumn) { throw "$GroundTruth has no VideoID column" }
$available = @($rows | ForEach-Object { $_.$videoColumn } | Where-Object { $_ } | Sort-Object -Unique)

if ($Labels -eq "baseline") {
    $Selected = @("0028_vid", "0031_vid")
    if ($Videos.Count -gt 0 -or $PSBoundParameters.ContainsKey("NumVideos")) {
        Write-Warning "-Labels baseline uses 0028_vid and 0031_vid only; ignoring -NumVideos/-Videos"
    }
} elseif ($Videos.Count -gt 0) {
    $Selected = @($Videos | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() } | Where-Object { $_ } |
        Sort-Object -Unique)
    $notInGt = @($Selected | Where-Object { $available -notcontains $_ })
    if ($notInGt.Count) { Write-Warning "Not in the ground truth (all their frames count as negative): $($notInGt -join ', ')" }
} else {
    if ($NumVideos -lt 0) { throw "-NumVideos must be 0 (all) or a positive integer" }
    if ($NumVideos -gt $available.Count) {
        throw "-NumVideos $NumVideos but the ground truth has only $($available.Count) videos: $($available -join ', ')"
    }
    $Selected = $(if ($NumVideos -eq 0) { @($available) } else { @($available | Select-Object -First $NumVideos) })
}

$needMovies = (-not $SkipMstcn -and ($MstcnFeatures | Where-Object { $_ -like "*emb*" -or $_ -like "*vmae*" })) -or
              ((-not $SkipXgboost -or -not $SkipLrcv) -and
               ($FrameFeatures | Where-Object { $_ -like "*emb*" -or $_ -like "*vmae*" }))
$missing = @()
foreach ($video in $Selected) {
    $files = @("$video.parquet", "${video}_pairs.parquet")
    if ($needMovies) { $files += "$video.mp4" }
    foreach ($file in $files) {
        if (-not (Test-Path (Join-Path $DataDir $file))) { $missing += (Join-Path $DataDir $file) }
    }
}
if ($missing.Count) { throw "Missing input files:`n  $($missing -join "`n  ")" }

$env:CIRCLING_GROUND_TRUTH = $GroundTruth
$env:CIRCLING_DATA_DIR = $DataDir
$env:CIRCLING_VIDEOS = $Selected -join ","
$env:CIRCLING_LABELS = $Labels
if ($HardNegatives -lt 0 -or $HardNegatives -gt 1) { throw "-HardNegatives must be between 0 and 1" }
if ($ActionFormerEpochs -lt 1) { throw "-ActionFormerEpochs must be positive" }
if ($ActionFormerMinScore -lt 0 -or $ActionFormerMinScore -gt 1) { throw "-ActionFormerMinScore must be between 0 and 1" }
if ($ActionFormerMinSeconds -lt 0) { throw "-ActionFormerMinSeconds must be nonnegative" }
$env:CIRCLING_HARD_NEGATIVES = "$HardNegatives"
$Suffix = $(if ($Labels -eq "baseline") { "_15spans" } else { "" }) +
          $(if ($HardNegatives -gt 0) { "_hn{0}" -f [math]::Round($HardNegatives * 100) } else { "" })

$gtHash = (Get-FileHash -Algorithm SHA256 $GroundTruth).Hash
$runKey = "$Labels|$gtHash|$($Selected -join ',')|$DataDir|hn=$HardNegatives|mstcn=$($MstcnFeatures -join ',')|frame=$($FrameFeatures -join ',')|tunePP=$(-not $SkipPostprocessTuning)"
$sha = [System.Security.Cryptography.SHA1]::Create()
$runHash = (($sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($runKey)) | ForEach-Object { $_.ToString("x2") }) -join "").Substring(0, 8)
$RunName = "{0}{1}videos{2}_{3}" -f $(if ($Labels -eq "baseline") { "baseline_" } else { "" }), $Selected.Count, $(if ($HardNegatives -gt 0) { "_hn{0}" -f [math]::Round($HardNegatives * 100) } else { "" }), $runHash
$RunDir = Join-Path $Root "outputs\runs\$RunName"
New-Item -ItemType Directory -Force $RunDir | Out-Null
$script:ProgressFile = Join-Path $RunDir "progress.json"
if ($Fresh -and (Test-Path $script:ProgressFile)) { Remove-Item $script:ProgressFile }
if (Test-Path $script:ProgressFile) {
    foreach ($key in @(Get-Content $script:ProgressFile -Raw | ConvertFrom-Json)) { if ($key) { $script:Done[$key] = $true } }
}
Start-Transcript -Path (Join-Path $RunDir "run.log") -Append | Out-Null

try {
    Write-Host "Ground truth : $GroundTruth"
    Write-Host "Videos       : $($Selected -join ', ') ($($Selected.Count) of $($available.Count) in the ground truth)"
    Write-Host "Labels       : $(if ($Labels -eq 'baseline') { 'baseline: the 15 hard-coded spans of the logistic regression (names end in _15spans)' } else { "ground truth file" })"
    Write-Host "Hard negatives: $(if ($HardNegatives -gt 0) { "{0:P0} of the negative segments (names end in $Suffix)" -f $HardNegatives } else { 'off' })"
    Write-Host "Data         : $DataDir"
    Write-Host "Python       : $Python"
    Write-Host "MS-TCN       : $(if ($SkipMstcn) { 'skipped' } else { $MstcnFeatures -join ', ' }) ($Epochs epochs)"
    Write-Host "ActionFormer : $(if ($SkipActionFormer) { 'skipped' } else { "pairmotion (up to $ActionFormerEpochs epochs, validation-selected confidence/duration unless explicitly overridden)" })"
    Write-Host "Frame models : $FrameFeatures (xgboost: $(-not $SkipXgboost), lrcv: $(-not $SkipLrcv)); postprocess: $(if ($SkipPostprocessTuning) { 'manual' } else { 'video-level OOF selection' })"
    Write-Host "Run folder   : $RunDir"
    Write-Host "Resume       : $(if ($Fresh) { 'no (-Fresh)' } else { "$($script:Done.Count) step(s) already done will be skipped" })"

    $null = Step "Environment check" -Always {
        Py -c "import torch, sys; print('python', sys.version.split()[0], '| torch', torch.__version__, '| CUDA', torch.cuda.is_available()); torch.cuda.is_available() or print('WARNING: no CUDA GPU, MS-TCN, ActionFormer and embeddings will be very slow')"
        Py -c "import dlc2action, xgboost, sklearn, polars, marimo; print('dlc2action', dlc2action.__version__ if hasattr(dlc2action, '__version__') else 'ok', '| xgboost', xgboost.__version__)"
        if (-not $SkipVideoMAE -and ($MstcnFeatures + $FrameFeatures | Where-Object { $_ -like "*vmae*" })) {
            Py -c "import transformers; print('transformers', transformers.__version__)"
        }
        Py -m models.common.config
    }

    # ---------------------------------------------------------------------------------- 1. data preparation
    $prepare = @("pair")
    if (-not $SkipMstcn) { $prepare += $MstcnFeatures }
    if (-not $SkipXgboost -or -not $SkipLrcv) {
        $prepare += @($FrameFeatures | ForEach-Object {
            if ($_ -eq "pairturnsvmae") { "pairvmae" } elseif ($_ -eq "pairturns") { "pair" } else { $_ }
        })
    }
    $prepare = @($prepare | Where-Object { $_ -ne "pose" } | Sort-Object -Unique)
    if (-not $SkipMstcn -or -not $SkipXgboost -or -not $SkipLrcv -or -not $SkipSartab -or -not $SkipActionFormer) {
        if (-not $SkipSartab -or -not $SkipActionFormer) { $prepare = @($prepare + "pairmotion" | Sort-Object -Unique) }
        foreach ($feature in $prepare) {
            $null = Step "Prepare data: $feature (pose H5, pair features, embeddings)" -Key "prepare $feature" {
                Py -m models.mstcn_dlc2action.train --prepare-only --features $feature
            }
        }
    }

    $predictFlags = @("--threshold", "$Threshold", "--min-seconds", "$MinSeconds", "--min-fraction", "$MinFraction")
    $newModelFlags = $predictFlags + @("--merge-seconds", "$MergeSeconds", "--smooth-seconds", "$SmoothSeconds")
    $tunedFrameFlags = @()
    $postprocessTrainFlag = @(if ($SkipPostprocessTuning) { "--skip-postprocess-tuning" })
    $actionFormerFlags = @("--merge-seconds", "$MergeSeconds")
    if ($PSBoundParameters.ContainsKey("ActionFormerMinScore")) {
        $actionFormerFlags += @("--min-score", "$ActionFormerMinScore")
    }
    if ($PSBoundParameters.ContainsKey("ActionFormerMinSeconds")) {
        $actionFormerFlags += @("--min-seconds", "$ActionFormerMinSeconds")
    }
    if ($SkipPostprocessTuning) {
        $tunedFrameFlags = $newModelFlags
    } else {
        # Only explicit user settings override the saved out-of-fold selection.
        foreach ($setting in @(
            @{ Name = "Threshold"; Flag = "--threshold"; Value = $Threshold },
            @{ Name = "MinSeconds"; Flag = "--min-seconds"; Value = $MinSeconds },
            @{ Name = "MinFraction"; Flag = "--min-fraction"; Value = $MinFraction },
            @{ Name = "MergeSeconds"; Flag = "--merge-seconds"; Value = $MergeSeconds },
            @{ Name = "SmoothSeconds"; Flag = "--smooth-seconds"; Value = $SmoothSeconds }
        )) {
            if ($PSBoundParameters.ContainsKey($setting.Name)) {
                $tunedFrameFlags += @($setting.Flag, "$($setting.Value)")
            }
        }
    }
    $mergeTag = $(if ($SmoothSeconds -gt 0) { "_smooth{0}s" -f $SmoothSeconds } else { "" }) +
                $(if ($MergeSeconds -gt 0) { "_merge{0}s" -f $MergeSeconds } else { "" })
    $ResultsDir = Join-Path $RunDir ("results_thr{0}_min{1}s_frac{2}{3}_ep{4}_af{5}{6}" -f $Threshold, $MinSeconds, $MinFraction, $mergeTag, $Epochs, $ActionFormerEpochs, $(if ($SkipPostprocessTuning) { "_manualpp" } else { "_autopp" }))

    # ---------------------------------------------------------------------------------- 2. MS-TCN
    if (-not $SkipMstcn) {
        foreach ($feature in $MstcnFeatures) {
            $freshFlag = @(if ($Fresh) { "--fresh" })
            $null = Step "MS-TCN $feature -> $($NameOf.mstcn[$feature])${Suffix}: train" -Key "mstcn train $feature epochs=$Epochs" {
                Py -m models.mstcn_dlc2action.train --features $feature --num-epochs $Epochs @freshFlag
            }
            $null = Step "MS-TCN $feature -> $($NameOf.mstcn[$feature])${Suffix}: predict" -Always {
                Py -m models.mstcn_dlc2action.predict --features $feature @newModelFlags
            }
        }
    }

    # ---------------------------------------------------------------------------------- 3. per-frame models
    if (-not $SkipXgboost) {
        foreach ($feature in $FrameFeatures) {
            $null = Step "XGBoost $feature -> $($NameOf.xgboost[$feature])${Suffix}: train" -Key "xgboost train $feature" {
                Py -m models.xgboost_frame.train --features $feature @postprocessTrainFlag
            }
            $null = Step "XGBoost $feature -> $($NameOf.xgboost[$feature])${Suffix}: predict" -Always {
                Py -m models.xgboost_frame.predict --features $feature @tunedFrameFlags
            }
        }
    }
    if (-not $SkipLrcv) {
        foreach ($feature in $FrameFeatures) {
            $null = Step "LogisticRegressionCV $feature -> $($NameOf.lrcv[$feature])${Suffix}: train" -Key "lrcv train $feature" {
                Py -m models.logistic_regression_cv.train --features $feature @postprocessTrainFlag
            }
            $null = Step "LogisticRegressionCV $feature -> $($NameOf.lrcv[$feature])${Suffix}: predict" -Always {
                Py -m models.logistic_regression_cv.predict --features $feature @tunedFrameFlags
            }
        }
    }

    # ---------------------------------------------------------------------------------- 4. ActionFormer (native temporal intervals)
    if (-not $SkipActionFormer) {
        $null = Step "ActionFormer pairmotion -> $($NameOf.actionformer.pairmotion)${Suffix}: train" -Key "actionformer train pairmotion epochs=$ActionFormerEpochs" {
            Py -m models.actionformer.train --features pairmotion --epochs $ActionFormerEpochs
        }
        $null = Step "ActionFormer pairmotion -> $($NameOf.actionformer.pairmotion)${Suffix}: predict" -Always {
            Py -m models.actionformer.predict --features pairmotion @actionFormerFlags
        }
    }

    # ---------------------------------------------------------------------------------- 5. SARTAB
    if (-not $SkipSartab) {
        $null = Step "SARTAB: train" -Key "sartab train" { Py -m models.sartab.train }
        $null = Step "SARTAB: predict" -Always { Py -m models.sartab.predict @newModelFlags }
    }

    # ---------------------------------------------------------------------------------- 6. existing baselines
    $baselineVideos = @($Selected | Where-Object { $_ -in @("0028_vid", "0031_vid") })
    if (-not $SkipBaselines -and $baselineVideos.Count) {
        $null = Step "Logistic regression (unchanged: 15 spans, 0028/0031): train" -Key "logistic_regression train" {
            Py -m models.logistic_regression.train --data-dir $DataDir
        }
        $null = Step "Logistic regression: predict" -Always {
            Py -m models.logistic_regression.predict --data-dir $DataDir @predictFlags
        }
        $null = Step "Rule-based detector (unchanged: 0028/0031)" -Key "rule_based" {
            Py models\rule_based\detector.py
        }
    } elseif (-not $SkipBaselines) {
        Write-Host "Logistic regression and rule-based cover 0028_vid/0031_vid only; neither is selected, skipping."
    }

    # ---------------------------------------------------------------------------------- 7. evaluation and figures
    if (-not $SkipReport) {
        $null = Step "Feature importance (models retrained since the last run only)" -Always {
            Py -m evaluation.feature_importance
        }
        $null = Step "Evaluate every model against the ground truth" -Always {
            Write-Host "> python -m evaluation.evaluate_models" -ForegroundColor DarkGray
            New-Item -ItemType Directory -Force $ResultsDir | Out-Null
            & $script:PythonExe -m evaluation.evaluate_models | Tee-Object -FilePath (Join-Path $ResultsDir "metrics.txt")
            if ($LASTEXITCODE -ne 0) { throw "evaluation failed" }
        }
        $null = Step "mAP@IoU tables (LaTeX, Markdown, CSV, PNG) -> outputs\figures" -Always {
            Py -m evaluation.tal_report
        }
        $null = Step "Report figures and slide table -> outputs\figures" -Always {
            Py model_performance_report_viz.py
        }
    }

    # ---------------------------------------------------------------------------------- 8. snapshot
    $null = Step "Snapshot this run -> $ResultsDir" -Always {
        New-Item -ItemType Directory -Force $ResultsDir | Out-Null
        foreach ($part in @("figures", "predictions", "models")) {
            $old = Join-Path $ResultsDir $part
            if (Test-Path $old) { Remove-Item $old -Recurse -Force }
        }
        $figures = Join-Path $Root "outputs\figures"
        if (Test-Path $figures) { Copy-Item $figures (Join-Path $ResultsDir "figures") -Recurse -Force }
        Copy-Item (Join-Path $Root "outputs\predictions") (Join-Path $ResultsDir "predictions") -Recurse -Force
        $modelDirs = @("circling_baseline") + (@("sartab") + $NameOf.actionformer.Values + $NameOf.mstcn.Values + $NameOf.xgboost.Values +
            $NameOf.lrcv.Values | ForEach-Object { "$_${Suffix}" })
        foreach ($name in $modelDirs) {
            $source = Join-Path $Root "outputs\$name"
            if (-not (Test-Path $source)) { continue }
            $target = Join-Path $ResultsDir "models\$name"
            New-Item -ItemType Directory -Force $target | Out-Null
            Get-ChildItem $source -File | Where-Object { $_.Extension -in ".json", ".md", ".png", ".csv" } |
                Copy-Item -Destination $target
        }
        @{ labels = $Labels; hard_negatives = $HardNegatives; ground_truth = $GroundTruth; videos = $Selected; data_dir = $DataDir; epochs = $Epochs
           actionformer = $(if ($SkipActionFormer) { $null } else { @{ features = "pairmotion"; epochs = $ActionFormerEpochs; min_score = $ActionFormerMinScore; min_seconds = $ActionFormerMinSeconds; filters = "validation-selected unless explicitly overridden" } })
           mstcn = $(if ($SkipMstcn) { @() } else { $MstcnFeatures }); frame_features = $FrameFeatures
           postprocess_tuning = (-not $SkipPostprocessTuning)
           threshold = $Threshold; min_seconds = $MinSeconds; min_fraction = $MinFraction; merge_seconds = $MergeSeconds; smooth_seconds = $SmoothSeconds } |
            ConvertTo-Json | Set-Content (Join-Path $ResultsDir "settings.json") -Encoding UTF8
    }

    Write-Host ""
    Write-Host "Finished. Figures and slide table: outputs\figures and $ResultsDir" -ForegroundColor Green
    Write-Host "Interactive report for this selection (same shell):"
    Write-Host "  & `"$Python`" -m marimo edit model_performance_report_viz.py"
}
finally {
    Stop-Transcript | Out-Null
}
