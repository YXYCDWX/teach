# Full training chain (3 models). Each script auto-skips when done / resumes from checkpoint.
$py = "D:\software\miniconda3\envs\dl\python.exe"
$base = "D:\PythonProject\teach"
Set-Location -LiteralPath $base   # YOLO 起始权重按相对路径解析, 固定 cwd 防止走网络下载兜底

& $py "$base\train_yolo.py" yolo11;   if ($LASTEXITCODE -ne 0) { exit 1 }
& $py "$base\train_frcnn.py";         if ($LASTEXITCODE -ne 0) { exit 1 }
& $py "$base\train_3d.py";            if ($LASTEXITCODE -ne 0) { exit 1 }
Write-Host "chain done"
