# Full training chain (6 models). Each script auto-skips when done / resumes from checkpoint.
$py = "D:\software\miniconda3\envs\dl\python.exe"
$base = "D:\PythonProject\teach"

& $py "$base\train_yolo.py" yolo11;      if ($LASTEXITCODE -ne 0) { exit 1 }
& $py "$base\train_yolo.py" yolo26;      if ($LASTEXITCODE -ne 0) { exit 1 }
& $py "$base\train_ssd.py";              if ($LASTEXITCODE -ne 0) { exit 1 }
& $py "$base\train_frcnn.py";            if ($LASTEXITCODE -ne 0) { exit 1 }
& $py "$base\train_3d.py" pointpillars;  if ($LASTEXITCODE -ne 0) { exit 1 }
& $py "$base\train_3d.py" voxelnet;      if ($LASTEXITCODE -ne 0) { exit 1 }
& $py "$base\train_rcnn3d.py";           if ($LASTEXITCODE -ne 0) { exit 1 }
& $py "$base\train_3d.py" centerpoint;   if ($LASTEXITCODE -ne 0) { exit 1 }
Write-Host "chain done"
