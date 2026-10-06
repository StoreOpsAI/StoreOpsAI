# Qwen3.6-35B-A3B 를 llama-server 로 띄웁니다 (Windows PowerShell).
# 사용 예:
#   .\scripts\start_llm.ps1                      # GPU 메모리가 넉넉할 때 (24GB 이상)
#   .\scripts\start_llm.ps1 -NCpuMoe 24          # VRAM 12~16GB: 전문가(MoE) 층 일부를 CPU에 둡니다
#   .\scripts\start_llm.ps1 -NCpuMoe 99 -Ngl 0   # GPU 없이 CPU만
param([string]$LlamaDir = (Join-Path $PSScriptRoot "..\..\llama.cpp\llama.cpp"), [string]$ModelDir = "C:\models\qwen3.6-35b-a3b", [int]$NCpuMoe = 0, [int]$Ngl = 99, [int]$Ctx = 16384, [int]$Port = 8003)
$ErrorActionPreference = "Stop"

$server = Join-Path $LlamaDir "llama-server.exe"
if (-not (Test-Path $server)) { throw "llama-server.exe 를 찾지 못했습니다: $server  (가이드 3단계 참고)" }
$cudaRuntime = Join-Path $LlamaDir "cudart-llama-bin-win-cuda-12.4-x64"
if (Test-Path $cudaRuntime) { $env:PATH = "$cudaRuntime;$env:PATH" }

$gguf = Get-ChildItem -Path $ModelDir -Recurse -Filter "*.gguf" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notmatch "mmproj" } | Sort-Object Name | Select-Object -First 1
if (-not $gguf) { throw "GGUF 파일을 찾지 못했습니다: $ModelDir  (가이드 4단계 참고)" }
Write-Host "모델 파일: $($gguf.FullName)"

$serverArgs = @(
  "--model", $gguf.FullName,
  "--alias", "qwen3.6-35b-a3b",     # config.yaml 의 llm.model 과 같아야 합니다
  "--host", "127.0.0.1",
  "--port", $Port,
  "-c", $Ctx,
  "-ngl", $Ngl,
  "--jinja"                          # 도구 호출(function calling)에 필요
)
if ($NCpuMoe -gt 0) { $serverArgs += @("--n-cpu-moe", $NCpuMoe) }

Write-Host "서버 주소: http://127.0.0.1:$Port/v1  (끄려면 Ctrl+C)"
& $server @serverArgs
