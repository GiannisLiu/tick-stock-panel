<#
.SYNOPSIS
  启动通达信客户端(TdxW.exe)，幂等：已在运行就不重复启动。

.DESCRIPTION
  TSP 的 tdx 数据源插件(五档盘口 + 全量分钟)依赖本机通达信客户端暴露的行情服务
  http://127.0.0.1:17709/。Docker 里的后端够不到宿主机的 GUI 进程，所以"把客户端
  拉起来"只能放在宿主机侧：

    1) 登录自启(推荐)：
       schtasks /Create /TN "Start 通达信" /SC ONLOGON /RL LIMITED /F /TR ^
         "powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File \"C:\Projects\tick-stock-panel\scripts\start_tdx.ps1\" -WaitReady 180"
       删除：schtasks /Delete /TN "Start 通达信" /F

    2) 包在服务启动命令前面(先拉起通达信，等它就绪，再起 TSP)：
       powershell -File C:\Projects\tick-stock-panel\scripts\start_tdx.ps1 -WaitReady 180
       docker run -d --name tsp ...

.PARAMETER ExePath
  显式指定 TdxW.exe 路径。不给就按注册表四个 Uninstall 键找，再退回常见安装目录。

.PARAMETER WaitReady
  启动后最多等多少秒直到 17709 可用(默认 0 = 不等)。客户端要登录进主界面后行情服务才通。

.PARAMETER ProbeUrl
  探活地址，默认 http://127.0.0.1:17709/。后端在容器里跑时它是宿主机的 127.0.0.1。

.OUTPUTS
  退出码：0 = 已在运行 或 启动成功(且就绪) · 2 = 未在运行时找不到 TdxW.exe(已在运行则直接跳过，
  不再校验 -ExePath) · 3 = 等待就绪超时
#>
[CmdletBinding()]
param(
  [string]$ExePath = "",
  [int]$WaitReady = 0,
  [string]$ProbeUrl = "http://127.0.0.1:17709/"
)

$ErrorActionPreference = "Stop"
$ProcessName = "TdxW"

function Get-TdxProcess {
  Get-Process -Name $ProcessName -ErrorAction SilentlyContinue | Select-Object -First 1
}

function Find-TdxExe {
  if ($ExePath) {
    if (Test-Path -LiteralPath $ExePath) { return (Resolve-Path -LiteralPath $ExePath).Path }
    Write-Host "[FAIL] -ExePath 指定的文件不存在: $ExePath"
    exit 2
  }
  # 通达信多版本共存: 注册表可能只记录其中一个，所以逐个键找第一个真实存在的 exe
  $keys = @(
    'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\通达信金融终端64',
    'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\通达信专业版',
    'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\通达信金融终端(量化模拟)',
    'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\通达信金融终端(测试)'
  )
  foreach ($key in $keys) {
    if (-not (Test-Path $key)) { continue }
    $props = Get-ItemProperty -Path $key -ErrorAction SilentlyContinue
    if (-not $props) { continue }
    $candidates = New-Object System.Collections.ArrayList
    if ($props.InstallLocation) { [void]$candidates.Add($props.InstallLocation) }
    if ($props.DisplayIcon) {
      # DisplayIcon 形如 "C:\path\TdxW.exe,0"，去掉引号与图标序号后取目录
      $icon = ($props.DisplayIcon -replace '"', '') -split ',' | Select-Object -First 1
      if ($icon) { [void]$candidates.Add((Split-Path -Parent $icon)) }
    }
    foreach ($dir in $candidates) {
      if (-not $dir) { continue }
      $exe = Join-Path $dir "TdxW.exe"
      if (Test-Path -LiteralPath $exe) { return $exe }
    }
  }
  foreach ($fallback in @(
      'C:\new_tdx64\TdxW.exe', 'C:\new_tdx\TdxW.exe',
      'D:\new_tdx64\TdxW.exe', 'D:\new_tdx\TdxW.exe')) {
    if (Test-Path -LiteralPath $fallback) { return $fallback }
  }
  return $null
}

function Test-TdxReady {
  $body = '{"id":1,"method":"get_market_snapshot","params":{"stock_code":"000001.SZ"}}'
  try {
    $resp = Invoke-RestMethod -Uri $ProbeUrl -Method Post -Body $body `
      -ContentType 'application/json; charset=utf-8' -TimeoutSec 5
    return [bool]$resp.result.LastClose
  } catch {
    return $false
  }
}

function Wait-TdxReady([int]$seconds) {
  $deadline = (Get-Date).AddSeconds($seconds)
  while ((Get-Date) -lt $deadline) {
    if (Test-TdxReady) { return $true }
    Start-Sleep -Seconds 3
  }
  return $false
}

if (Get-TdxProcess) {
  Write-Host "[OK] 通达信已在运行，跳过启动"
} else {
  $exe = Find-TdxExe
  if (-not $exe) {
    Write-Host "[FAIL] 未找到 TdxW.exe，请用 -ExePath 指定安装路径"
    exit 2
  }
  Start-Process -FilePath $exe
  Write-Host "[OK] 已启动: $exe"
}

if ($WaitReady -gt 0) {
  Write-Host "等待通达信行情服务就绪(最多 $WaitReady 秒, 需登录进主界面): $ProbeUrl"
  if (-not (Wait-TdxReady $WaitReady)) {
    Write-Host "[FAIL] 等待 $WaitReady 秒后 $ProbeUrl 仍不可用(客户端可能停在登录界面)"
    exit 3
  }
  Write-Host "[OK] 行情服务已就绪"
}

exit 0
