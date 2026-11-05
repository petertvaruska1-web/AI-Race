# AI Race helper setup. It lets this computer train and run AIs for an AI Race game.
# The game's setup file (setup.cmd) runs it with the game's address and the player's own key
# in AIRACE_URL and AIRACE_KEY. Everything goes to %LOCALAPPDATA%\AIRaceHelper.

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$GameUrl = $env:AIRACE_URL.TrimEnd("/")
$Key = $env:AIRACE_KEY
$HelperHome = Join-Path $env:LOCALAPPDATA "AIRaceHelper"
$Ml = Join-Path $HelperHome "ml"
$Data = Join-Path $HelperHome "data"
$Headers = @{ Authorization = "Bearer $Key" }

function Say([string]$Text) {
    Write-Host ""
    Write-Host $Text -ForegroundColor Cyan
}

try {
    Say "Setting up the AI Race helper. The first time takes about 15 minutes."
    New-Item -ItemType Directory -Force $HelperHome | Out-Null

    # 1. uv installs Python and the AI libraries for the helper.
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        Say "Step 1 of 5: uv is already installed."
    } else {
        Say "Step 1 of 5: installing uv (it installs Python for the helper)..."
        powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
        $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    }
    $Uv = (Get-Command uv).Source

    # 2. The helper's code, from the game.
    Say "Step 2 of 5: downloading the helper from the game..."
    $Zip = Join-Path $HelperHome "ml.zip"
    Invoke-WebRequest -Uri "$GameUrl/helper/ml.zip" -Headers $Headers -OutFile $Zip -UseBasicParsing
    # The code is replaced; the installed libraries (.venv) stay, so an update is quick.
    if (Test-Path (Join-Path $Ml "src")) { Remove-Item -Recurse -Force (Join-Path $Ml "src") }
    Expand-Archive -Force $Zip $Ml
    Remove-Item -Force $Zip

    # 3. Python and the AI libraries.
    Say "Step 3 of 5: installing Python and the AI libraries (about 3 GB the first time)..."
    Push-Location $Ml
    & $Uv sync
    $synced = $LASTEXITCODE
    Pop-Location
    if ($synced -ne 0) { throw "Python and the AI libraries could not be installed." }

    # 4. The game's data (the same for every player in the game).
    Say "Step 4 of 5: downloading the game's data (about 1.1 GB the first time)..."
    & $Uv run --project $Ml airace-ml fetch-data --server $GameUrl --key $Key --data-root $Data
    if ($LASTEXITCODE -ne 0) { throw "The game's data could not be downloaded." }

    # 5. "Connect your computer" in the game opens an airace-helper:// link that starts the helper.
    Say "Step 5 of 5: letting the game start the helper..."
    $Run = "`"$Uv`" run --project `"$Ml`" airace-ml worker --server `"%1`" --data-root `"$Data`" --device auto"
    $Base = "HKCU:\Software\Classes\airace-helper"
    New-Item -Force $Base | Out-Null
    Set-ItemProperty $Base "(Default)" "URL:AI Race helper"
    Set-ItemProperty $Base "URL Protocol" ""
    New-Item -Force "$Base\shell\open\command" | Out-Null
    Set-ItemProperty "$Base\shell\open\command" "(Default)" $Run

    # A Start-menu shortcut that asks for today's game link (the link changes every session).
    Set-Content -Encoding UTF8 (Join-Path $HelperHome "key.txt") $Key
    $Start = Join-Path $HelperHome "start.ps1"
    @"
`$address = Read-Host "Paste the game's address (your friend sends it, it starts with https://)"
`$key = (Get-Content -Raw "$HelperHome\key.txt").Trim()
& "$Uv" run --project "$Ml" airace-ml worker --server "airace-helper://connect?url=`$(`$address.Trim())&key=`$key" --data-root "$Data" --device auto
"@ | Set-Content -Encoding UTF8 $Start
    $Shell = New-Object -ComObject WScript.Shell
    $Programs = [Environment]::GetFolderPath("Programs")
    $Link = $Shell.CreateShortcut((Join-Path $Programs "AI Race Helper.lnk"))
    $Link.TargetPath = "powershell.exe"
    $Link.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$Start`""
    $Link.WorkingDirectory = $HelperHome
    $Link.Save()

    Say "All done. In the game, press 'Connect your computer' to start the helper."
} catch {
    Write-Host ""
    Write-Host "The setup stopped: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Run the setup file again; it carries on where it stopped." -ForegroundColor Red
    exit 1
}
