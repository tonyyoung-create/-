param(
    [string]$RemoteUrl = 'https://github.com/tonyyoung-create/-.git',
    [string]$Branch = 'main'
)

# Usage: set environment variable GITHUB_TOKEN and run this script in project folder
if (-not $env:GITHUB_TOKEN) {
    Write-Error "請先設定環境變數 GITHUB_TOKEN，內容為你在 GitHub 建立的 Personal Access Token。"
    exit 1
}

$token = $env:GITHUB_TOKEN
$authUrl = $RemoteUrl -replace 'https://', "https://$($token)@"

git remote remove origin -ErrorAction SilentlyContinue | Out-Null
git remote add origin $authUrl

git branch -M $Branch
git push -u origin $Branch

Write-Host "推送完成（若沒有權限或 URL 錯誤，請檢查 GITHUB_TOKEN 與 RemoteUrl）。"
