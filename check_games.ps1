Write-Host "=== XBOX GAMES ==="
Get-ChildItem 'C:\XboxGames' -Directory -Force -ErrorAction SilentlyContinue | ForEach-Object {
    $size = (Get-ChildItem $_.FullName -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum -ErrorAction SilentlyContinue).Sum
    [PSCustomObject]@{Game=$_.Name; SizeGB=[math]::Round($size/1GB,2)}
} | Sort-Object SizeGB -Descending | Format-Table -AutoSize

Write-Host "`n=== STEAM APPS ==="
$steamPaths = @('C:\Program Files (x86)\Steam\steamapps\common', 'C:\Program Files\Steam\steamapps\common')
foreach ($sp in $steamPaths) {
    if (Test-Path $sp) {
        Get-ChildItem $sp -Directory -Force -ErrorAction SilentlyContinue | ForEach-Object {
            $size = (Get-ChildItem $_.FullName -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum -ErrorAction SilentlyContinue).Sum
            [PSCustomObject]@{Game=$_.Name; SizeGB=[math]::Round($size/1GB,2)}
        } | Sort-Object SizeGB -Descending | Format-Table -AutoSize
    }
}

Write-Host "`n=== EPIC GAMES ==="
$epicPath = 'C:\Program Files\Epic Games'
if (Test-Path $epicPath) {
    Get-ChildItem $epicPath -Directory -Force -ErrorAction SilentlyContinue | ForEach-Object {
        $size = (Get-ChildItem $_.FullName -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum -ErrorAction SilentlyContinue).Sum
        [PSCustomObject]@{Game=$_.Name; SizeGB=[math]::Round($size/1GB,2)}
    } | Sort-Object SizeGB -Descending | Format-Table -AutoSize
}
