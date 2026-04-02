Write-Host "=== TOP FOLDERS IN USER PROFILE ==="
Get-ChildItem 'C:\Users\dario' -Directory -Force -ErrorAction SilentlyContinue | ForEach-Object {
    try {
        $size = (Get-ChildItem $_.FullName -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        [PSCustomObject]@{Folder=$_.Name; SizeGB=[math]::Round($size/1GB,2)}
    } catch {}
} | Sort-Object SizeGB -Descending | Select-Object -First 15 | Format-Table -AutoSize

Write-Host "`n=== TOP FOLDERS ON C:\ ==="
Get-ChildItem 'C:\' -Directory -Force -ErrorAction SilentlyContinue | Where-Object {
    $_.Name -notin @('Windows','Users','$Recycle.Bin','System Volume Information','Recovery')
} | ForEach-Object {
    try {
        $size = (Get-ChildItem $_.FullName -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum -ErrorAction SilentlyContinue).Sum
        [PSCustomObject]@{Folder=$_.FullName; SizeGB=[math]::Round($size/1GB,2)}
    } catch {}
} | Sort-Object SizeGB -Descending | Select-Object -First 10 | Format-Table -AutoSize

Write-Host "`n=== OTHER DRIVES ==="
Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Name -ne 'C' } | ForEach-Object {
    [PSCustomObject]@{Drive=$_.Name; FreeGB=[math]::Round($_.Free/1GB,1); UsedGB=[math]::Round($_.Used/1GB,1)}
} | Format-Table -AutoSize

Write-Host "`n=== TEMP FOLDER SIZE ==="
$tempSize = (Get-ChildItem 'C:\Users\dario\AppData\Local\Temp' -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
Write-Host ("Temp: {0} GB" -f [math]::Round($tempSize/1GB,2))

Write-Host "`n=== RECYCLE BIN ==="
$shell = New-Object -ComObject Shell.Application
$rb = $shell.NameSpace(0x0a)
Write-Host ("Items in Recycle Bin: {0}" -f $rb.Items().Count)
