# Sea of Thieves (Xbox) - rimuovi via AppxPackage
Write-Host "=== Rimozione Sea of Thieves (Xbox) ==="
$pkg = Get-AppxPackage -Name "*SeaOfThieves*" -ErrorAction SilentlyContinue
if ($pkg) {
    $pkg | Remove-AppxPackage -ErrorAction SilentlyContinue
    Write-Host "AppxPackage rimosso"
}
if (Test-Path 'C:\XboxGames\Sea of Thieves') {
    Remove-Item 'C:\XboxGames\Sea of Thieves' -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "Cartella Sea of Thieves rimossa"
} else {
    Write-Host "Cartella Sea of Thieves non trovata (gia rimossa dal package manager)"
}

# Fortnite (Epic)
Write-Host "`n=== Rimozione Fortnite (Epic) ==="
if (Test-Path 'C:\Program Files\Epic Games\Fortnite') {
    Remove-Item 'C:\Program Files\Epic Games\Fortnite' -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "Cartella Fortnite rimossa"
} else {
    Write-Host "Cartella Fortnite non trovata"
}

# Rainbow Six Siege (Steam)
Write-Host "`n=== Rimozione Rainbow Six Siege (Steam) ==="
if (Test-Path "C:\Program Files (x86)\Steam\steamapps\common\Tom Clancy's Rainbow Six Siege") {
    Remove-Item "C:\Program Files (x86)\Steam\steamapps\common\Tom Clancy's Rainbow Six Siege" -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "Cartella Rainbow Six Siege rimossa"
} else {
    Write-Host "Cartella Rainbow Six Siege non trovata"
}

# Risultato finale
Write-Host "`n=== SPAZIO LIBERO ==="
$free = [math]::Round((Get-PSDrive C).Free / 1GB, 1)
Write-Host "Spazio libero su C: $free GB"
