$docsPath = "d:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\Chatter-Criteria\CAMP10_Chatter_detection_Methodes\MaxEnt_SPRT\MaxEnt_SPRT\docs"

$mdFiles = Get-ChildItem -Path $docsPath -Filter "*.md" -Recurse

Write-Host "Encontrados $($mdFiles.Count) archivos .md para reparar"

foreach ($file in $mdFiles) {
    $content = Get-Content -Path $file.FullName -Encoding UTF8 -Raw
    
    $content = $content -replace [char]0xE2 + [char]0x80 + [char]0x9C, [char]0x201C
    $content = $content -replace [char]0xE2 + [char]0x80 + [char]0x9D, [char]0x201D
    $content = $content -replace [char]0xE2 + [char]0x80 + [char]0x93, [char]0x2013
    $content = $content -replace [char]0xE2 + [char]0x80 + [char]0x94, [char]0x2014
    $content = $content -replace [char]0xC3 + [char]0x97, [char]0x00D7
    $content = $content -replace [char]0xE2 + [char]0x86 + [char]0x92, [char]0x2192
    $content = $content -replace [char]0xE2 + [char]0x89 + [char]0xA5, [char]0x2265
    
    [System.IO.File]::WriteAllText($file.FullName, $content, [System.Text.Encoding]::UTF8)
    
    Write-Host "Reparado: $($file.Name)"
}

Write-Host "Listo: todos los archivos .md en UTF-8"
