$ErrorActionPreference = "Stop"

$DatasetDir = $PSScriptRoot
$PagesDir = Join-Path $DatasetDir "pages"

if (!(Test-Path $PagesDir)) {
  throw "Khong tim thay folder pages: $PagesDir"
}

$WordFiles = Get-ChildItem $PagesDir -File | Where-Object {
  $_.Extension.ToLowerInvariant() -in @(".doc", ".docx")
}

if (!$WordFiles) {
  Write-Output "Khong co file DOC/DOCX de convert."
  exit 0
}

$Word = New-Object -ComObject Word.Application
$Word.Visible = $false
$Word.DisplayAlerts = 0

$Converted = @()
try {
  foreach ($File in $WordFiles) {
    $OutPath = Join-Path $PagesDir ($File.BaseName + "_from_word.pdf")
    if ((Test-Path $OutPath) -and ((Get-Item $OutPath).LastWriteTime -ge $File.LastWriteTime)) {
      $Converted += $OutPath
      continue
    }

    $Doc = $null
    try {
      $Doc = $Word.Documents.Open($File.FullName, $false, $true)
      $Doc.ExportAsFixedFormat($OutPath, 17)
      $Converted += $OutPath
      Write-Output "Converted: $($File.Name) -> $(Split-Path $OutPath -Leaf)"
    } finally {
      if ($Doc -ne $null) {
        $Doc.Close($false)
      }
    }
  }
} finally {
  $Word.Quit()
}

Write-Output "Converted count: $($Converted.Count)"
