$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$packageRoot = Join-Path $projectRoot "drive_package"
$packageArtifacts = Join-Path $packageRoot "artifacts"
$packageModels = Join-Path $packageArtifacts "trained_models"
$zipPath = Join-Path $projectRoot "CreditMind_Drive_Package.zip"

$resolvedProjectRoot = [System.IO.Path]::GetFullPath($projectRoot)
$resolvedPackageRoot = [System.IO.Path]::GetFullPath($packageRoot)

if (-not $resolvedPackageRoot.StartsWith($resolvedProjectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "La carpeta temporal debe permanecer dentro del proyecto."
}

if (Test-Path $packageRoot) {
    Remove-Item -LiteralPath $packageRoot -Recurse -Force
}

New-Item -ItemType Directory -Path $packageModels -Force | Out-Null

$datasetCandidates = @(
    (Join-Path $projectRoot "Loan_default_limpio.csv"),
    (Join-Path $projectRoot "data\Loan_default_limpio.csv"),
    (Join-Path $projectRoot "artifacts\Loan_default_limpio.csv")
)

$dataset = $datasetCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $dataset) {
    throw "No se encontro Loan_default_limpio.csv en la raiz, data/ o artifacts/."
}

Copy-Item -LiteralPath $dataset -Destination (Join-Path $packageRoot "Loan_default_limpio.csv")

$requiredFiles = @(
    "artifacts\features.json",
    "artifacts\model_registry_notebook.json",
    "artifacts\trained_models\logistic_regression.pkl",
    "artifacts\trained_models\random_forest.pkl",
    "artifacts\trained_models\xgboost.pkl",
    "artifacts\trained_models\neural_network.pkl"
)

foreach ($relativePath in $requiredFiles) {
    $source = Join-Path $projectRoot $relativePath
    if (-not (Test-Path $source)) {
        throw "Falta el archivo requerido: $relativePath"
    }

    if ($relativePath -like "artifacts\trained_models\*") {
        Copy-Item -LiteralPath $source -Destination $packageModels
    }
    else {
        Copy-Item -LiteralPath $source -Destination $packageArtifacts
    }
}

$instructions = @"
CreditMind - Archivos pesados

1. Extrae este ZIP.
2. Copia Loan_default_limpio.csv a la raiz del repositorio CreditMind.
3. Copia la carpeta artifacts sobre la carpeta artifacts del repositorio.
4. Confirma que artifacts/trained_models contenga los cuatro archivos .pkl.
5. Sigue las instrucciones del README.md del repositorio.
"@

Set-Content -LiteralPath (Join-Path $packageRoot "INSTRUCCIONES.txt") -Value $instructions -Encoding UTF8

if (Test-Path $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}

Compress-Archive -Path (Join-Path $packageRoot "*") -DestinationPath $zipPath -CompressionLevel Optimal
Remove-Item -LiteralPath $packageRoot -Recurse -Force

Write-Host "Paquete creado:"
Write-Host $zipPath
