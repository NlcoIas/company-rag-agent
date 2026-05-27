# Pre-warm Ollama models before the demo. Loads qwen3:8b + nomic-embed-text
# into VRAM so the first evaluator visit doesn't see a 30-60s cold start.
# Run once before opening the demo URL.

param(
    [string]$OllamaHost = "http://127.0.0.1:11434",
    [string]$LlmModel = "qwen3:8b",
    [string]$EmbedModel = "nomic-embed-text"
)

$ErrorActionPreference = "Stop"

function Show-Stage {
    param([string]$Msg)
    Write-Output ("[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $Msg)
}

Show-Stage "checking ollama daemon at $OllamaHost"
try {
    $tags = Invoke-WebRequest -Uri "$OllamaHost/api/tags" -UseBasicParsing -TimeoutSec 5
    $tagsJson = $tags.Content | ConvertFrom-Json
    $names = ($tagsJson.models | ForEach-Object { $_.name })
    Show-Stage "daemon up. installed models: $($names -join ', ')"
    if ($names -notcontains "$LlmModel" -and $names -notcontains "${LlmModel}:latest") {
        Write-Error "model $LlmModel not installed — run 'ollama pull $LlmModel'"
    }
    if ($names -notcontains "$EmbedModel" -and $names -notcontains "${EmbedModel}:latest") {
        Write-Error "model $EmbedModel not installed — run 'ollama pull $EmbedModel'"
    }
} catch {
    Write-Error "ollama daemon not reachable at $OllamaHost`: $($_.Exception.Message)"
}

$t0 = Get-Date
Show-Stage "warming $EmbedModel ..."
$null = Invoke-WebRequest -Uri "$OllamaHost/api/embeddings" `
    -Method POST `
    -Body (@{model = $EmbedModel; prompt = "warmup probe"} | ConvertTo-Json) `
    -ContentType 'application/json' `
    -TimeoutSec 120 `
    -UseBasicParsing
$t1 = Get-Date
Show-Stage ("$EmbedModel warm in {0:n1}s" -f ($t1 - $t0).TotalSeconds)

Show-Stage "warming $LlmModel ..."
$null = Invoke-WebRequest -Uri "$OllamaHost/api/generate" `
    -Method POST `
    -Body (@{model = $LlmModel; prompt = "hi"; stream = $false; options = @{num_predict = 4}} | ConvertTo-Json) `
    -ContentType 'application/json' `
    -TimeoutSec 180 `
    -UseBasicParsing
$t2 = Get-Date
Show-Stage ("$LlmModel warm in {0:n1}s" -f ($t2 - $t1).TotalSeconds)

Show-Stage ("both models warm. total {0:n1}s. ready for demo." -f ($t2 - $t0).TotalSeconds)
