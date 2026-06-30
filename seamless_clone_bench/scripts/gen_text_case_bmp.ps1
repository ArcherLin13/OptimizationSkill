# Generates 729x126 text-case BMPs (no OpenCV/Python required).
param([string]$OutDir = (Join-Path $PSScriptRoot "..\samples\text_case"))

$W = 729
$H = 126
$MaskRect = @(5, 5, 718, 115)
$Center = @(364, 62)

$Font = @{
    " " = @(".....",".....",".....",".....",".....",".....",".....")
    "0" = @(".###.","#...#","#..##","#.#.#","##..#","#...#",".###.")
    "1" = @("..#..",".##..","..#..","..#..","..#..","..#..",".###.")
    "2" = @(".###.","#...#","....#","...#.","..#..",".#...","#####")
    "3" = @(".###.","#...#","....#","..##.","....#","#...#",".###.")
    "4" = @("#...#","#...#","#...#","#####","....#","....#","....#")
    "5" = @("#####","#....","#....","####.","....#","#...#",".###.")
    "6" = @(".###.","#...#","#....","####.","#...#","#...#",".###.")
    "7" = @("#####","....#","...#.","..#..",".#...",".#...",".#...")
    "8" = @(".###.","#...#","#...#",".###.","#...#","#...#",".###.")
    "9" = @(".###.","#...#","#...#",".####","....#","#...#",".###.")
    "A" = @(".###.","#...#","#...#","#####","#...#","#...#","#...#")
    "B" = @("####.","#...#","#...#","####.","#...#","#...#","####.")
    "C" = @(".###.","#...#","#....","#....","#....","#...#",".###.")
    "D" = @("####.","#...#","#...#","#...#","#...#","#...#","####.")
    "E" = @("#####","#....","#....","####.","#....","#....","#####")
    "F" = @("#####","#....","#....","####.","#....","#....","#....")
    "G" = @(".###.","#...#","#....","#..##","#...#","#...#",".###.")
    "H" = @("#...#","#...#","#...#","#####","#...#","#...#","#...#")
    "I" = @(".###.","..#..","..#..","..#..","..#..","..#..",".###.")
    "J" = @("..###","...#.","...#.","...#.","...#.","#..#.",".##..")
    "K" = @("#...#","#..#.","#.#..","##...","#.#..","#..#.","#...#")
    "L" = @("#....","#....","#....","#....","#....","#....","#####")
    "M" = @("#...#","##.##","#.#.#","#.#.#","#...#","#...#","#...#")
    "N" = @("#...#","##..#","#.#.#","#..##","#...#","#...#","#...#")
    "O" = @(".###.","#...#","#...#","#...#","#...#","#...#",".###.")
    "P" = @("####.","#...#","#...#","####.","#....","#....","#....")
    "Q" = @(".###.","#...#","#...#","#...#","#.#.#","#..#.",".##.#")
    "R" = @("####.","#...#","#...#","####.","#.#..","#..#.","#...#")
    "S" = @(".####","#....","#....",".###.","....#","....#","####.")
    "T" = @("#####","..#..","..#..","..#..","..#..","..#..","..#..")
    "U" = @("#...#","#...#","#...#","#...#","#...#","#...#",".###.")
    "V" = @("#...#","#...#","#...#","#...#","#...#",".#.#.","..#..")
    "W" = @("#...#","#...#","#...#","#.#.#","#.#.#","##.##","#...#")
    "X" = @("#...#","#...#",".#.#.","..#..",".#.#.","#...#","#...#")
    "Y" = @("#...#","#...#",".#.#.","..#..","..#..","..#..","..#..")
    "Z" = @("#####","....#","...#.","..#..",".#...","#....","#####")
    ":" = @(".....","..#..","..#..",".....","..#..","..#..",".....")
    "/" = @("....#","...#.","..#..",".#...","#....",".....",".....")
}

function New-Image([byte[]]$Bgr) {
    $img = New-Object 'System.Collections.Generic.List[byte[][]]'
    for ($y = 0; $y -lt $H; $y++) {
        $row = New-Object byte[] ($W * 3)
        for ($x = 0; $x -lt $W; $x++) {
            $row[$x * 3] = $Bgr[0]
            $row[$x * 3 + 1] = $Bgr[1]
            $row[$x * 3 + 2] = $Bgr[2]
        }
        [void]$img.Add($row)
    }
    return $img
}

function Fill-Rect($img, $x, $y, $w, $h, [byte[]]$Color) {
    $y0 = [Math]::Max(0, $y)
    $y1 = [Math]::Min($H, $y + $h)
    $x0 = [Math]::Max(0, $x)
    $x1 = [Math]::Min($W, $x + $w)
    for ($yy = $y0; $yy -lt $y1; $yy++) {
        $row = $img[$yy]
        for ($xx = $x0; $xx -lt $x1; $xx++) {
            $row[$xx * 3] = $Color[0]
            $row[$xx * 3 + 1] = $Color[1]
            $row[$xx * 3 + 2] = $Color[2]
        }
    }
}

function Draw-Text($img, $x, $y, [string]$Text, [byte[]]$Color, [int]$Scale = 2) {
    $cx = $x
    foreach ($ch in $Text.ToUpper().ToCharArray()) {
        $key = [string]$ch
        if (-not $Font.ContainsKey($key)) { $key = " " }
        $glyph = $Font[$key]
        for ($row = 0; $row -lt $glyph.Count; $row++) {
            $line = $glyph[$row]
            for ($col = 0; $col -lt $line.Length; $col++) {
                if ($line[$col] -eq '#') {
                    Fill-Rect $img ($cx + $col * $Scale) ($y + $row * $Scale) $Scale $Scale $Color
                }
            }
        }
        $cx += 6 * $Scale
        $cx += 2 * $Scale
    }
}

function Write-Bmp([string]$Path, $img) {
    $rowStride = [int](([Math]::Ceiling($W * 3 / 4.0)) * 4)
    $pixelBytes = $rowStride * $H
    $fs = [System.IO.File]::Create($Path)
    try {
        $bw = New-Object System.IO.BinaryWriter($fs)
        $bw.Write([byte[]](0x42, 0x4D))
        $bw.Write([int32](54 + $pixelBytes))
        $bw.Write([int32]0)
        $bw.Write([int32]0)
        $bw.Write([int32]54)
        $bw.Write([int32]40)
        $bw.Write([int32]$W)
        $bw.Write([int32]$H)
        $bw.Write([int16]1)
        $bw.Write([int16]24)
        $bw.Write([int32]0)
        $bw.Write([int32]$pixelBytes)
        $bw.Write([int32]2835)
        $bw.Write([int32]2835)
        $bw.Write([int32]0)
        $bw.Write([int32]0)
        $padLen = $rowStride - $W * 3
        $pad = New-Object byte[] $padLen
        for ($y = $H - 1; $y -ge 0; $y--) {
            $bw.Write($img[$y])
            if ($padLen -gt 0) { $bw.Write($pad) }
        }
    } finally {
        $fs.Close()
    }
}

$out = (Resolve-Path -LiteralPath $OutDir -ErrorAction SilentlyContinue)
if (-not $out) {
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    $out = Resolve-Path -LiteralPath $OutDir
}

$dst = New-Image @(90, 40, 25)
Draw-Text $dst 16 18 "BACKGROUND 729x126" @(235, 235, 235) 2
Draw-Text $dst 16 48 "DST: ABCD EFGH IJKL MNOP QRST" @(220, 220, 220) 2
Draw-Text $dst 16 78 "harmony seamless clone test" @(200, 200, 120) 2
for ($x = 20; $x -lt ($W - 40); $x += 54) {
    Draw-Text $dst $x 100 "8gQy" @(200, 200, 120) 1
}

$src = New-Image @(210, 230, 255)
Draw-Text $src 16 18 "SOURCE PATCH CLONE" @(20, 20, 160) 2
Draw-Text $src 16 48 "SRC: 0123456789 9876543210" @(30, 30, 140) 2
Draw-Text $src 16 78 "glyph edge sharpness check" @(40, 40, 120) 2
for ($x = 20; $x -lt ($W - 40); $x += 54) {
    Draw-Text $src $x 100 "6pZj" @(180, 60, 20) 1
}

$mask = New-Image @(0, 0, 0)
Fill-Rect $mask $MaskRect[0] $MaskRect[1] $MaskRect[2] $MaskRect[3] @(0, 220, 0)

Write-Bmp (Join-Path $out "src.bmp") $src
Write-Bmp (Join-Path $out "dst.bmp") $dst
Write-Bmp (Join-Path $out "mask.bmp") $mask
Set-Content -Path (Join-Path $out "center.txt") -Value "$($Center[0]),$($Center[1])" -NoNewline -Encoding ascii
Add-Content -Path (Join-Path $out "center.txt") -Value "" -Encoding ascii

Write-Host "Wrote text case BMPs to: $out"
Write-Host "  src.bmp dst.bmp mask.bmp center.txt"
