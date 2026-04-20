# Simple Rule-Based Flatfoot Classifier

This version is intentionally small and easy to understand.

It only does:
- thresholding
- bounding box
- 3-part row partition
- Arch Index calculation
- final classification (`flatfoot` / `not_flatfoot`)
- optional text visual for boxing and partitioning

## Run single file

```powershell
& "e:/Flat Foot/.venv/Scripts/python.exe" "Phase-2/rule_based_flatfoot_v1.py" --csv "E:/Flat Foot/Pressure_Data/subject1_normal_trial1_pressure.csv" --show-visual
```

## Run on a folder (first few files)

```powershell
& "e:/Flat Foot/.venv/Scripts/python.exe" "Phase-2/rule_based_flatfoot_v1.py" --folder "E:/Flat Foot/Pressure_Data" --limit 5
```

## Notes

- Main decision rule used:
  - `AI > 0.26` => `flatfoot`
  - else => `not_flatfoot`
- `--show-visual` prints a simple text view with:
  - bounding box border
  - partition lines between Zone 3 / Zone 2 / Zone 1
