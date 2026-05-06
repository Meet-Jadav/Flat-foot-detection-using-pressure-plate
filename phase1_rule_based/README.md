# Phase 1 - Rule Based Detector

This phase is where I tried to understand the pressure plate data before doing anything fancy. The main idea is simple: if the arch collapses, the midfoot keeps more contact with the plate, so the pressure map starts looking fuller in the middle.

What this phase computes:
- Arch Index, which is the midfoot contact area divided by total contact area
- Chippaux Smirak Index, which compares the narrowest midfoot width to the widest forefoot width
- Center of Pressure, which is the weighted average position of the load on the foot
- A fuzzy rule-based label instead of a hard yes/no answer

The pipeline is split across small files so I can inspect each part separately:
- `preprocess.py` loads CSV files, cleans weird rows, thresholds the pressure map, and separates connected foot regions
- `visualize.py` draws the heatmaps and the row-sum pressure profile
- `arch_index.py` computes AI, CSI, and CoP
- `features.py` turns one foot into a feature dictionary
- `rule_classifier.py` turns those features into a readable screening result

Why rule-based still matters:
- It is easy to explain in a viva
- It gives a clinical anchor before the ML models start guessing
- It works even when the dataset is too small for deep learning

Where it fails:
- The foot has to be oriented reasonably well
- A noisy scan can shift the bounding box
- Some feet do not fit neatly into the old threshold ranges
- It cannot learn weird patterns from data the way ML can

Run from inside `final/phase1_rule_based` or from the `final` folder with the script path:

```powershell
python preprocess.py --csv "../data/raw/subject1_normal_trial1_pressure.csv"
python arch_index.py --csv "../data/raw/subject1_normal_trial1_pressure.csv"
python visualize.py --csv "../data/raw/subject1_normal_trial1_pressure.csv"
python rule_classifier.py --ai 0.31
```

I kept a few TODOs in the code on purpose because this part grew over time and the dataset is still a bit messy.