# Data

The project uses the **UCI Human Activity Recognition Using Smartphones** dataset. It is not committed
(about 60 MB zipped). `edgefit.data.load_har()` downloads it into `data/raw/` on the first run. If the
download fails, get the zip from the [UCI page](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
and save it as `data/raw/uci_har.zip`.

| | |
|---|---|
| Subjects | 30 volunteers, aged 19–48, smartphone worn on the waist |
| Activities | walking, walking upstairs, walking downstairs, sitting, standing, laying |
| Windows | 10,299 windows of 2.56 s (50 % overlap) |
| Features | 561 engineered accelerometer/gyroscope features, scaled to [-1, 1] |

The original 70/30 train/test files are merged; the project defines its own split by subject
(subjects 25–30 are the final test set).

Source: Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L. & Parra, X. (2013). *Human Activity Recognition
Using Smartphones* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C54S4K — licensed
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
